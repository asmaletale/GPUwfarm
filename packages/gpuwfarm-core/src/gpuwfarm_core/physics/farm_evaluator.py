"""
FarmEvaluator — orchestrates the full FLORIS-traceable physics pipeline.

Pipeline per wind condition (wd, ws):
    1.  Rotate turbine coordinates to wind frame.
    2.  Compute all pairwise (dx, dy).
    3.  CrespoHernandez → TI_eff (P, T_src, T_dst).
    4.  GaussVelocityDeflection → delta (P, T_src, T_dst).
    5.  GaussVelocityDeficit → deficit (P, T_src, T_dst).
    6.  WakeCombination → total_deficit (P, T_dst).
    7.  TabulatedPowerCurve → power_kW (P, T_dst).
    8.  AEP += sum(power) * freq * 8760.

All tensors stay on GPU. No cp.asnumpy() inside this loop.

FLORIS source equivalents:
    floris/core/solver.py  — cc_solver() / sequential_solver()
    floris/floris_model.py — get_farm_AEP()
"""
from __future__ import annotations
import numpy as np
import cupy as cp

from gpuwfarm_core.config import WakeConfig, FarmConfig, TurbineConfig
from gpuwfarm_core.physics.wake_turbulence.crespo_hernandez import CrespoHernandez
from gpuwfarm_core.physics.wake_deflection.gauss import GaussVelocityDeflection
from gpuwfarm_core.physics.wake_velocity.gauss import GaussVelocityDeficit
from gpuwfarm_core.physics.wake_combination.sosfs import SOSFS
from gpuwfarm_core.physics.wake_combination.fls import FLS
from gpuwfarm_core.physics.wake_combination.max import MAX
from gpuwfarm_core.physics.turbine.power_curve import TabulatedPowerCurve, TurbineData
from gpuwfarm_core.wind.wind_rose import WindRose


_COMBINATION_CLASSES = {"SOSFS": SOSFS, "FLS": FLS, "MAX": MAX}

# Jacobi fixed-point passes for waked-source inflow (see farm_evaluator.evaluate()).
# Must cover the longest downstream wake chain in the farm; 3 covers rows up to
# depth 3, bump for deeper/denser layouts. Not a physics parameter -- purely a
# solver convergence knob, so it does not come from the FLORIS YAML config.
N_JACOBI_ITERS = 3


class FarmEvaluator:
    """
    Vectorised batch farm evaluator for the genetic algorithm population.

    Accepts pop: (P, T, 3) — [x, y, yaw_rad] on GPU.
    Returns AEP: (P,) — annual energy production in kWh.
    """

    def __init__(
        self,
        farm_cfg:    FarmConfig,
        turbine_cfg: TurbineConfig,
        wake_cfg:    WakeConfig,
        turbine_data: TurbineData | None = None,
    ) -> None:
        self.farm_cfg    = farm_cfg
        self.turbine_cfg = turbine_cfg
        self.wake_cfg    = wake_cfg

        self.turbulence_model  = CrespoHernandez(wake_cfg)
        self.deflection_model  = GaussVelocityDeflection(wake_cfg)
        self.velocity_model    = GaussVelocityDeficit(wake_cfg)
        self.combination_model = _COMBINATION_CLASSES[wake_cfg.combination]()
        self.power_curve       = TabulatedPowerCurve(turbine_data, farm_cfg.air_density)

        self.D  = turbine_cfg.rotor_diameter
        self.HH = turbine_cfg.hub_height

    # ──────────────────────────────────────────────────────────────────
    # Main entry point
    # ──────────────────────────────────────────────────────────────────

    def evaluate(self, pop: cp.ndarray, wind_rose: WindRose, per_turbine: bool = False) -> cp.ndarray:
        """
        Compute AEP for every individual in the population.

        Args:
            pop:         (P, T, 3) CuPy float32 — x, y, yaw
            wind_rose:   WindRose object
            per_turbine: if True, do not sum over turbines

        Returns:
            (P,) CuPy float32 AEP in kWh, or (P, T) if per_turbine=True
        """
        P, T, _ = pop.shape
        x   = pop[:, :, 0]   # (P, T)
        y   = pop[:, :, 1]   # (P, T)
        yaw = pop[:, :, 2]   # (P, T) radians

        total_AEP = cp.zeros((P, T) if per_turbine else P, dtype=cp.float32)

        for wd_rad, ws_float, freq, ti_float in wind_rose.conditions():
            if freq < 1e-9:
                continue

            cos_w = float(np.cos(wd_rad))
            sin_w = float(np.sin(wd_rad))
            ws    = cp.float32(ws_float)
            ti    = float(ti_float)

            # 1. Rotate to wind frame
            xw = x * cos_w + y * sin_w    # (P, T) downwind
            yw = -x * sin_w + y * cos_w   # (P, T) crosswind

            # 2. Pairwise displacement: dx[p, i, j] = x_j - x_i in wind frame
            #    Positive dx means turbine j is downstream of turbine i.
            dx_raw = xw[:, None, :] - xw[:, :, None]   # (P, T_src, T_dst)
            dy_raw = yw[:, None, :] - yw[:, :, None]   # (P, T_src, T_dst)
            downstream_mask = dx_raw > 0.1              # (P, T, T)

            # Numerical floor so sigma formulas never receive dx ≤ 0
            dx_safe = cp.maximum(dx_raw, cp.float32(1.0))

            # 3-8. Jacobi fixed-point solve for waked-source inflow.
            #
            # Every source turbine's Ct/axial-induction/u_inf should come from its own
            # local (possibly waked) inflow, not freestream -- otherwise an interior
            # turbine (e.g. T2 in a row of 3+) sheds a too-strong wake onto turbines
            # behind it. Because wake dependencies are strictly downstream (a DAG, not
            # a cycle), iterating "each turbine's source state <- previous iteration's
            # effective speed" converges to the exact sequential-solver fixed point in
            # `chain_depth` passes, with no per-individual sort. Iteration 0 uses
            # freestream, exactly reproducing the old one-shot approximation.
            u_src = cp.full((P, T), ws, dtype=cp.float32)   # (P, T) local inflow at each source

            for _ in range(N_JACOBI_ITERS):
                # 3. Ct / axial induction from local (waked) source inflow
                ct = self.power_curve.ct_gpu(u_src)                # (P, T)
                ai = self.power_curve.axial_induction_gpu(u_src)   # (P, T)

                # 4. CrespoHernandez added TI: (P, T_src, T_dst)
                ti_added = self.turbulence_model.compute(
                    dx=dx_safe,
                    axial_induction=ai,
                    ambient_ti=ti,
                    rotor_diameter=self.D,
                )
                # FLORIS (solver.py sequential_solver) only lets a source's added TI
                # count toward a destination turbine if it is downstream, laterally
                # within 2D of the source, and within 15D downstream (the wake-added-
                # turbulence "area of influence"). Without these, sources far off to
                # the side or long downstream still added TI, over-widening sigma for
                # every other turbine on non-in-line layouts (e.g. a 3x3 grid).
                lateral_mask = cp.abs(dy_raw) < cp.float32(2.0 * self.D)
                range_mask   = dx_raw <= cp.float32(15.0 * self.D)
                ti_mask = downstream_mask & lateral_mask & range_mask
                ti_added = cp.where(ti_mask, ti_added, cp.zeros_like(ti_added))

                # Per-turbine effective TI: FLORIS combines multiple upstream sources
                # via max(), not RSS sum (solver.py: `np.maximum(sqrt(ti_added**2 +
                # ambient**2), running_TI)`, applied per source) -- the strongest single
                # wake sets the TI, it doesn't stack additively across sources. Since
                # ti_added >= 0, max_i sqrt(ti_added_i**2 + amb**2) == sqrt(max_i(ti_added_i)**2 + amb**2).
                # For each destination turbine j: TI_j = sqrt(TI_amb² + max_i TI_added[i,j]²)
                ti_added_max = cp.max(ti_added, axis=1)   # (P, T_dst)
                ti_eff_per_dst = cp.sqrt(
                    cp.float32(ti ** 2) + ti_added_max ** 2
                )  # (P, T_dst)

                # Broadcast: each src turbine i uses the *source* turbine's TI for sigma
                # ti_eff_pairs[p, i, j] = TI at source turbine i
                ti_eff_pairs = cp.broadcast_to(
                    ti_eff_per_dst[:, :, None], (P, T, T)
                ).copy()   # (P, T_src, T_dst) — TI at src i, repeated across dst dimension

                # 5. Wake deflection delta: (P, T_src, T_dst)
                delta = self.deflection_model.compute(
                    dx=dx_safe,
                    ct=ct,
                    ti_eff=ti_eff_pairs,
                    yaw=yaw,
                    u_inf=u_src,
                    rotor_diameter=self.D,
                    # dx is already relative (xw[dst]-xw[src]), so the source sits at
                    # relative position 0 here, not its absolute xw -- passing xw
                    # corrupts near/far-wake boundary detection for any source not at
                    # x=0 (e.g. a middle turbine in a row of 3+).
                    x_i=cp.zeros_like(xw),
                )

                # 6. Velocity deficit: (P, T_src, T_dst)
                deficit = self.velocity_model.compute(
                    dx=dx_safe,
                    dy=dy_raw,
                    delta=delta,
                    ct=ct,
                    ti_eff=ti_eff_pairs,
                    yaw=yaw,
                    u_inf=u_src,
                    rotor_diameter=self.D,
                    x_i=cp.zeros_like(xw),   # see note above deflection_model.compute()
                )

                # Apply downstream mask (deficit only counts downstream)
                deficit = cp.where(downstream_mask, deficit, cp.zeros_like(deficit))

                # 7. Wake combination → total deficit per turbine: (P, T_dst)
                # total_deficit stays a fraction of FREESTREAM (ws); only the
                # source-side quantities (Ct, sigma, u_inf) become local.
                total_deficit = self.combination_model.combine(deficit)
                total_deficit = cp.clip(total_deficit, cp.float32(0.0), cp.float32(0.95))

                # 8. New effective speed, fed back as next iteration's source inflow
                u_src = ws * (cp.float32(1.0) - total_deficit)   # (P, T)

            u_eff    = u_src
            power_kw = self.power_curve.power_gpu(u_eff, yaw)    # (P, T) kW

            if per_turbine:
                total_AEP += power_kw * cp.float32(freq) * cp.float32(8760.0)
            else:
                farm_power = cp.sum(power_kw, axis=1)    # (P,) kW
                total_AEP += farm_power * cp.float32(freq) * cp.float32(8760.0)

        return total_AEP
