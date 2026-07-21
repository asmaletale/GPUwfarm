"""
FarmEvaluator — orchestrates the full FLORIS-traceable physics pipeline.

Pipeline, batched over every wind condition at once:
    1.  Flatten (wd, ws) into a single `findex` axis F (WindRose.flat_conditions()),
        mirroring FLORIS's own n_findex flattening (floris/wind_data.py).
    2.  Rotate turbine coordinates to each wind-frame -> (P, F, T), then fold
        (P, F) into one batch axis B = P*F.
    3.  Compute all pairwise (dx, dy) over the B axis.
    4.  CrespoHernandez → TI_eff (B, T_src, T_dst).
    5.  GaussVelocityDeflection → delta (B, T_src, T_dst).
    6.  GaussVelocityDeficit → deficit (B, T_src, T_dst).
    7.  WakeCombination → total_deficit (B, T_dst).
    8.  TabulatedPowerCurve → power_kW (B, T_dst) → reshape (P, F, T).
    9.  AEP = sum_F(power * freq) * 8760.

All F wind conditions run as a single batched tensor op instead of a Python
loop over conditions() -- every physics kernel below is agnostic to what the
leading "batch" axis represents, so folding (P, F) -> B needs no changes to
the wake_velocity/wake_deflection/wake_turbulence/wake_combination modules.

Memory scales as O(B * T^2) = O(P * F * T^2) per pairwise tensor; for very
large population/turbine/wind-rose-resolution combinations this may need to
be chunked over F in the future -- not implemented, since current workloads
(P<=256, T<=50, F<=150) fit comfortably (tens of MB per tensor).

All tensors stay on GPU. No cp.asnumpy() inside this loop.

FLORIS source equivalents:
    floris/core/solver.py  — cc_solver() / sequential_solver()
    floris/floris_model.py — get_farm_AEP()
    floris/wind_data.py    — n_findex flattening
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

        wd_rad, ws_arr, freq_arr, ti_arr = wind_rose.flat_conditions()   # each (F,) numpy
        F = wd_rad.shape[0]
        if F == 0:
            return cp.zeros((P, T) if per_turbine else P, dtype=cp.float32)

        wd_rad_g = cp.asarray(wd_rad)   # (F,)
        ws_g     = cp.asarray(ws_arr)   # (F,)
        freq_g   = cp.asarray(freq_arr)  # (F,)
        ti_g     = cp.asarray(ti_arr)    # (F,)

        cos_w = cp.cos(wd_rad_g)   # (F,)
        sin_w = cp.sin(wd_rad_g)   # (F,)

        # 1. Rotate to wind frame for every (P, F) combination at once
        xw = x[:, None, :] * cos_w[None, :, None] + y[:, None, :] * sin_w[None, :, None]   # (P, F, T) downwind
        yw = -x[:, None, :] * sin_w[None, :, None] + y[:, None, :] * cos_w[None, :, None]   # (P, F, T) crosswind
        yaw_b = cp.broadcast_to(yaw[:, None, :], (P, F, T))                                 # (P, F, T)

        # Fold (P, F) into a single batch axis B -- every physics kernel below only
        # cares about the leading axis being "batch", not what it represents, so this
        # needs no changes to the wake_velocity/deflection/turbulence/combination modules.
        B = P * F
        xw    = xw.reshape(B, T)
        yw    = yw.reshape(B, T)
        yaw_b = yaw_b.reshape(B, T)

        ws_b   = cp.broadcast_to(ws_g[None, :], (P, F)).reshape(B)     # (B,)
        ti_b   = cp.broadcast_to(ti_g[None, :], (P, F)).reshape(B)     # (B,)

        # 2. Pairwise displacement: dx[b, i, j] = x_j - x_i in wind frame
        #    Positive dx means turbine j is downstream of turbine i.
        dx_raw = xw[:, None, :] - xw[:, :, None]   # (B, T_src, T_dst)
        dy_raw = yw[:, None, :] - yw[:, :, None]   # (B, T_src, T_dst)
        downstream_mask = dx_raw > 0.1              # (B, T, T)

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
        u_src = cp.broadcast_to(ws_b[:, None], (B, T)).copy()   # (B, T) local inflow at each source

        for _ in range(N_JACOBI_ITERS):
            # 3. Ct / axial induction from local (waked) source inflow
            ct = self.power_curve.ct_gpu(u_src)                # (B, T)
            ai = self.power_curve.axial_induction_gpu(u_src)   # (B, T)

            # 4. CrespoHernandez added TI: (B, T_src, T_dst)
            ti_added = self.turbulence_model.compute(
                dx=dx_safe,
                axial_induction=ai,
                ambient_ti=ti_b[:, None, None],   # (B,1,1) -- per-findex ambient TI
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
            ti_added_max = cp.max(ti_added, axis=1)   # (B, T_dst)
            ti_eff_per_dst = cp.sqrt(
                ti_b[:, None] ** 2 + ti_added_max ** 2
            )  # (B, T_dst)

            # Broadcast: each src turbine i uses the *source* turbine's TI for sigma
            # ti_eff_pairs[b, i, j] = TI at source turbine i
            ti_eff_pairs = cp.broadcast_to(
                ti_eff_per_dst[:, :, None], (B, T, T)
            ).copy()   # (B, T_src, T_dst) — TI at src i, repeated across dst dimension

            # 5. Wake deflection delta: (B, T_src, T_dst)
            delta = self.deflection_model.compute(
                dx=dx_safe,
                ct=ct,
                ti_eff=ti_eff_pairs,
                yaw=yaw_b,
                u_inf=u_src,
                rotor_diameter=self.D,
                # dx is already relative (xw[dst]-xw[src]), so the source sits at
                # relative position 0 here, not its absolute xw -- passing xw
                # corrupts near/far-wake boundary detection for any source not at
                # x=0 (e.g. a middle turbine in a row of 3+).
                x_i=cp.zeros_like(xw),
            )

            # 6. Velocity deficit: (B, T_src, T_dst)
            deficit = self.velocity_model.compute(
                dx=dx_safe,
                dy=dy_raw,
                delta=delta,
                ct=ct,
                ti_eff=ti_eff_pairs,
                yaw=yaw_b,
                u_inf=u_src,
                rotor_diameter=self.D,
                x_i=cp.zeros_like(xw),   # see note above deflection_model.compute()
            )

            # Apply downstream mask (deficit only counts downstream)
            deficit = cp.where(downstream_mask, deficit, cp.zeros_like(deficit))

            # 7. Wake combination → total deficit per turbine: (B, T_dst)
            # total_deficit stays a fraction of FREESTREAM (ws); only the
            # source-side quantities (Ct, sigma, u_inf) become local.
            total_deficit = self.combination_model.combine(deficit)
            total_deficit = cp.clip(total_deficit, cp.float32(0.0), cp.float32(0.95))

            # 8. New effective speed, fed back as next iteration's source inflow
            u_src = ws_b[:, None] * (cp.float32(1.0) - total_deficit)   # (B, T)

        u_eff    = u_src
        power_kw = self.power_curve.power_gpu(u_eff, yaw_b)    # (B, T) kW

        # 9. Unfold B -> (P, F) and integrate over wind conditions:
        # AEP = sum_F [ power(wd, ws) * freq(wd, ws) ] * 8760 h/yr
        power_pft = power_kw.reshape(P, F, T)
        weighted  = power_pft * freq_g[None, :, None] * cp.float32(8760.0)   # (P, F, T)

        if per_turbine:
            return cp.sum(weighted, axis=1)          # (P, T)
        return cp.sum(weighted, axis=(1, 2))          # (P,)
