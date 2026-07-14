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

from config import WakeConfig, FarmConfig, TurbineConfig
from physics.wake_turbulence.crespo_hernandez import CrespoHernandez
from physics.wake_deflection.gauss import GaussVelocityDeflection
from physics.wake_velocity.gauss import GaussVelocityDeficit
from physics.wake_combination.sosfs import SOSFS
from physics.wake_combination.fls import FLS
from physics.wake_combination.max import MAX
from physics.turbine.power_curve import TabulatedPowerCurve, TurbineData
from wind.wind_rose import WindRose


_COMBINATION_CLASSES = {"SOSFS": SOSFS, "FLS": FLS, "MAX": MAX}


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

            # 3. Freestream Ct at each turbine (first-pass approximation).
            #    FLORIS sequential solver initialises with freestream Ct too.
            u_fs = cp.full((P, T), ws, dtype=cp.float32)
            ct   = self.power_curve.ct_gpu(u_fs)          # (P, T)
            ai   = self.power_curve.axial_induction_gpu(u_fs)  # (P, T)

            # 4. CrespoHernandez added TI: (P, T_src, T_dst)
            ti_added = self.turbulence_model.compute(
                dx=dx_safe,
                axial_induction=ai,
                ambient_ti=ti,
                rotor_diameter=self.D,
            )
            # Zero out upstream contributions
            ti_added = cp.where(downstream_mask, ti_added, cp.zeros_like(ti_added))

            # Per-turbine effective TI: RSS of ambient + added TI from all upstream srcs
            # For each destination turbine j: TI_j = sqrt(TI_amb² + Σ_i TI_added[i,j]²)
            ti_eff_per_dst = cp.sqrt(
                cp.float32(ti ** 2) + cp.sum(ti_added ** 2, axis=1)
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
                u_inf=float(ws),
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
                u_inf=float(ws),
                rotor_diameter=self.D,
                x_i=cp.zeros_like(xw),   # see note above deflection_model.compute()
            )

            # Apply downstream mask (deficit only counts downstream)
            deficit = cp.where(downstream_mask, deficit, cp.zeros_like(deficit))

            # 7. Wake combination → total deficit per turbine: (P, T_dst)
            total_deficit = self.combination_model.combine(deficit)
            total_deficit = cp.clip(total_deficit, cp.float32(0.0), cp.float32(0.95))

            # 8. Effective wind speed and power
            u_eff    = ws * (cp.float32(1.0) - total_deficit)    # (P, T)
            power_kw = self.power_curve.power_gpu(u_eff, yaw)    # (P, T) kW

            if per_turbine:
                total_AEP += power_kw * cp.float32(freq) * cp.float32(8760.0)
            else:
                farm_power = cp.sum(power_kw, axis=1)    # (P,) kW
                total_AEP += farm_power * cp.float32(freq) * cp.float32(8760.0)

        return total_AEP
