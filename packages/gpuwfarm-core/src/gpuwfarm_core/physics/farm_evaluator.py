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
    # Pipeline stages
    #
    # evaluate() below is nothing but these called in order. They are public
    # and pure (nothing is cached on self) so a main script, a notebook or an
    # RL loop can run the pipeline step by step and keep every intermediate
    # tensor. See example_core.py at the repo root for the written-out form.
    # ──────────────────────────────────────────────────────────────────

    def upload_conditions(
        self, wind_rose: WindRose
    ) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
        """
        Stage 1 — flatten the wind rose to a findex axis and upload it.

        Mirrors FLORIS's own n_findex flattening (floris/wind_data.py). Bins
        below the frequency floor are dropped by WindRose.flat_conditions(), so
        F can be 0 for a degenerate rose — evaluate() handles that, and a
        hand-written pipeline must too.

        Returns:
            wd_rad, ws, freq, ti — each (F,) CuPy float32. wd_rad is in radians,
            ordered wd-outer / ws-inner.
        """
        wd_rad, ws_arr, freq_arr, ti_arr = wind_rose.flat_conditions()   # each (F,) numpy
        return (cp.asarray(wd_rad), cp.asarray(ws_arr),
                cp.asarray(freq_arr), cp.asarray(ti_arr))

    def to_wind_frame(
        self,
        pop:    cp.ndarray,
        wd_rad: cp.ndarray,
        ws:     cp.ndarray,
        ti:     cp.ndarray,
    ) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
        """
        Stage 2 — rotate to every wind frame, then fold (P, F) into one batch axis.

        Every physics kernel only cares that the leading axis is "batch", not what
        it represents, which is why folding to B = P*F needs no changes to the
        wake_velocity / wake_deflection / wake_turbulence / combination modules.

        Args:
            pop:    (P, T, 3) — x, y, yaw_rad
            wd_rad: (F,) wind directions in radians, from upload_conditions()
            ws:     (F,) wind speeds
            ti:     (F,) ambient turbulence intensities

        Returns:
            xw:    (B, T) downwind coordinate
            yw:    (B, T) crosswind coordinate
            yaw:   (B, T) yaw angle, tiled across the F axis
            ws_b:  (B,)   freestream speed of each batch row
            ti_b:  (B,)   ambient TI of each batch row
        """
        P, T, _ = pop.shape
        F = wd_rad.shape[0]
        x   = pop[:, :, 0]   # (P, T)
        y   = pop[:, :, 1]   # (P, T)
        yaw = pop[:, :, 2]   # (P, T) radians

        cos_w = cp.cos(wd_rad)   # (F,)
        sin_w = cp.sin(wd_rad)   # (F,)

        xw = x[:, None, :] * cos_w[None, :, None] + y[:, None, :] * sin_w[None, :, None]    # (P, F, T)
        yw = -x[:, None, :] * sin_w[None, :, None] + y[:, None, :] * cos_w[None, :, None]   # (P, F, T)
        yaw_b = cp.broadcast_to(yaw[:, None, :], (P, F, T))                                 # (P, F, T)

        B = P * F
        ws_b = cp.broadcast_to(ws[None, :], (P, F)).reshape(B)   # (B,)
        ti_b = cp.broadcast_to(ti[None, :], (P, F)).reshape(B)   # (B,)

        return xw.reshape(B, T), yw.reshape(B, T), yaw_b.reshape(B, T), ws_b, ti_b

    @staticmethod
    def pairwise_displacement(
        xw: cp.ndarray, yw: cp.ndarray
    ) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
        """
        Stage 3 — all-pairs displacement in the wind frame.

        dx[b, i, j] = xw[b, j] - xw[b, i], so positive dx means turbine j is
        downstream of turbine i.

        Args:
            xw, yw: (B, T) from to_wind_frame()

        Returns:
            dx:         (B, T_src, T_dst) downwind separation, floored at 1.0 m so
                        the sigma formulas never receive dx <= 0. Pass this one to
                        the wake models.
            dy:         (B, T_src, T_dst) crosswind separation, unclamped (sign matters)
            downstream: (B, T_src, T_dst) bool — j is genuinely downstream of i
        """
        dx_raw = xw[:, None, :] - xw[:, :, None]   # (B, T_src, T_dst)
        dy_raw = yw[:, None, :] - yw[:, :, None]   # (B, T_src, T_dst)
        downstream_mask = dx_raw > 0.1             # (B, T, T)

        # Numerical floor so sigma formulas never receive dx <= 0
        dx_safe = cp.maximum(dx_raw, cp.float32(1.0))
        return dx_safe, dy_raw, downstream_mask

    def effective_ti(
        self,
        ti_added:        cp.ndarray,
        dx:              cp.ndarray,
        dy:              cp.ndarray,
        downstream_mask: cp.ndarray,
        ti_b:            cp.ndarray,
    ) -> cp.ndarray:
        """
        Stage 4b — gate CrespoHernandez's added TI by area of influence, combine
        it across sources, and broadcast back to pair shape.

        Two FLORIS behaviours live here rather than in CrespoHernandez, which
        deliberately returns added TI only:

        1. Area of influence (solver.py sequential_solver): a source's added TI
           counts toward a destination only if the destination is downstream,
           laterally within 2D, and within 15D downstream. Without these, sources
           far off to the side or long downstream still added TI, over-widening
           sigma for every other turbine on non-in-line layouts (e.g. a 3x3 grid).
        2. Combination is max(), not an RSS sum across sources. FLORIS applies
           maximum(sqrt(ti_added**2 + ambient**2), running_TI) per source, so the
           strongest single wake sets the TI and contributions do not stack. Since
           ti_added >= 0, max_i sqrt(ti_added_i**2 + amb**2) equals
           sqrt(max_i(ti_added_i)**2 + amb**2), which is what is computed here.

        dx may be the floored dx from pairwise_displacement: it differs from the
        raw value only where dx_raw < 1.0 m, which is inside the 15D range gate
        either way.

        Args:
            ti_added:        (B, T_src, T_dst) raw added TI from the turbulence model
            dx, dy:          (B, T_src, T_dst) from pairwise_displacement()
            downstream_mask: (B, T_src, T_dst) bool
            ti_b:            (B,) ambient TI per batch row

        Returns:
            (B, T_src, T_dst) effective TI, where [b, i, j] is the TI *at source i*
            and is constant along the dst axis. Both gauss models rely on that:
            they read ti_eff[:, :, 0:1] as "TI at the source".
        """
        B, T, _ = ti_added.shape
        lateral_mask = cp.abs(dy) < cp.float32(2.0 * self.D)
        range_mask   = dx <= cp.float32(15.0 * self.D)
        ti_mask = downstream_mask & lateral_mask & range_mask
        ti_added = cp.where(ti_mask, ti_added, cp.zeros_like(ti_added))

        # For each turbine j: TI_j = sqrt(TI_amb**2 + max_i TI_added[i, j]**2)
        ti_added_max   = cp.max(ti_added, axis=1)                          # (B, T)
        ti_eff_per_dst = cp.sqrt(ti_b[:, None] ** 2 + ti_added_max ** 2)   # (B, T)

        # ponytail: materializes (B,T,T) though it is constant along the dst axis
        # (~54 MB per Jacobi pass at P=256/T=20). Returning the (B,T,1) view would
        # broadcast just as well -- both gauss models read ti_eff[:, :, 0:1] -- but
        # every other ti_eff use in wake_velocity/gauss.py and
        # wake_deflection/gauss.py needs checking first. Upgrade when memory binds.
        return cp.broadcast_to(ti_eff_per_dst[:, :, None], (B, T, T)).copy()

    @staticmethod
    def update_inflow(
        ws_b: cp.ndarray, total_deficit: cp.ndarray, cap: float = 0.95
    ) -> cp.ndarray:
        """
        Stage 8 — combined deficit to effective wind speed.

        total_deficit stays a fraction of FREESTREAM; only the source-side
        quantities (Ct, sigma, u_inf) become local. Feed the result back as the
        next Jacobi pass's source inflow.

        Args:
            ws_b:          (B,) freestream speed per batch row
            total_deficit: (B, T) combined deficit from the combination model
            cap:           maximum deficit fraction

        Returns:
            (B, T) effective wind speed
        """
        total_deficit = cp.clip(total_deficit, cp.float32(0.0), cp.float32(cap))
        return ws_b[:, None] * (cp.float32(1.0) - total_deficit)

    @staticmethod
    def integrate_aep(
        power_kw: cp.ndarray, freq: cp.ndarray, per_turbine: bool = False
    ) -> cp.ndarray:
        """
        Stage 9 — unfold B back to (P, F) and integrate over wind conditions:

            AEP = sum_F [ power(wd, ws) * freq(wd, ws) ] * 8760 h/yr

        P is inferred as power_kw.shape[0] // len(freq), so a hand-written
        pipeline never has to thread shape arguments around.

        Args:
            power_kw: (B, T) turbine power in kW
            freq:     (F,) bin frequencies, summing to 1

        Returns:
            (P,) AEP in kWh, or (P, T) if per_turbine
        """
        B, T = power_kw.shape
        F = freq.shape[0]
        P = B // F

        power_pft = power_kw.reshape(P, F, T)
        weighted  = power_pft * freq[None, :, None] * cp.float32(8760.0)   # (P, F, T)

        if per_turbine:
            return cp.sum(weighted, axis=1)          # (P, T)
        return cp.sum(weighted, axis=(1, 2))         # (P,)

    # ──────────────────────────────────────────────────────────────────
    # Main entry point
    # ──────────────────────────────────────────────────────────────────

    def evaluate(
        self,
        pop: cp.ndarray,
        wind_rose: WindRose,
        per_turbine: bool = False,
        zero_yaw: bool = False,
        n_jacobi: int = N_JACOBI_ITERS,
    ) -> cp.ndarray:
        """
        Compute AEP for every individual in the population.

        This is exactly the stage methods above called in order. Call them
        yourself instead when you want the intermediates — see example_core.py
        at the repo root.

        Args:
            pop:         (P, T, 3) CuPy float32 — x, y, yaw
            wind_rose:   WindRose object
            per_turbine: if True, do not sum over turbines
            zero_yaw:    caller guarantees pop[:, :, 2] is identically 0 (e.g. a
                         layout-only optimisation). Lets the deflection model be
                         skipped — see `skip_deflection` below. Never inferred
                         from the data: `cp.all(yaw == 0)` would force a device
                         sync inside the fitness loop.
            n_jacobi:    fixed-point passes for waked-source inflow. Must be >= the
                         longest downstream wake chain in the farm; see the
                         N_JACOBI_ITERS note at the top of this module.

        Returns:
            (P,) CuPy float32 AEP in kWh, or (P, T) if per_turbine=True
        """
        P, T, _ = pop.shape

        # 1. Wind rose -> device
        wd_rad, ws, freq, ti = self.upload_conditions(wind_rose)
        if wd_rad.shape[0] == 0:
            return cp.zeros((P, T) if per_turbine else P, dtype=cp.float32)

        # 2. Wind frame, with (P, F) folded into the batch axis B
        xw, yw, yaw_b, ws_b, ti_b = self.to_wind_frame(pop, wd_rad, ws, ti)

        # 3. Pairwise geometry
        dx, dy, downstream_mask = self.pairwise_displacement(xw, yw)

        # 4-8. Jacobi fixed-point solve for waked-source inflow.
        #
        # Every source turbine's Ct/axial-induction/u_inf should come from its own
        # local (possibly waked) inflow, not freestream -- otherwise an interior
        # turbine (e.g. T2 in a row of 3+) sheds a too-strong wake onto turbines
        # behind it. Because wake dependencies are strictly downstream (a DAG, not
        # a cycle), iterating "each turbine's source state <- previous iteration's
        # effective speed" converges to the exact sequential-solver fixed point in
        # `chain_depth` passes, with no per-individual sort. Iteration 0 uses
        # freestream, exactly reproducing the old one-shot approximation.
        u_src = cp.broadcast_to(ws_b[:, None], xw.shape).copy()   # (B, T)

        # dx is already relative (xw[dst] - xw[src]), so the source sits at relative
        # position 0, not at its absolute xw -- passing xw corrupts near/far-wake
        # boundary detection for any source not at x=0 (e.g. a middle turbine in a
        # row of 3+). Hoisted out of the loop: it is a loop-invariant constant.
        x_zero = cp.zeros_like(xw)

        # With yaw = 0 every deflection term vanishes identically: theta_c0 is
        # proportional to yaw (wake_deflection/gauss.py), so delta0 = mid_term = 0
        # and all that survives is the ad + bd*dx offset. When those are zero too —
        # the FLORIS default — the model just fills an all-zero (B, T, T) tensor. A
        # Python scalar broadcasts into the deficit's (dy - delta) instead, with no
        # allocation and no 54 MB read per Jacobi pass: measured ~40% off
        # evaluate() at P=256, T=20.
        skip_deflection = zero_yaw and self.wake_cfg.ad == 0.0 and self.wake_cfg.bd == 0.0

        for _ in range(n_jacobi):
            # 4. Ct / axial induction from local (waked) source inflow
            ct = self.power_curve.ct_gpu(u_src)                  # (B, T)
            ai = self.power_curve.axial_induction_from_ct(ct)    # (B, T)

            # 5. CrespoHernandez added TI, then area-of-influence gating + max()
            ti_added = self.turbulence_model.compute(
                dx=dx,
                axial_induction=ai,
                ambient_ti=ti_b[:, None, None],   # (B,1,1) -- per-findex ambient TI
                rotor_diameter=self.D,
            )
            ti_eff = self.effective_ti(ti_added, dx, dy, downstream_mask, ti_b)

            # 6. Wake deflection: (B, T_src, T_dst), or a broadcastable scalar 0
            if skip_deflection:
                delta = cp.float32(0.0)
            else:
                delta = self.deflection_model.compute(
                    dx=dx, ct=ct, ti_eff=ti_eff, yaw=yaw_b, u_inf=u_src,
                    rotor_diameter=self.D, x_i=x_zero,
                )

            # 7. Velocity deficit: (B, T_src, T_dst)
            deficit = self.velocity_model.compute(
                dx=dx, dy=dy, delta=delta, ct=ct, ti_eff=ti_eff, yaw=yaw_b,
                u_inf=u_src, rotor_diameter=self.D, x_i=x_zero,
            )
            deficit = cp.where(downstream_mask, deficit, cp.zeros_like(deficit))

            # 8. Combine, then feed back as the next pass's source inflow
            u_src = self.update_inflow(ws_b, self.combination_model.combine(deficit))

        power_kw = self.power_curve.power_gpu(u_src, yaw_b)    # (B, T) kW

        # 9. Integrate over the wind rose
        return self.integrate_aep(power_kw, freq, per_turbine=per_turbine)
