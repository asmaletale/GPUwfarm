"""
Bastankhah–Porté-Agel Gaussian wake velocity deficit — GPU port.

FLORIS source: floris/core/wake_velocity/gauss.py
FLORIS class:  GaussVelocityDeficit

Key equations ported verbatim (see inline comments for FLORIS line refs):

  Near-wake (xR < x < x0):
      sigma_y/z = linear ramp from 0.501*D*sqrt(Ct/2) to sigma_y0/z0

  Far-wake (x >= x0):
      ky = ka * TI + kb
      sigma_y = ky * (x - x0) + sigma_y0

  Deficit amplitude:
      C = 1 - sqrt(clip(1 - Ct*cos(yaw) / (8*sigma_y*sigma_z/D²), 0, 1))

  Gaussian kernel (2D hub-height, no wind veer):
      deficit = C * exp(-(dy - delta)² / (2*sigma_y²))

GPU deviation from FLORIS:
  - Hub-height point evaluation only; FLORIS evaluates on a 3D mesh.
  - rC() helper adapted for 2D (z = hub_height everywhere → z-term vanishes).
  - ne.evaluate() replaced with CuPy broadcasting.
  - All-pairs simultaneous; FLORIS processes turbines in downstream order.
"""
from __future__ import annotations
import cupy as cp
from physics.base import BaseWakeVelocity
from config import WakeConfig


class GaussVelocityDeficit(BaseWakeVelocity):
    """
    Gaussian velocity deficit.

    Refs:
      Bastankhah & Porté-Agel, 2014, "A new analytical model for wind-turbine
      wakes," Renewable Energy.
      Bastankhah & Porté-Agel, 2016, "Experimental and theoretical study of
      wind turbine wakes in yawed conditions," J. Fluid Mech.
    """

    def __init__(self, cfg: WakeConfig) -> None:
        self.alpha = cfg.alpha
        self.beta  = cfg.beta
        self.ka    = cfg.ka
        self.kb    = cfg.kb

    # ------------------------------------------------------------------
    # Internal helpers (mirroring FLORIS rC() and gaussian_function())
    # ------------------------------------------------------------------

    @staticmethod
    def _sigma_initial(
        ct: cp.ndarray,   # (P, T_src, 1) broadcast-ready
        yaw: cp.ndarray,  # (P, T_src, 1) in FLORIS sign (–yaw_input)
        u_inf: float,
        D: float,
    ):
        """sigma_y0, sigma_z0 from FLORIS near-wake init (gauss.py lines ~50-56)."""
        sqrt_1_ct = cp.sqrt(cp.clip(1.0 - ct, 0.0, 1.0))
        uR = u_inf * ct / (2.0 * (1.0 - sqrt_1_ct + 1e-8))
        u0 = u_inf * sqrt_1_ct
        sigma_z0 = D * 0.5 * cp.sqrt(uR / (u_inf + u0 + 1e-8))
        sigma_y0 = sigma_z0 * cp.cos(yaw)
        return sigma_y0, sigma_z0, uR, u0

    @staticmethod
    def _x0(
        ct: cp.ndarray,      # (P, T_src, 1)
        yaw: cp.ndarray,     # (P, T_src, 1)
        ti: cp.ndarray,      # (P, T_src, 1) TI at source turbine
        x_i: cp.ndarray,     # (P, T_src, 1) source x
        alpha: float, beta: float, D: float,
    ) -> cp.ndarray:
        """Far-wake start distance (FLORIS gauss.py lines ~60-67)."""
        sqrt_1_ct = cp.sqrt(cp.clip(1.0 - ct, 0.0, 1.0))
        denom = cp.float32(2.0 ** 0.5) * (
            4.0 * alpha * ti + 2.0 * beta * (1.0 - sqrt_1_ct) + 1e-8
        )
        return D * cp.cos(yaw) * (1.0 + sqrt_1_ct) / denom + x_i

    @staticmethod
    def _C(
        ct: cp.ndarray,     # (P, T, T)
        yaw: cp.ndarray,    # (P, T, T) broadcast
        sigma_y: cp.ndarray,
        sigma_z: cp.ndarray,
        D: float,
    ) -> cp.ndarray:
        """
        Velocity deficit amplitude C.
        FLORIS gauss.py rC():
            C = 1 - sqrt(clip(1 - Ct*cos(yaw) / (8*sigma_y*sigma_z/D²), 0, 1))
        """
        inner = 1.0 - ct * cp.cos(yaw) / (8.0 * sigma_y * sigma_z / (D * D) + 1e-12)
        return 1.0 - cp.sqrt(cp.clip(inner, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        dx: cp.ndarray,           # (P, T_src, T_dst)
        dy: cp.ndarray,           # (P, T_src, T_dst)
        delta: cp.ndarray,        # (P, T_src, T_dst) lateral deflection
        ct: cp.ndarray,           # (P, T_src)
        ti_eff: cp.ndarray,       # (P, T_src, T_dst)
        yaw: cp.ndarray,          # (P, T_src) radians (user sign convention)
        u_inf: float,
        rotor_diameter: float,
        x_i: cp.ndarray,          # (P, T_src) source x in wind frame
    ) -> cp.ndarray:               # (P, T_src, T_dst) velocity deficit fraction
        """
        Compute Gaussian velocity deficit for all source–destination pairs.

        Returns velocity_deficit ∈ [0, 1] where 0 = no wake.
        The effective wind speed at destination j is:
            U_eff_j = U_inf * (1 - combined_deficit_j)
        after the combination step.
        """
        D = rotor_diameter

        # FLORIS applies opposite sign convention inside gauss.py
        yaw_int = -yaw   # (P, T_src)

        # Broadcast to (P, T_src, 1) for pair operations
        ct_b   = ct[:, :, None]
        yaw_b  = yaw_int[:, :, None]
        xi_b   = x_i[:, :, None]

        # TI at the source turbine (use ambient column 0 approximation)
        ti_src = ti_eff[:, :, 0:1]   # (P, T_src, 1)

        # Initial sigma and far-wake start
        sigma_y0, sigma_z0, _, _ = self._sigma_initial(ct_b, yaw_b, u_inf, D)
        x0 = self._x0(ct_b, yaw_b, ti_src, xi_b, self.alpha, self.beta, D)
        xR = xi_b   # near-wake starts at turbine location

        # ── Far-wake sigma (FLORIS gauss.py lines ~117-122) ──
        ky = self.ka * ti_eff + self.kb   # (P, T_src, T_dst)
        kz = ky
        sigma_y_fw = ky * (dx - x0) + sigma_y0
        sigma_z_fw = kz * (dx - x0) + sigma_z0
        # Use sigma0 upstream of x0
        sigma_y_fw = cp.where(dx >= x0, sigma_y_fw, sigma_y0)
        sigma_z_fw = cp.where(dx >= x0, sigma_z_fw, sigma_z0)

        # ── Near-wake sigma (linear ramp, FLORIS gauss.py lines ~88-104) ──
        ramp_denom = cp.where(cp.abs(x0 - xR) > 1.0, x0 - xR, cp.ones_like(x0))
        near_wake_ramp_up   = (dx - xR) / ramp_denom
        near_wake_ramp_down = (x0 - dx) / ramp_denom

        sigma_nw_inner = 0.501 * D * cp.sqrt(cp.clip(ct_b / 2.0, 0.0, None))
        sigma_y_nw = near_wake_ramp_down * sigma_nw_inner + near_wake_ramp_up * sigma_y0
        sigma_z_nw = near_wake_ramp_down * sigma_nw_inner + near_wake_ramp_up * sigma_z0
        # Fix upstream of xR
        sigma_y_nw = cp.where(dx >= xR, sigma_y_nw, 0.5 * D)
        sigma_z_nw = cp.where(dx >= xR, sigma_z_nw, 0.5 * D)

        # ── Select sigma by region ──
        near_wake_mask = (dx > xR + 0.1) & (dx < x0)
        far_wake_mask  = dx >= x0

        sigma_y = cp.where(near_wake_mask, sigma_y_nw, cp.where(far_wake_mask, sigma_y_fw, 0.5 * D))
        sigma_z = cp.where(near_wake_mask, sigma_z_nw, cp.where(far_wake_mask, sigma_z_fw, 0.5 * D))
        sigma_y = cp.maximum(sigma_y, 1e-3)
        sigma_z = cp.maximum(sigma_z, 1e-3)

        # ── Deficit amplitude C ──
        C = self._C(ct_b, yaw_b, sigma_y, sigma_z, D)

        # ── 2D Gaussian (hub-height; z−HH = 0, wind_veer = 0) ──
        # r² = (dy - delta)² / (2*sigma_y²)
        r_sq = (dy - delta) ** 2 / (2.0 * sigma_y ** 2)
        gaussian = cp.exp(-r_sq)

        velocity_deficit = C * gaussian

        # Zero upstream contributions (dx ≤ xR)
        downstream_mask = dx > (xR + 0.1)
        return cp.where(downstream_mask, velocity_deficit, cp.zeros_like(velocity_deficit))
