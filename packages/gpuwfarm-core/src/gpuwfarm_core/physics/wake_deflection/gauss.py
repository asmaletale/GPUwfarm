"""
Gauss wake deflection model — GPU port.

FLORIS source: floris/core/wake_deflection/gauss.py
FLORIS class:  GaussVelocityDeflection

Computes lateral wake-centre displacement (delta) as a function of
downstream distance.  The result is consumed by GaussVelocityDeficit
as the `delta` argument to shift the Gaussian kernel laterally.

Near-wake deflection: linear ramp from 0 to delta0 (Bastankhah 2016).
Far-wake deflection:  logarithmic formula (Bastankhah 2016 + King 2019).

GPU deviation from FLORIS:
    - ne.evaluate() replaced with CuPy broadcasting.
    - Point evaluation at hub height; no 3D grid.
    - No secondary steering (GCH vortex) — interface stub provided.
"""
from __future__ import annotations
import numpy as np
import cupy as cp
from gpuwfarm_core.physics.base import BaseWakeDeflection
from gpuwfarm_core.config import WakeConfig


class GaussVelocityDeflection(BaseWakeDeflection):
    """
    Lateral wake-centre deflection.

    Refs:
      Bastankhah & Porté-Agel, 2016, "Experimental and theoretical study of
      wind turbine wakes in yawed conditions", J. Fluid Mech.
      King et al., 2019, "Controls-oriented model for secondary steering
      effects in wind farm control."
    """

    def __init__(self, cfg: WakeConfig) -> None:
        self.alpha = cfg.alpha
        self.beta  = cfg.beta
        self.ka    = cfg.ka
        self.kb    = cfg.kb
        self.ad    = cfg.ad
        self.bd    = cfg.bd
        self.dm    = cfg.dm

    # ------------------------------------------------------------------
    # Shared pre-computation (also used by GaussVelocityDeficit)
    # ------------------------------------------------------------------

    def near_far_wake_boundary(
        self,
        ct: cp.ndarray,          # (P, T_src) thrust coefficient
        ti: cp.ndarray,          # (P, T_src, T_dst) effective TI
        yaw: cp.ndarray,         # (P, T_src) radians (FLORIS sign convention: −yaw)
        u_inf: cp.ndarray,       # (P, T_src) local inflow at the source turbine
        D: float,
        x_i: cp.ndarray,         # (P, T_src) source x-coordinate in wind frame
    ):
        """
        Returns (x0, sigma_y0, sigma_z0, uR, u0) broadcast to (P, T_src, T_dst).

        FLORIS equations (gauss.py lines for x0, sigma_y0):
            uR      = u_inf * Ct / (2*(1 - sqrt(1-Ct)))
            u0      = u_inf * sqrt(1-Ct)
            sigma_z0 = D * 0.5 * sqrt(uR / (u_inf + u0))
            sigma_y0 = sigma_z0 * cos(yaw)
            x0      = D * cos(yaw) * (1 + sqrt(1-Ct)) /
                      (sqrt(2)*(4*alpha*TI + 2*beta*(1 - sqrt(1-Ct)))) + x_i
        """
        # Broadcast from (P, T_src) → (P, T_src, 1) for pair ops
        ct_   = ct[:, :, None]     # (P, T, 1)
        yaw_  = yaw[:, :, None]    # (P, T, 1) already in FLORIS −sign
        xi_   = x_i[:, :, None]    # (P, T, 1)
        u_b   = u_inf[:, :, None]  # (P, T, 1) local inflow at source

        # ti is already (P, T_src, T_dst); take representative TI at each src
        # (we use the src's own TI row: ambient + wakes from turbines upstream of src)
        # For x0/sigma0 we use the TI at dx→0 (i.e. at the turbine itself), which is
        # the column TI[:, src, src].  Since that diagonal is 0, we take ambient TI
        # as a reasonable approximation consistent with FLORIS usage.
        ti_src = ti[:, :, 0:1]    # (P, T_src, 1) — use first col as TI of the source

        sqrt_1_ct = cp.sqrt(cp.clip(1.0 - ct_, 0.0, 1.0))
        uR = u_b * ct_ / (2.0 * (1.0 - sqrt_1_ct + 1e-8))
        u0 = u_b * sqrt_1_ct

        sigma_z0 = D * 0.5 * cp.sqrt(uR / (u_b + u0 + 1e-8))
        sigma_y0 = sigma_z0 * cp.cos(yaw_)

        sqrt2 = cp.float32(2.0 ** 0.5)
        x0 = (
            D * cp.cos(yaw_) * (1.0 + sqrt_1_ct)
            / (
                sqrt2
                * (4.0 * self.alpha * ti_src + 2.0 * self.beta * (1.0 - sqrt_1_ct) + 1e-8)
            )
        ) + xi_

        return x0, sigma_y0, sigma_z0, uR, u0

    def compute(
        self,
        dx: cp.ndarray,           # (P, T_src, T_dst) x-dist in wind frame
        ct: cp.ndarray,           # (P, T_src)
        ti_eff: cp.ndarray,       # (P, T_src, T_dst)
        yaw: cp.ndarray,          # (P, T_src) radians
        u_inf: cp.ndarray,        # (P, T_src) local inflow at the source turbine
        rotor_diameter: float,
        x_i: cp.ndarray,          # (P, T_src) source x in wind frame
    ) -> cp.ndarray:               # (P, T_src, T_dst) lateral displacement
        """
        Full FLORIS GaussVelocityDeflection.function() ported to CuPy.

        FLORIS applies opposite yaw sign convention internally.
        We mirror that: yaw_int = −yaw.
        """
        D = rotor_diameter

        # FLORIS opposite sign convention
        yaw_int = -yaw   # (P, T_src)

        x0, sigma_y0, sigma_z0, uR, u0 = self.near_far_wake_boundary(
            ct, ti_eff, yaw_int, u_inf, D, x_i
        )

        ct_   = ct[:, :, None]
        yaw_  = yaw_int[:, :, None]
        xi_   = x_i[:, :, None]
        u_b   = u_inf[:, :, None]   # (P, T, 1) local inflow at source

        # Wake expansion in far wake
        ky = self.ka * ti_eff + self.kb   # (P, T, T)
        kz = ky

        sigma_y = ky * (dx - x0) + sigma_y0
        sigma_z = kz * (dx - x0) + sigma_z0
        # Use sigma_y0/sigma_z0 upstream of x0
        sigma_y = cp.where(dx >= x0, sigma_y, sigma_y0)
        sigma_z = cp.where(dx >= x0, sigma_z, sigma_z0)

        # Auxiliary scalars for far-wake log formula
        C0 = 1.0 - u0 / (u_b + 1e-8)                  # (P, T, 1)
        M0 = C0 * (2.0 - C0)
        E0 = C0**2 - 3.0 * np.exp(1.0/12.0) * C0 + 3.0 * np.exp(1.0/3.0)

        sqrt_M0 = cp.sqrt(cp.clip(M0, 1e-12, None))

        # theta_c0: skew angle (radians)
        sqrt_1_ct_cos = cp.sqrt(cp.clip(1.0 - ct_ * cp.cos(yaw_), 0.0, 1.0))
        theta_c0 = (
            self.dm * (0.3 * cp.deg2rad(cp.rad2deg(yaw_)) / (cp.cos(yaw_) + 1e-8))
            * (1.0 - sqrt_1_ct_cos)
        )
        # delta0: initial wake deflection at x = x0
        delta0 = cp.tan(theta_c0) * (x0 - xi_)

        # Near-wake deflection (linear ramp xR → x0, xR = x_i)
        xR = xi_
        ramp_denom = cp.where(cp.abs(x0 - xR) > 1e-3, x0 - xR, cp.ones_like(x0) * 1e-3)
        delta_nw = ((dx - xR) / ramp_denom) * delta0 + (self.ad + self.bd * (dx - xi_))
        delta_nw = delta_nw * ((dx >= xR) & (dx <= x0))

        # Far-wake deflection (logarithmic formula)
        ratio = cp.sqrt(cp.clip(sigma_y * sigma_z / (sigma_y0 * sigma_z0 + 1e-12), 1e-12, None))
        ln_num = (1.6 + sqrt_M0) * (1.6 * ratio - sqrt_M0)
        ln_den = (1.6 - sqrt_M0) * (1.6 * ratio + sqrt_M0)

        # Guard against log(≤0)
        log_arg = cp.clip(ln_num / (ln_den + 1e-12), 1e-12, None)

        ky_kz_M0 = cp.clip(ky * kz * M0, 1e-12, None)
        mid_term = (
            theta_c0 * E0 / 5.2
            * cp.sqrt(cp.clip(sigma_y0 * sigma_z0 / ky_kz_M0, 0.0, None))
            * cp.log(log_arg)
        )
        delta_fw = delta0 + mid_term + (self.ad + self.bd * (dx - xi_))
        delta_fw = delta_fw * (dx > x0)

        return delta_nw + delta_fw
