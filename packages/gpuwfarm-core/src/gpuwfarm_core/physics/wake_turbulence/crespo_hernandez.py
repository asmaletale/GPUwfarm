"""
Crespo-Hernandez wake-turbulence model — GPU port.

FLORIS source: floris/core/wake_turbulence/crespo_hernandez.py
FLORIS class:  CrespoHernandez

Equation (exact FLORIS formula):
    TI_wake = constant * axial_induction^ai
                        * ambient_TI^initial
                        * (delta_x / D)^downstream

Parameters follow FLORIS defaults, which differ from the original
Crespo 1996 paper.  See https://github.com/NREL/floris/issues/773.

GPU deviation from FLORIS:
    ne.evaluate() replaced with CuPy broadcasting.
    No other structural change.
"""
from __future__ import annotations
import cupy as cp
from gpuwfarm_core.physics.base import BaseWakeTurbulence
from gpuwfarm_core.config import WakeConfig


class CrespoHernandez(BaseWakeTurbulence):
    """
    Wake-added turbulence intensity from each upstream turbine.

    Ref: Crespo & Hernandez, 1996, "Turbulence characteristics in wind-turbine
         wakes", Journal of Wind Engineering and Industrial Aerodynamics.

    FLORIS defaults (initial=0.1) differ from the 1996 paper (initial=0.0325).
    The positive exponent is used for consistency with prior FLORIS versions.
    """

    def __init__(self, cfg: WakeConfig) -> None:
        self.initial    = cfg.ch_initial     # exponent on ambient TI
        self.constant   = cfg.ch_constant    # scaling constant
        self.ai         = cfg.ch_ai          # exponent on axial induction
        self.downstream = cfg.ch_downstream  # exponent on normalised downstream distance

    def compute(
        self,
        dx: cp.ndarray,             # (P, T_src, T_dst) downstream distance (m)
        axial_induction: cp.ndarray,  # (P, T_src) induction factor of each src turbine
        ambient_ti: cp.ndarray | float,  # scalar, or broadcastable to (P, 1, 1) when
                                          # ambient TI varies per batch row (e.g. per
                                          # findex in FarmEvaluator's flattened wind rose)
        rotor_diameter: float,
    ) -> cp.ndarray:                  # (P, T_src, T_dst) wake-added TI
        """
        Evaluate Crespo-Hernandez formula for all source–destination pairs.

        The downstream mask mirrors FLORIS:
            delta_x <= 0.1 → upstream, set to 1.0 for the power (avoids nan),
            then zeroed in the return mask.

        Returns the ADDED turbulence only (not combined with ambient TI).
        Combination (RSS) is performed in FarmEvaluator.
        """
        # Mask: downstream if delta_x > 0.1 (mirrors FLORIS upstream_mask logic)
        downstream_mask = dx > 0.1   # (P, T, T) bool

        # Replace non-positive dx with 1.0 to prevent nan from negative powers
        delta_x_safe = cp.where(downstream_mask, dx, cp.ones_like(dx))

        # Broadcast axial_induction from (P, T_src) to (P, T_src, T_dst)
        ai = axial_induction[:, :, None]   # (P, T_src, 1) → broadcasts to (P, T, T)

        ti_wake = (
            self.constant
            * ai ** self.ai
            * ambient_ti ** self.initial
            * (delta_x_safe / rotor_diameter) ** self.downstream
        )

        # Zero out upstream contributions
        return ti_wake * downstream_mask
