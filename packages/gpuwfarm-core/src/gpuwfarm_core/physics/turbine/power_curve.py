"""
Tabulated power curve and cosine yaw-loss model — GPU port.

FLORIS source: floris/core/turbine/operation_models.py
FLORIS classes: SimpleTurbine, CosineLossTurbine

Power: scipy interp1d on (wind_speed, power_kW) table, extended to CuPy
       via cp.interp (linear, no extrapolation → 0 outside bounds).

Thrust: similar interpolation on (wind_speed, Ct) table, clipped [0.0001, 0.9999].

Yaw loss (CosineLossTurbine):
    P_yaw = P_table(U_eff) * cos(yaw)^cosine_loss_exponent_yaw

Air density correction (SimpleTurbine):
    U_corrected = U_eff * (rho / rho_ref)^(1/3)

GPU deviation from FLORIS:
    - scipy interp1d → cp.interp (linear, same accuracy).
    - Tables uploaded to GPU once at __init__; never moved per-individual.

Built-in dataset: NREL 5 MW reference turbine (Jonkman et al. 2009)
    https://www.nrel.gov/docs/fy09osti/38060.pdf  Table 3-1 and Fig. 3-2
"""
from __future__ import annotations
import pathlib
import numpy as np
import cupy as cp
import yaml
from dataclasses import dataclass, field
from typing import Optional


# Bundled default turbine — FLORIS's own nrel_5MW.yaml (turbine_library
# format). Single source of truth; see config.py module docstring.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[6]
_DEFAULT_TURBINE_YAML = _REPO_ROOT / "examples" / "nrel_5MW.yaml"


@dataclass
class TurbineData:
    """Holds the power / Ct tables for one turbine type."""
    wind_speeds: np.ndarray          # (N,) m/s, must be sorted ascending
    power_kw:    np.ndarray          # (N,) kW
    ct_values:   np.ndarray          # (N,) thrust coefficient
    ref_air_density: float = 1.225   # kg/m³ used to generate the tables
    cosine_loss_exponent_yaw: float = 1.88  # CosineLossTurbine default

    @classmethod
    def from_turbine_yaml(cls, path: str | pathlib.Path) -> "TurbineData":
        """Parse a FLORIS turbine_library/*.yaml file's power_thrust_table."""
        with open(path) as f:
            t = yaml.safe_load(f)
        pt = t["power_thrust_table"]
        return cls(
            wind_speeds=np.array(pt["wind_speed"], dtype=np.float32),
            power_kw=np.array(pt["power"], dtype=np.float32),
            ct_values=np.array(pt["thrust_coefficient"], dtype=np.float32),
            ref_air_density=float(pt.get("ref_air_density", 1.225)),
            cosine_loss_exponent_yaw=float(pt.get("cosine_loss_exponent_yaw", 1.88)),
        )

    @classmethod
    def nrel_5mw(cls) -> "TurbineData":
        return cls.from_turbine_yaml(_DEFAULT_TURBINE_YAML)


class TabulatedPowerCurve:
    """
    GPU-resident tabulated power curve and Ct interpolant.

    Tables are uploaded once to GPU in __init__ and reused for every
    population evaluation without CPU/GPU transfer.

    FLORIS equivalent: SimpleTurbine + CosineLossTurbine
    """

    def __init__(
        self,
        turbine_data: Optional[TurbineData] = None,
        air_density: float = 1.225,
    ) -> None:
        if turbine_data is None:
            turbine_data = TurbineData.nrel_5mw()
        self.td = turbine_data
        self.air_density = air_density

        # Upload to GPU once
        self._ws_gpu  = cp.asarray(turbine_data.wind_speeds)
        self._pow_gpu = cp.asarray(turbine_data.power_kw)
        self._ct_gpu  = cp.asarray(turbine_data.ct_values)

    # ------------------------------------------------------------------
    # Air density correction (SimpleTurbine)
    # FLORIS: rotor_velocity_air_density_correction()
    # U_corrected = U * (rho / rho_ref)^(1/3)
    # ------------------------------------------------------------------

    def _density_corrected_velocity(self, u_eff: cp.ndarray) -> cp.ndarray:
        return u_eff * (self.air_density / self.td.ref_air_density) ** (1.0 / 3.0)

    # ------------------------------------------------------------------
    # Ct interpolation (SimpleTurbine.thrust_coefficient)
    # ------------------------------------------------------------------

    def ct_gpu(self, u_eff: cp.ndarray) -> cp.ndarray:
        """
        Interpolated thrust coefficient.

        Args:
            u_eff: CuPy array of any shape (effective wind speed m/s)
        Returns:
            Ct, clipped to [0.0001, 0.9999] — same as FLORIS
        """
        u_corr = self._density_corrected_velocity(u_eff)
        ct = cp.interp(u_corr, self._ws_gpu, self._ct_gpu)
        return cp.clip(ct, 0.0001, 0.9999)

    # ------------------------------------------------------------------
    # Axial induction (SimpleTurbine.axial_induction)
    # FLORIS: (1 - sqrt(1 - Ct)) / 2
    # ------------------------------------------------------------------

    def axial_induction_gpu(self, u_eff: cp.ndarray) -> cp.ndarray:
        ct = self.ct_gpu(u_eff)
        return (1.0 - cp.sqrt(cp.clip(1.0 - ct, 0.0, 1.0))) / 2.0

    # ------------------------------------------------------------------
    # Power with cosine yaw loss (CosineLossTurbine)
    # FLORIS: P = P_table(U_eff * correction) * cos(yaw)^exp
    # ------------------------------------------------------------------

    def power_gpu(self, u_eff: cp.ndarray, yaw: cp.ndarray) -> cp.ndarray:
        """
        Power in kW for each turbine in the population.

        Args:
            u_eff: (P, T) effective rotor-averaged wind speed (m/s)
            yaw:   (P, T) yaw angle (radians)
        Returns:
            (P, T) power in kW
        """
        u_corr = self._density_corrected_velocity(u_eff)
        # FLORIS uses fill_value=0.0 for out-of-bounds (SimpleTurbine.power)
        p_base = cp.interp(u_corr, self._ws_gpu, self._pow_gpu,
                           left=cp.float32(0.0), right=cp.float32(0.0))
        yaw_loss = cp.cos(yaw) ** self.td.cosine_loss_exponent_yaw
        return p_base * yaw_loss
