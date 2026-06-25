"""
Abstract base classes mirroring the FLORIS BaseModel interface.

FLORIS source: floris/core/base.py

All subclasses receive CuPy arrays (not NumPy), shaped (P, T, T) or (P, T)
where P = population size (batch dimension), T = number of turbines.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import cupy as cp


class BaseWakeVelocity(ABC):
    """Computes the velocity deficit field for one upstream turbine acting on all downstream turbines."""

    @abstractmethod
    def compute(
        self,
        dx: cp.ndarray,       # (P, T, T) downstream distance
        dy: cp.ndarray,       # (P, T, T) crosswind distance
        delta: cp.ndarray,    # (P, T, T) wake centre lateral deflection
        ct: cp.ndarray,       # (P, T) thrust coefficient at each turbine
        ti_eff: cp.ndarray,   # (P, T, T) effective TI at each dst turbine from each src
        yaw: cp.ndarray,      # (P, T) yaw angle (radians)
        u_inf: float,         # free-stream wind speed (m/s)
        rotor_diameter: float,
    ) -> cp.ndarray:           # (P, T, T) velocity deficit fraction
        ...


class BaseWakeTurbulence(ABC):
    """Computes the wake-added turbulence intensity from each upstream turbine."""

    @abstractmethod
    def compute(
        self,
        dx: cp.ndarray,          # (P, T, T) downstream distance
        axial_induction: cp.ndarray,  # (P, T) axial induction at each src turbine
        ambient_ti: float,       # scalar
        rotor_diameter: float,
    ) -> cp.ndarray:              # (P, T, T) added TI from src i at dst j
        ...


class BaseWakeDeflection(ABC):
    """Computes the lateral wake-centre displacement at each downstream turbine location."""

    @abstractmethod
    def compute(
        self,
        dx: cp.ndarray,       # (P, T, T)
        ct: cp.ndarray,       # (P, T)
        ti_eff: cp.ndarray,   # (P, T, T) effective TI used for sigma
        yaw: cp.ndarray,      # (P, T) radians
        u_inf: float,
        rotor_diameter: float,
    ) -> cp.ndarray:           # (P, T, T) lateral deflection delta
        ...


class BaseWakeCombination(ABC):
    """Combines individual wake deficit contributions into a single total deficit."""

    @abstractmethod
    def combine(self, deficits: cp.ndarray) -> cp.ndarray:
        """
        Args:
            deficits: (P, T_src, T_dst) individual deficit from each src at each dst
        Returns:
            (P, T_dst) combined deficit
        """
        ...


class ProjectionOperator(ABC):
    """Maps a population of layouts to the feasible set."""

    @abstractmethod
    def project(self, pop: cp.ndarray) -> cp.ndarray:
        """
        Args:
            pop: (P, T, 2) positions [x, y] on GPU
        Returns:
            (P, T, 2) corrected positions
        """
        ...
