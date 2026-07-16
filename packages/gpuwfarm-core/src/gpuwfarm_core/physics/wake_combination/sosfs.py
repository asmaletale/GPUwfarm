"""
Sum-of-squares freestream superposition (SOSFS) — GPU port.

FLORIS source: floris/core/wake_combination/sosfs.py
FLORIS class:  SOSFS

Formula (exact FLORIS):
    combined_deficit = sqrt(sum_i(deficit_i²))

Ref: Katic et al., 1986, "A simple model for cluster efficiency."

GPU deviation: np.hypot replaced with CuPy reduction.
"""
from __future__ import annotations
import cupy as cp
from gpuwfarm_core.physics.base import BaseWakeCombination


class SOSFS(BaseWakeCombination):
    """Sum-of-squares freestream superposition."""

    def combine(self, deficits: cp.ndarray) -> cp.ndarray:
        """
        Args:
            deficits: (P, T_src, T_dst)
        Returns:
            (P, T_dst) — combined deficit via RSS
        """
        return cp.sqrt(cp.sum(deficits ** 2, axis=1))
