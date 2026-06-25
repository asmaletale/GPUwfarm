"""
Maximum wake deficit superposition (MAX) — GPU port.

FLORIS source: floris/core/wake_combination/max.py
FLORIS class:  MAX

Formula (exact FLORIS):
    combined_deficit = max_i(deficit_i)

Ref: Gunn & Stock-Williams, 2016, "Limitations to the validity of single wake
     superposition in wind farm yield assessment."

GPU deviation: np.maximum replaced with CuPy reduction.
"""
from __future__ import annotations
import cupy as cp
from physics.base import BaseWakeCombination


class MAX(BaseWakeCombination):
    """Maximum wake deficit superposition."""

    def combine(self, deficits: cp.ndarray) -> cp.ndarray:
        """
        Args:
            deficits: (P, T_src, T_dst)
        Returns:
            (P, T_dst) — combined deficit via element-wise maximum
        """
        return cp.max(deficits, axis=1)
