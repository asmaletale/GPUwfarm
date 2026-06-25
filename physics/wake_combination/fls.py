"""
Freestream linear superposition (FLS) — GPU port.

FLORIS source: floris/core/wake_combination/fls.py
FLORIS class:  FLS

Formula (exact FLORIS):
    combined_deficit = sum_i(deficit_i)

GPU deviation: none.
"""
from __future__ import annotations
import cupy as cp
from physics.base import BaseWakeCombination


class FLS(BaseWakeCombination):
    """Linear freestream superposition."""

    def combine(self, deficits: cp.ndarray) -> cp.ndarray:
        """
        Args:
            deficits: (P, T_src, T_dst)
        Returns:
            (P, T_dst) — combined deficit via linear sum
        """
        return cp.sum(deficits, axis=1)
