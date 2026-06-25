"""
Validation tests for wake combination models (SOSFS, FLS, MAX).

Verifies that each model combines deficits differently and
reproduces the exact FLORIS formulas.

FLORIS source: floris/core/wake_combination/{sosfs,fls,max}.py
"""
import numpy as np
import pytest

try:
    import cupy as cp
    _HAS_GPU = True
except Exception:
    _HAS_GPU = False

pytestmark = pytest.mark.skipif(not _HAS_GPU, reason="CuPy not available")

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from physics.wake_combination.sosfs import SOSFS
from physics.wake_combination.fls import FLS
from physics.wake_combination.max import MAX


def _deficits(d1: float, d2: float) -> cp.ndarray:
    """(1, 2, 1) deficit tensor: two sources, one destination."""
    arr = cp.zeros((1, 2, 1), dtype=cp.float32)
    arr[0, 0, 0] = d1
    arr[0, 1, 0] = d2
    return arr


class TestWakeCombination:

    def test_sosfs_formula(self):
        """SOSFS = sqrt(sum of squares) — exact FLORIS formula."""
        d1, d2 = 0.3, 0.4
        result = float(cp.asnumpy(SOSFS().combine(_deficits(d1, d2)))[0, 0])
        expected = np.hypot(d1, d2)
        assert abs(result - expected) < 1e-5, \
            f"SOSFS: {result:.6f} vs {expected:.6f}"

    def test_fls_formula(self):
        """FLS = linear sum — exact FLORIS formula."""
        d1, d2 = 0.2, 0.35
        result = float(cp.asnumpy(FLS().combine(_deficits(d1, d2)))[0, 0])
        assert abs(result - (d1 + d2)) < 1e-5

    def test_max_formula(self):
        """MAX = element-wise maximum — exact FLORIS formula."""
        d1, d2 = 0.15, 0.45
        result = float(cp.asnumpy(MAX().combine(_deficits(d1, d2)))[0, 0])
        assert abs(result - max(d1, d2)) < 1e-5

    def test_models_differ(self):
        """Three models must produce different results for two overlapping wakes."""
        d1, d2 = 0.3, 0.25
        def_arr = _deficits(d1, d2)
        r_sosfs = float(cp.asnumpy(SOSFS().combine(def_arr))[0, 0])
        r_fls   = float(cp.asnumpy(FLS().combine(def_arr))[0, 0])
        r_max   = float(cp.asnumpy(MAX().combine(def_arr))[0, 0])

        # FLS ≥ SOSFS ≥ MAX (for positive deficits)
        assert r_fls >= r_sosfs >= r_max, \
            f"Expected FLS({r_fls:.4f}) ≥ SOSFS({r_sosfs:.4f}) ≥ MAX({r_max:.4f})"

    def test_single_wake_all_models_agree(self):
        """With only one active wake, all models return the same value."""
        d = 0.35
        arr = _deficits(d, 0.0)
        r_sosfs = float(cp.asnumpy(SOSFS().combine(arr))[0, 0])
        r_fls   = float(cp.asnumpy(FLS().combine(arr))[0, 0])
        r_max   = float(cp.asnumpy(MAX().combine(arr))[0, 0])
        assert abs(r_sosfs - d) < 1e-5
        assert abs(r_fls   - d) < 1e-5
        assert abs(r_max   - d) < 1e-5

    def test_shape_output(self):
        """Output shape is (P, T_dst)."""
        P, T_src, T_dst = 4, 5, 3
        deficits = cp.random.rand(P, T_src, T_dst).astype(cp.float32)
        for model in [SOSFS(), FLS(), MAX()]:
            out = model.combine(deficits)
            assert out.shape == (P, T_dst), \
                f"{model.__class__.__name__}: expected ({P},{T_dst}), got {out.shape}"
