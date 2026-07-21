"""
Validation tests for CrespoHernandez turbulence model.

Cross-checks against hand-computed values using the FLORIS formula:
    TI_wake = constant * ai^ai_exp * TI_amb^initial * (dx/D)^downstream

FLORIS source: floris/core/wake_turbulence/crespo_hernandez.py
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

from gpuwfarm_core.config import WakeConfig
from gpuwfarm_core.physics.wake_turbulence.crespo_hernandez import CrespoHernandez


def _ref_ti_wake(ai, ti_amb, dx_D, cfg: WakeConfig) -> float:
    """Reference calculation from FLORIS formula."""
    return cfg.ch_constant * ai**cfg.ch_ai * ti_amb**cfg.ch_initial * dx_D**cfg.ch_downstream


class TestCrespoHernandez:

    def setup_method(self):
        self.cfg   = WakeConfig()
        self.model = CrespoHernandez(self.cfg)
        self.D     = 120.0

    def test_single_pair_at_5D(self):
        """Single upstream turbine, evaluate at 5D downstream."""
        P, T = 1, 2
        dx = cp.zeros((P, T, T), dtype=cp.float32)
        dx[0, 0, 1] = 5.0 * self.D   # turbine 0 → turbine 1

        ai = cp.full((P, T), 0.3, dtype=cp.float32)
        ti_amb = 0.06

        result = cp.asnumpy(self.model.compute(dx, ai, ti_amb, self.D))

        expected = _ref_ti_wake(0.3, 0.06, 5.0, self.cfg)
        assert abs(result[0, 0, 1] - expected) < 1e-4, \
            f"TI_wake at 5D: {result[0,0,1]:.6f} vs expected {expected:.6f}"

    def test_upstream_is_zero(self):
        """No TI added upstream (dx ≤ 0)."""
        P, T = 1, 2
        dx = cp.zeros((P, T, T), dtype=cp.float32)
        dx[0, 1, 0] = -5.0 * self.D   # turbine 1 is UPSTREAM of turbine 0

        ai = cp.full((P, T), 0.3, dtype=cp.float32)
        result = cp.asnumpy(self.model.compute(dx, ai, 0.06, self.D))

        assert result[0, 1, 0] == pytest.approx(0.0, abs=1e-6), \
            "Upstream TI contribution should be zero"

    def test_ti_increases_with_induction(self):
        """Higher axial induction → higher wake TI."""
        P, T = 1, 2
        dx_val = 6.0 * self.D
        dx = cp.zeros((P, T, T), dtype=cp.float32)
        dx[0, 0, 1] = dx_val

        ai_lo = cp.full((P, T), 0.1, dtype=cp.float32)
        ai_hi = cp.full((P, T), 0.4, dtype=cp.float32)

        ti_lo = float(cp.asnumpy(self.model.compute(dx, ai_lo, 0.06, self.D))[0, 0, 1])
        ti_hi = float(cp.asnumpy(self.model.compute(dx, ai_hi, 0.06, self.D))[0, 0, 1])

        assert ti_hi > ti_lo, "Higher induction should produce higher wake TI"

    def test_ti_decreases_downstream(self):
        """Wake TI decreases with downstream distance (downstream exponent < 0)."""
        P, T = 1, 2

        for near_D, far_D in [(3.0, 10.0), (5.0, 15.0)]:
            dx_near = cp.zeros((P, T, T), dtype=cp.float32)
            dx_near[0, 0, 1] = near_D * self.D
            dx_far = cp.zeros((P, T, T), dtype=cp.float32)
            dx_far[0, 0, 1] = far_D * self.D
            ai = cp.full((P, T), 0.3, dtype=cp.float32)

            ti_near = float(cp.asnumpy(self.model.compute(dx_near, ai, 0.06, self.D))[0, 0, 1])
            ti_far  = float(cp.asnumpy(self.model.compute(dx_far,  ai, 0.06, self.D))[0, 0, 1])
            assert ti_near > ti_far, f"TI should decrease from {near_D}D to {far_D}D"

    def test_batch_consistency(self):
        """Batch result matches single-pair result."""
        P, T = 8, 3
        dx = cp.zeros((P, T, T), dtype=cp.float32)
        dx[:, 0, 1] = 7.0 * self.D
        ai = cp.full((P, T), 0.25, dtype=cp.float32)

        result = cp.asnumpy(self.model.compute(dx, ai, 0.08, self.D))
        expected = _ref_ti_wake(0.25, 0.08, 7.0, self.cfg)
        assert np.allclose(result[:, 0, 1], expected, rtol=1e-4)
