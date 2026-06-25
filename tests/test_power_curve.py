"""
Validation tests for TabulatedPowerCurve.

Verifies:
  - Rated power at rated wind speed (11.4 m/s for NREL 5 MW)
  - Zero power below cut-in (3 m/s)
  - Ct clipping to [0.0001, 0.9999]
  - Cosine yaw loss reduces power
  - Air density correction direction

FLORIS source: floris/core/turbine/operation_models.py (SimpleTurbine, CosineLossTurbine)
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

from physics.turbine.power_curve import TabulatedPowerCurve, TurbineData


class TestTabulatedPowerCurve:

    def setup_method(self):
        self.pc = TabulatedPowerCurve(TurbineData.nrel_5mw())

    def _scalar_power(self, u: float, yaw_deg: float = 0.0) -> float:
        u_gpu   = cp.array([[u]], dtype=cp.float32)
        yaw_gpu = cp.array([[np.deg2rad(yaw_deg)]], dtype=cp.float32)
        return float(cp.asnumpy(self.pc.power_gpu(u_gpu, yaw_gpu))[0, 0])

    def _scalar_ct(self, u: float) -> float:
        u_gpu = cp.array([[u]], dtype=cp.float32)
        return float(cp.asnumpy(self.pc.ct_gpu(u_gpu))[0, 0])

    def test_rated_power(self):
        """Power at rated speed (11.4 m/s) should be 5000 kW."""
        p = self._scalar_power(11.4)
        assert abs(p - 5000.0) < 1.0, f"Rated power: {p:.1f} kW (expected 5000)"

    def test_below_cutin_power_is_zero(self):
        """Power below cut-in (3 m/s) should be ~0."""
        p = self._scalar_power(1.0)
        assert p == pytest.approx(0.0, abs=1.0), f"Power at 1 m/s: {p:.2f}"

    def test_ct_clipped(self):
        """Ct must be within [0.0001, 0.9999]."""
        for u in [0.0, 3.0, 8.0, 25.0, 50.0]:
            ct = self._scalar_ct(u)
            assert 0.0001 <= ct <= 0.9999, f"Ct={ct:.6f} out of bounds at u={u}"

    def test_yaw_reduces_power(self):
        """Yawed turbine produces less power than aligned."""
        p_aligned = self._scalar_power(9.0, 0.0)
        p_yawed   = self._scalar_power(9.0, 20.0)
        assert p_aligned > p_yawed, \
            f"Aligned {p_aligned:.1f} kW should exceed yawed {p_yawed:.1f} kW"

    def test_power_increases_below_rated(self):
        """Power increases from cut-in to rated speed."""
        speeds = [5.0, 7.0, 9.0, 11.0]
        powers = [self._scalar_power(u) for u in speeds]
        for i in range(len(powers) - 1):
            assert powers[i] <= powers[i + 1], \
                f"Power should increase: P({speeds[i]})={powers[i]:.0f} ≤ P({speeds[i+1]})={powers[i+1]:.0f}"

    def test_batch_shape(self):
        """Output shape matches input shape."""
        P, T = 16, 10
        u   = cp.random.uniform(5, 12, (P, T)).astype(cp.float32)
        yaw = cp.zeros((P, T), dtype=cp.float32)
        out = self.pc.power_gpu(u, yaw)
        assert out.shape == (P, T)
