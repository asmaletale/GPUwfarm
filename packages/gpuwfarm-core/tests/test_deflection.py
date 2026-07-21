"""
Validation tests for GaussVelocityDeflection.

Tests verify:
  1. No deflection for zero yaw.
  2. Positive deflection for positive yaw angle.
  3. Deflection increases with downstream distance (in far wake).
  4. Deflection sign flips with yaw sign.

FLORIS source: floris/core/wake_deflection/gauss.py
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
from gpuwfarm_core.physics.wake_deflection.gauss import GaussVelocityDeflection


D    = 120.0
UINF = 9.0
CT   = 0.8
TI   = 0.06


def _make_pair(dx_val: float, yaw_deg: float, P: int = 1):
    T = 2
    dx      = cp.zeros((P, T, T), dtype=cp.float32)
    ct      = cp.full((P, T), CT,  dtype=cp.float32)
    ti_eff  = cp.full((P, T, T), TI, dtype=cp.float32)
    yaw     = cp.zeros((P, T),      dtype=cp.float32)
    x_i     = cp.zeros((P, T),      dtype=cp.float32)

    dx[:, 0, 1] = dx_val
    yaw[:, 0]   = np.deg2rad(yaw_deg)
    return dx, ct, ti_eff, yaw, x_i


class TestGaussVelocityDeflection:

    def setup_method(self):
        self.cfg   = WakeConfig()
        self.model = GaussVelocityDeflection(self.cfg)

    def test_zero_deflection_for_zero_yaw(self):
        """No yaw → no deflection (with ad=bd=0)."""
        dx, ct, ti_eff, yaw, x_i = _make_pair(8.0 * D, 0.0)
        delta = cp.asnumpy(
            self.model.compute(dx, ct, ti_eff, yaw, UINF, D, x_i)
        )
        assert abs(delta[0, 0, 1]) < 1e-3, \
            f"Expected ~0 deflection for 0 yaw, got {delta[0,0,1]:.4f}"

    def test_positive_yaw_gives_deflection(self):
        """Non-zero yaw produces non-zero deflection."""
        dx, ct, ti_eff, yaw, x_i = _make_pair(8.0 * D, 20.0)
        delta = cp.asnumpy(
            self.model.compute(dx, ct, ti_eff, yaw, UINF, D, x_i)
        )
        assert abs(delta[0, 0, 1]) > 0.01 * D, \
            f"Expected significant deflection for yaw=20°, got {delta[0,0,1]:.4f}"

    def test_deflection_sign_flips_with_yaw(self):
        """Opposite yaw → opposite deflection."""
        dx_p, ct, ti_p, yaw_p, x_i = _make_pair(8.0 * D,  15.0)
        dx_n, _,  ti_n, yaw_n, _   = _make_pair(8.0 * D, -15.0)
        d_p = float(cp.asnumpy(
            self.model.compute(dx_p, ct, ti_p, yaw_p, UINF, D, x_i)
        )[0, 0, 1])
        d_n = float(cp.asnumpy(
            self.model.compute(dx_n, ct, ti_n, yaw_n, UINF, D, x_i)
        )[0, 0, 1])
        assert np.sign(d_p) != np.sign(d_n), \
            f"Deflection sign should flip: +yaw={d_p:.3f}, -yaw={d_n:.3f}"

    def test_deflection_increases_with_downstream_distance(self):
        """Deflection grows with downstream distance in far wake."""
        yaw = 20.0
        dists = [4.0, 8.0, 12.0]
        deflections = []
        for dist_D in dists:
            dx, ct, ti, y, x_i = _make_pair(dist_D * D, yaw)
            d = float(cp.asnumpy(
                self.model.compute(dx, ct, ti, y, UINF, D, x_i)
            )[0, 0, 1])
            deflections.append(abs(d))
        assert deflections[0] <= deflections[1] <= deflections[2], \
            f"Deflection should grow downstream: {deflections}"

    def test_no_deflection_upstream(self):
        """Upstream turbines are not deflected by downstream turbines."""
        dx, ct, ti, yaw, x_i = _make_pair(-5.0 * D, 20.0)
        delta = cp.asnumpy(
            self.model.compute(dx, ct, ti, yaw, UINF, D, x_i)
        )
        assert abs(delta[0, 0, 1]) < 1e-3, \
            "No deflection for upstream pair"
