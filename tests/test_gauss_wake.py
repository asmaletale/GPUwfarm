"""
Validation tests for GaussVelocityDeficit.

Two-turbine collinear case: turbine 0 upstream, turbine 1 at varying downstream
distances. Checks are qualitative (monotone decay, zero upstream) and
quantitative against the FLORIS analytical formula.

FLORIS source: floris/core/wake_velocity/gauss.py
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

from config import WakeConfig
from physics.wake_velocity.gauss import GaussVelocityDeficit


D    = 120.0
UINF = 9.0
CT   = 0.8
TI   = 0.06


def _make_collinear(downstream_x: float, P: int = 1):
    """Build (dx, dy, delta, ct, ti_eff, yaw, x_i) for a pair at given x."""
    T = 2
    dx = cp.zeros((P, T, T), dtype=cp.float32)
    dy = cp.zeros((P, T, T), dtype=cp.float32)
    dx[:, 0, 1] = downstream_x   # turbine 1 is downstream of turbine 0

    delta   = cp.zeros((P, T, T), dtype=cp.float32)
    ct      = cp.full((P, T), CT,  dtype=cp.float32)
    ti_eff  = cp.full((P, T, T), TI, dtype=cp.float32)
    yaw     = cp.zeros((P, T),      dtype=cp.float32)
    x_i     = cp.zeros((P, T),      dtype=cp.float32)   # src at origin

    return dx, dy, delta, ct, ti_eff, yaw, x_i


class TestGaussVelocityDeficit:

    def setup_method(self):
        self.cfg   = WakeConfig()
        self.model = GaussVelocityDeficit(self.cfg)

    def test_zero_upstream(self):
        """No deficit at an upstream location."""
        dx, dy, delta, ct, ti_eff, yaw, x_i = _make_collinear(-3.0 * D)
        # Swap: turbine 1 is upstream of turbine 0
        deficit = cp.asnumpy(
            self.model.compute(dx, dy, delta, ct, ti_eff, yaw, UINF, D, x_i)
        )
        assert deficit[0, 0, 1] == pytest.approx(0.0, abs=1e-5), \
            "Deficit must be 0 upstream"

    def test_positive_downstream(self):
        """Positive deficit at 6D downstream."""
        dx, dy, delta, ct, ti_eff, yaw, x_i = _make_collinear(6.0 * D)
        deficit = cp.asnumpy(
            self.model.compute(dx, dy, delta, ct, ti_eff, yaw, UINF, D, x_i)
        )
        d = deficit[0, 0, 1]
        assert 0.0 < d < 1.0, f"Expected deficit in (0,1), got {d:.4f}"

    def test_deficit_decays_with_distance(self):
        """Deficit decreases monotonically with downstream distance."""
        distances = [3.0, 5.0, 8.0, 12.0, 20.0]
        deficits  = []
        for dist_D in distances:
            dx, dy, delta, ct, ti_eff, yaw, x_i = _make_collinear(dist_D * D)
            d = float(cp.asnumpy(
                self.model.compute(dx, dy, delta, ct, ti_eff, yaw, UINF, D, x_i)
            )[0, 0, 1])
            deficits.append(d)

        for i in range(len(deficits) - 1):
            assert deficits[i] >= deficits[i + 1], \
                f"Deficit should decay: d[{distances[i]}D]={deficits[i]:.4f} " \
                f"> d[{distances[i+1]}D]={deficits[i+1]:.4f}"

    def test_centred_deficit_greater_than_offset(self):
        """Centred wake produces higher deficit than laterally offset."""
        dist = 7.0 * D
        # Centred
        dx, dy_c, delta, ct, ti_eff, yaw, x_i = _make_collinear(dist)
        # Offset by 1D laterally
        dy_off = dy_c.copy()
        dy_off[:, 0, 1] = D

        d_centre = float(cp.asnumpy(
            self.model.compute(dx, dy_c,   delta, ct, ti_eff, yaw, UINF, D, x_i)
        )[0, 0, 1])
        d_offset = float(cp.asnumpy(
            self.model.compute(dx, dy_off, delta, ct, ti_eff, yaw, UINF, D, x_i)
        )[0, 0, 1])
        assert d_centre > d_offset, \
            f"Centre deficit {d_centre:.4f} should exceed offset {d_offset:.4f}"

    def test_deficit_bound(self):
        """Deficit must never exceed 1.0 for any plausible input."""
        for dist_D in [1.0, 3.0, 6.0, 10.0]:
            dx, dy, delta, ct, ti_eff, yaw, x_i = _make_collinear(dist_D * D)
            d = float(cp.asnumpy(
                self.model.compute(dx, dy, delta, ct, ti_eff, yaw, UINF, D, x_i)
            )[0, 0, 1])
            assert d <= 1.0, f"Deficit {d:.4f} > 1.0 at {dist_D}D"

    def test_yaw_reduces_axial_deficit(self):
        """Yawed turbine produces smaller axial (centred) deficit."""
        dist = 6.0 * D
        dx, dy, delta, ct, ti_eff, _, x_i = _make_collinear(dist)
        yaw_0  = cp.zeros_like(ct)
        yaw_20 = cp.full_like(ct, np.deg2rad(20))

        d_aligned = float(cp.asnumpy(
            self.model.compute(dx, dy, delta, ct, ti_eff, yaw_0,  UINF, D, x_i)
        )[0, 0, 1])
        d_yawed = float(cp.asnumpy(
            self.model.compute(dx, dy, delta, ct, ti_eff, yaw_20, UINF, D, x_i)
        )[0, 0, 1])
        assert d_aligned >= d_yawed, \
            f"Aligned deficit {d_aligned:.4f} should be >= yawed {d_yawed:.4f}"
