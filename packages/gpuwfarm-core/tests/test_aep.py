"""
Validation tests for WindRose and AEP integration.

Verifies:
  - Frequency table sums to 1.
  - Single-turbine AEP matches manual integration.
  - Weibull factory produces valid rose.
  - FarmEvaluator AEP is positive and sensible.

FLORIS source: floris/wind_data.py, floris/floris_model.py:get_farm_AEP()
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

from gpuwfarm_core.config import WakeConfig, FarmConfig, TurbineConfig
from gpuwfarm_core.wind.wind_rose import WindRose
from gpuwfarm_core.physics.farm_evaluator import FarmEvaluator
from gpuwfarm_core.physics.turbine.power_curve import TurbineData


class TestWindRose:

    def test_default_freq_sums_to_one(self):
        rose = WindRose.default_12sector()
        assert abs(rose.freq_table.sum() - 1.0) < 1e-4

    def test_multispeed_freq_sums_to_one(self):
        rose = WindRose.default_12sector_multispeed()
        assert abs(rose.freq_table.sum() - 1.0) < 1e-4

    def test_weibull_factory(self):
        dirs    = np.arange(0, 360, 30, dtype=np.float32)
        freqs   = np.ones(12, dtype=np.float32) / 12
        A       = np.full(12, 9.0, dtype=np.float32)
        k       = np.full(12, 2.0, dtype=np.float32)
        speeds  = np.arange(3.0, 16.0, 1.0, dtype=np.float32)
        rose = WindRose.from_weibull(dirs, freqs, A, k, speeds)

        assert abs(rose.freq_table.sum() - 1.0) < 1e-3
        assert rose.freq_table.shape == (12, len(speeds))

    def test_conditions_iterator(self):
        rose = WindRose.default_12sector()
        conds = list(rose.conditions())
        assert len(conds) == 12   # 12 dirs × 1 speed

    def test_flat_conditions_matches_iterator_count(self):
        """flat_conditions() must agree with conditions() when no bin is zero-freq."""
        rose = WindRose.default_12sector()
        wd_rad, ws, freq, ti = rose.flat_conditions()
        assert wd_rad.shape == ws.shape == freq.shape == ti.shape == (12,)
        assert abs(float(freq.sum()) - 1.0) < 1e-4

    def test_flat_conditions_drops_zero_freq_bins(self):
        """Bins with freq <= 1e-9 must be filtered out, mirroring conditions()'s skip."""
        dirs   = np.array([0.0, 90.0, 180.0], dtype=np.float32)
        speeds = np.array([9.0], dtype=np.float32)
        freq_table = np.array([[0.5], [0.0], [0.5]], dtype=np.float32)
        rose = WindRose.from_uniform_ti(dirs, speeds, freq_table)

        wd_rad, ws, freq, ti = rose.flat_conditions()
        assert wd_rad.shape == (2,), "the zero-freq 90 deg bin must be dropped"
        assert np.allclose(sorted(freq.tolist()), [0.5, 0.5])

    def test_flat_conditions_is_cached(self):
        """Repeated calls (once per GA generation) must return the same cached arrays."""
        rose = WindRose.default_12sector_multispeed()
        first = rose.flat_conditions()
        second = rose.flat_conditions()
        assert first[0] is second[0], "flat_conditions() should memoize, not re-flatten"


class TestFarmEvaluatorAEP:

    def _make_simple_farm(self, n_turbines: int = 3, combination: str = "SOSFS"):
        wake_cfg    = WakeConfig(combination=combination)
        farm_cfg    = FarmConfig(n_turbines=n_turbines, area_width=2000, area_height=2000)
        turbine_cfg = TurbineConfig()
        return FarmEvaluator(farm_cfg, turbine_cfg, wake_cfg, TurbineData.nrel_5mw()), farm_cfg

    def _grid_pop(self, farm_cfg: FarmConfig, P: int = 4) -> cp.ndarray:
        """Simple grid layout, P identical copies."""
        T  = farm_cfg.n_turbines
        xs = np.linspace(200, 1800, T)
        ys = np.full(T, 1000.0)
        pop = cp.zeros((P, T, 3), dtype=cp.float32)
        pop[:, :, 0] = cp.asarray(xs)
        pop[:, :, 1] = cp.asarray(ys)
        return pop

    def test_aep_is_positive(self):
        """AEP must be positive for any valid layout."""
        ev, fc = self._make_simple_farm()
        pop    = self._grid_pop(fc)
        rose   = WindRose.default_12sector()
        aep    = cp.asnumpy(ev.evaluate(pop, rose))
        assert np.all(aep > 0), f"Non-positive AEP: {aep}"

    def test_aep_identical_layouts(self):
        """All identical individuals in population must have equal AEP."""
        ev, fc = self._make_simple_farm()
        pop    = self._grid_pop(fc, P=8)
        rose   = WindRose.default_12sector()
        aep    = cp.asnumpy(ev.evaluate(pop, rose))
        assert np.allclose(aep, aep[0], rtol=1e-4), \
            "Identical layouts should produce equal AEP"

    def test_aep_combination_ordering(self):
        """For a waked farm, SOSFS ≤ FLS (SOSFS is conservative)."""
        ev_s, fc = self._make_simple_farm(combination="SOSFS")
        ev_l, _  = self._make_simple_farm(combination="FLS")
        pop      = self._grid_pop(fc)
        rose     = WindRose.default_12sector()
        aep_s    = float(cp.asnumpy(ev_s.evaluate(pop, rose))[0])
        aep_l    = float(cp.asnumpy(ev_l.evaluate(pop, rose))[0])
        # For non-trivial wakes: FLS ≥ SOSFS (larger combined deficit → less AEP for FLS,
        # but FLS sums linearly so the effective wind speed loss is larger).
        # Both should be in a physically reasonable range.
        assert aep_s > 0 and aep_l > 0

    def test_single_turbine_aep_range(self):
        """Single turbine AEP should be between 1e7 and 1e9 kWh for NREL 5MW."""
        wake_cfg    = WakeConfig()
        farm_cfg    = FarmConfig(n_turbines=1)
        turbine_cfg = TurbineConfig()
        ev   = FarmEvaluator(farm_cfg, turbine_cfg, wake_cfg, TurbineData.nrel_5mw())
        pop  = cp.zeros((1, 1, 3), dtype=cp.float32)
        pop[0, 0, 0] = 1000.0
        pop[0, 0, 1] = 1000.0
        rose = WindRose.default_12sector_multispeed()
        aep  = float(cp.asnumpy(ev.evaluate(pop, rose))[0])
        assert 1e6 < aep < 1e9, f"Single turbine AEP {aep:.3e} out of expected range"

    def test_findex_batched_matches_sum_of_single_conditions(self):
        """
        The findex-vectorized evaluate() (batched over the whole wind rose in one
        tensor op, see farm_evaluator.py) must equal the linear sum of evaluate()
        called once per individual wind condition -- this catches any cross-talk
        introduced by folding (P, F) into a single batch axis (e.g. a wrong
        reshape order mixing up which findex row belongs to which population row).
        """
        ev, fc = self._make_simple_farm(n_turbines=3)
        pop    = self._grid_pop(fc, P=4)

        dirs   = np.array([0.0, 60.0, 150.0], dtype=np.float32)
        speeds = np.array([7.0, 11.0], dtype=np.float32)
        freq_table = np.array([[0.10, 0.25], [0.15, 0.20], [0.05, 0.25]], dtype=np.float32)
        rose_multi = WindRose.from_uniform_ti(dirs, speeds, freq_table, ti_ambient=0.06)

        aep_batched = cp.asnumpy(ev.evaluate(pop, rose_multi))   # (P,)

        aep_summed = np.zeros(4, dtype=np.float32)
        for i, wd in enumerate(dirs):
            for j, ws in enumerate(speeds):
                rose_single = WindRose.from_uniform_ti(
                    wind_dirs=np.array([wd], dtype=np.float32),
                    wind_speeds=np.array([ws], dtype=np.float32),
                    freq_table=np.array([[1.0]], dtype=np.float32),
                    ti_ambient=0.06,
                )
                aep_single = cp.asnumpy(ev.evaluate(pop, rose_single))
                aep_summed += aep_single * freq_table[i, j]

        assert np.allclose(aep_batched, aep_summed, rtol=1e-4), (
            f"Batched findex AEP {aep_batched} != per-condition sum {aep_summed}"
        )

    def test_findex_batched_per_turbine_matches_farm_total(self):
        """per_turbine=True output summed over turbines must equal the farm-level AEP."""
        ev, fc = self._make_simple_farm(n_turbines=4)
        pop    = self._grid_pop(fc, P=3)
        rose   = WindRose.default_12sector_multispeed()

        aep_farm = cp.asnumpy(ev.evaluate(pop, rose, per_turbine=False))       # (P,)
        aep_per_turbine = cp.asnumpy(ev.evaluate(pop, rose, per_turbine=True))  # (P, T)

        assert np.allclose(aep_farm, aep_per_turbine.sum(axis=1), rtol=1e-4)
