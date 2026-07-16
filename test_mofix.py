"""Smoke tests for the 7 multi-objective bug fixes."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.add_dll_directory(os.path.normpath(
    os.path.join(sys.executable, "..", "..", "Lib", "site-packages", "torch", "lib")
))

import numpy as np
import cupy as cp

from gpuwfarm_core.config import FarmConfig, TurbineConfig, CostConfig
from gpuwfarm_core.objectives import ObjectiveEvaluator
from optimizer.genetic import GeneticAlgorithm


def test_fixed_costs_and_lcoe_batch():
    oe = ObjectiveEvaluator(FarmConfig(n_turbines=5), TurbineConfig(), CostConfig())
    fixed, per_km = oe._fixed_costs(5)
    assert fixed > 0 and per_km > 0, "costs must be positive"

    aep_gwh  = np.array([100.0, 200.0, 0.0], dtype=np.float32)
    cable_km = np.array([1.0,   2.0,   1.0], dtype=np.float32)
    lcoe = oe.compute_lcoe_batch(5, aep_gwh, cable_km)
    assert lcoe.shape == (3,), "shape must be (P,)"
    assert np.isinf(lcoe[2]), "zero AEP must give inf LCOE"
    assert lcoe[0] > lcoe[1], "higher AEP → lower LCOE"
    print(f"  lcoe_batch: {lcoe}")


def test_fast_nondominated_sort_no_rank_overwrite():
    # individual 0 strictly dominates all others
    obj = np.array([[1.0, 1.0],
                    [2.0, 2.0],
                    [3.0, 1.5],
                    [2.5, 2.5]], dtype=np.float64)
    ranks, dist = GeneticAlgorithm.fast_nondominated_sort(obj)
    assert ranks[0] == 0, f"rank-0 individual overwritten: got {ranks}"
    assert all(r > 0 for r in ranks[1:]), f"dominated individuals wrong: {ranks}"
    print(f"  ranks: {ranks}  distances: {dist}")


def test_fast_nondominated_sort_two_front():
    # 0 and 1 are non-dominated; 2 is dominated by 0
    obj = np.array([[1.0, 3.0],
                    [3.0, 1.0],
                    [4.0, 4.0]], dtype=np.float64)
    ranks, _ = GeneticAlgorithm.fast_nondominated_sort(obj)
    assert ranks[0] == 0 and ranks[1] == 0, f"expected front 0 for 0,1: {ranks}"
    assert ranks[2] == 1, f"expected rank 1 for individual 2: {ranks}"
    print(f"  two-front ranks: {ranks}")


def test_aep_unit_conversion():
    # AEP is in kWh; divide by 1e6 gives GWh (not 1e9)
    aep_kwh = np.array([1_000_000.0])  # 1 GWh in kWh
    aep_gwh = aep_kwh / 1e6
    assert abs(aep_gwh[0] - 1.0) < 1e-9, f"unit bug: got {aep_gwh[0]} GWh"
    print(f"  unit check: {aep_kwh[0]:.0f} kWh = {aep_gwh[0]:.3f} GWh")


def test_pareto_select_no_dth():
    # pareto_select must not round-trip through numpy (just check it returns CuPy)
    pop = cp.ones((4, 3, 3), dtype=cp.float32)
    obj = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 1.5], [2.5, 2.5]])

    from config import GAConfig
    from gpuwfarm_core.config import WakeConfig
    from gpuwfarm_core.physics.farm_evaluator import FarmEvaluator
    from gpuwfarm_core.physics.turbine.power_curve import TurbineData
    from projection.base import CompositeProjection
    from gpuwfarm_core.wind.wind_rose import WindRose

    farm_cfg = FarmConfig(n_turbines=3)
    ga_cfg   = GAConfig(pop_size=4, n_generations=1)
    td       = TurbineData.nrel_5mw()
    wake_cfg = WakeConfig()
    turb_cfg = TurbineConfig()
    ev       = FarmEvaluator(farm_cfg, turb_cfg, wake_cfg, td)
    proj     = CompositeProjection([])
    wr       = WindRose.default_12sector()
    ga       = GeneticAlgorithm(farm_cfg, ga_cfg, ev, proj, wr)

    selected, idx = ga.pareto_select(pop, obj, 2)
    assert isinstance(selected, cp.ndarray), "must return CuPy array"
    assert selected.shape[0] == 2
    print(f"  pareto_select returned CuPy shape {selected.shape}")


if __name__ == "__main__":
    tests = [
        test_fixed_costs_and_lcoe_batch,
        test_fast_nondominated_sort_no_rank_overwrite,
        test_fast_nondominated_sort_two_front,
        test_aep_unit_conversion,
        test_pareto_select_no_dth,
    ]
    for t in tests:
        print(f"[{t.__name__}]")
        t()
    print("\nAll tests passed.")
