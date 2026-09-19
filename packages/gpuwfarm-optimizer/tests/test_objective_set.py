"""
ObjectiveSet: stacking, sign convention, and exact backward compatibility.

The compatibility checks below recompute the objective columns from the raw
core calls, the way GeneticAlgorithm.compute_objectives did before objectives
became pluggable. They are what stops the refactor from silently changing the
numbers in a history file.
"""
import os
import sys

import numpy as np
import pytest

if sys.platform == "win32":
    _t = os.path.join(os.path.dirname(sys.executable), "..", "Lib",
                      "site-packages", "torch", "lib")
    if os.path.isdir(os.path.normpath(_t)):
        os.add_dll_directory(os.path.normpath(_t))

import cupy as cp

from gpuwfarm_core.config import (
    FarmConfig, TurbineConfig, WakeConfig, CostConfig, VisualImpactConfig,
)
from gpuwfarm_core.objectives import ObjectiveEvaluator
from gpuwfarm_core.physics.farm_evaluator import FarmEvaluator
from gpuwfarm_core.physics.turbine.power_curve import TurbineData
from gpuwfarm_core.wind.wind_rose import WindRose
from gpuwfarm_opt.config import GAConfig
from gpuwfarm_opt.genetic import GeneticAlgorithm
from gpuwfarm_opt.objectives import (
    ObjectiveSet, LCOE, VisualImpact, NegativeAEP, cable_length_km,
    default_objective_set,
)
from gpuwfarm_opt.projection.base import CompositeProjection
from gpuwfarm_opt.projection.boundary import BoundaryProjection
from gpuwfarm_opt.projection.spacing import PairwiseSpacingProjection

P, T = 8, 5


def _ga(mode="lcoe_vi", objectives=None):
    farm_cfg = FarmConfig(n_turbines=T)
    ev = FarmEvaluator(farm_cfg, TurbineConfig(), WakeConfig(), TurbineData.nrel_5mw())
    proj = CompositeProjection([
        PairwiseSpacingProjection(farm_cfg, n_passes=3),
        BoundaryProjection(farm_cfg),
    ])
    return GeneticAlgorithm(
        farm_cfg, GAConfig(pop_size=P, n_generations=1), ev, proj,
        WindRose.default_12sector(),
        cost_cfg=CostConfig(), vi_cfg=VisualImpactConfig(),
        objectives_mode=mode, objectives=objectives,
    )


@pytest.mark.parametrize("mode", ["lcoe_vi", "aep_vi"])
def test_matches_the_pre_pluggable_formula(mode):
    """The delegate must reproduce the old hardcoded columns bit for bit."""
    ga = _ga(mode)
    pop = ga.project(ga.init_population())
    aep = ga.evaluate(pop)
    got = ga.compute_objectives(pop, aep)

    # Recomputed the way genetic.py used to do it, inline.
    x, y = pop[:, :, 0], pop[:, :, 1]
    cx = x.mean(axis=1, keepdims=True)
    cy = y.mean(axis=1, keepdims=True)
    cable = cp.sqrt((x - cx) ** 2 + (y - cy) ** 2).sum(axis=1) / cp.float32(1000.0)
    aep_gwh = aep / cp.float32(1e6)
    vi = ga.obj_eval.compute_vi_batch(x, y, ga.wind_rose)
    if mode == "aep_vi":
        col0 = (-aep_gwh).astype(cp.float32)
    else:
        col0 = ga.obj_eval.compute_lcoe_batch(T, aep_gwh, cable)
    want = np.column_stack([
        cp.asnumpy(col0).astype(np.float32),
        cp.asnumpy(vi).astype(np.float32),
    ])

    assert got.shape == (P, 2)
    assert got.dtype == np.float32
    np.testing.assert_array_equal(got, want)


def test_cable_length_matches_inline_centroid_formula():
    ga = _ga()
    pop = ga.project(ga.init_population())
    x, y = pop[:, :, 0], pop[:, :, 1]
    cx, cy = x.mean(axis=1, keepdims=True), y.mean(axis=1, keepdims=True)
    want = cp.sqrt((x - cx) ** 2 + (y - cy) ** 2).sum(axis=1) / cp.float32(1000.0)
    np.testing.assert_array_equal(
        cp.asnumpy(cable_length_km(pop)), cp.asnumpy(want)
    )


def test_max_direction_is_negated_and_arbitrary_M_works():
    """A third objective drops in without touching anything else."""
    class RawAEP:
        name, direction = "aep_gwh", "max"

        def __call__(self, pop, aep, wind_rose):
            return (aep / cp.float32(1e6)).astype(cp.float32)

    ga = _ga()
    oset = ObjectiveSet([
        LCOE(ga.obj_eval), VisualImpact(ga.obj_eval), RawAEP(),
    ])
    pop = ga.project(ga.init_population())
    aep = ga.evaluate(pop)
    obj = oset(pop, aep, ga.wind_rose)

    assert len(oset) == 3
    assert oset.names == ["lcoe", "vi", "aep_gwh"]
    assert obj.shape == (P, 3)
    # "max" column comes back negated, so smaller is better everywhere.
    np.testing.assert_allclose(
        obj[:, 2], cp.asnumpy(-aep / cp.float32(1e6)), rtol=1e-6
    )
    # And Pareto ranking consumes M=3 with no changes.
    ranks, cd = ga.fast_nondominated_sort(obj)
    assert ranks.shape == (P,) and cd.shape == (P,)
    assert ranks.min() == 0


def test_explicit_objectives_argument_overrides_mode():
    ga = _ga(mode="lcoe_vi", objectives=[NegativeAEP()])
    assert ga.objectives.names == ["neg_aep_gwh"]
    pop = ga.project(ga.init_population())
    assert ga.compute_objectives(pop, ga.evaluate(pop)).shape == (P, 1)


def test_bad_inputs_rejected():
    with pytest.raises(ValueError):
        ObjectiveSet([])
    with pytest.raises(ValueError):
        default_objective_set("nope", None)

    class Bad:
        name, direction = "bad", "sideways"

        def __call__(self, pop, aep, wind_rose):
            return None

    with pytest.raises(ValueError):
        ObjectiveSet([Bad()])
