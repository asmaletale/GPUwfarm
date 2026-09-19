"""
MOPSO: stage purity, archive invariants, and front quality.

The purity check is the important one structurally -- it is what lets a
hand-written loop call the stages in any order. The hypervolume check is the
PSO analogue of the GA's "history is monotonically non-decreasing" assert: the
archive is elitist, so the front it represents must never get worse.
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
from gpuwfarm_core.physics.farm_evaluator import FarmEvaluator
from gpuwfarm_core.physics.turbine.power_curve import TurbineData
from gpuwfarm_core.wind.wind_rose import WindRose
from gpuwfarm_opt.config import PSOConfig
from gpuwfarm_opt.projection.base import CompositeProjection
from gpuwfarm_opt.projection.boundary import BoundaryProjection
from gpuwfarm_opt.projection.spacing import PairwiseSpacingProjection
from gpuwfarm_opt.swarm import ParticleSwarm

P, T = 16, 5


def _pso(**kw):
    farm_cfg = FarmConfig(n_turbines=T)
    ev = FarmEvaluator(farm_cfg, TurbineConfig(), WakeConfig(), TurbineData.nrel_5mw())
    proj = CompositeProjection([
        PairwiseSpacingProjection(farm_cfg, n_passes=3),
        BoundaryProjection(farm_cfg),
    ])
    cfg = PSOConfig(pop_size=P, n_generations=kw.pop("n_generations", 5), **kw)
    return ParticleSwarm(
        farm_cfg, cfg, ev, proj, WindRose.default_12sector(),
        cost_cfg=CostConfig(), vi_cfg=VisualImpactConfig(),
    )


def _state(pso):
    pop = pso.project(pso.init_population())
    vel = pso.init_swarm(pop)
    aep = pso.evaluate(pop)
    obj = pso.compute_objectives(pop, aep)
    archive, archive_obj = pso.archive_update(None, None, pop, obj)
    return pop, vel, aep, obj, archive, archive_obj


def _hypervolume(front, ref):
    """Exact 2-D sweep, minimisation. Mirrors analyze_history's helper."""
    if len(front) == 0:
        return 0.0
    order = np.argsort(front[:, 0])
    f0, f1 = front[order, 0], front[order, 1]
    hv = sum((f0[i + 1] - f0[i]) * (ref[1] - f1[i]) for i in range(len(f0) - 1))
    return max(0.0, hv + (ref[0] - f0[-1]) * (ref[1] - f1[-1]))


def test_stages_do_not_mutate_their_inputs():
    """Purity: every stage returns new arrays and leaves its arguments alone."""
    pso = _pso()
    pop, vel, aep, obj, archive, archive_obj = _state(pso)
    snap = [a.copy() for a in (pop, vel, archive)]
    snap_obj = [o.copy() for o in (obj, archive_obj)]

    leaders = pso.select_leaders(archive, archive_obj, P)
    new_vel = pso.velocity(pop, vel, pop, leaders)
    new_pop = pso.advance(pop, new_vel)
    pso.update_pbest(new_pop, obj, pop, obj)
    pso.archive_update(archive, archive_obj, new_pop, obj)

    for before, after in zip(snap, (pop, vel, archive)):
        cp.testing.assert_array_equal(before, after)
    for before, after in zip(snap_obj, (obj, archive_obj)):
        np.testing.assert_array_equal(before, after)


def test_archive_is_bounded_and_all_nondominated():
    pso = _pso(archive_size=8, n_generations=8)
    pop, vel, aep, obj, archive, archive_obj = _state(pso)
    pbest, pbest_obj = pop, obj

    for _ in range(8):
        leaders = pso.select_leaders(archive, archive_obj, P)
        vel = pso.velocity(pop, vel, pbest, leaders)
        pop = pso.project(pso.advance(pop, vel))
        obj = pso.compute_objectives(pop, pso.evaluate(pop))
        pbest, pbest_obj = pso.update_pbest(pop, obj, pbest, pbest_obj)
        archive, archive_obj = pso.archive_update(archive, archive_obj, pop, obj)

        assert len(archive_obj) <= 8
        assert archive.shape[0] == archive_obj.shape[0]
        # Every archive member must be non-dominated within the archive.
        ranks, _ = pso.fast_nondominated_sort(archive_obj)
        assert (ranks == 0).all()


def test_archive_hypervolume_never_regresses():
    """The archive is elitist, so its front can only improve."""
    pso = _pso(n_generations=10)
    pop, vel, aep, obj, archive, archive_obj = _state(pso)
    pbest, pbest_obj = pop, obj

    seen = [archive_obj.copy()]
    for _ in range(10):
        leaders = pso.select_leaders(archive, archive_obj, P)
        vel = pso.velocity(pop, vel, pbest, leaders)
        pop = pso.project(pso.advance(pop, vel))
        obj = pso.compute_objectives(pop, pso.evaluate(pop))
        pbest, pbest_obj = pso.update_pbest(pop, obj, pbest, pbest_obj)
        archive, archive_obj = pso.archive_update(archive, archive_obj, pop, obj)
        seen.append(archive_obj.copy())

    # One reference point dominated by everything observed, so the HVs compare.
    all_obj = np.vstack(seen)
    ref = all_obj.max(axis=0) + np.maximum(np.ptp(all_obj, axis=0) * 0.1, 1e-6)
    hv = [_hypervolume(f, ref) for f in seen]
    assert all(b >= a - 1e-6 * max(1.0, abs(a)) for a, b in zip(hv, hv[1:])), hv


def test_update_pbest_keeps_incumbent_unless_dominated():
    pso = _pso()
    pop, vel, aep, obj, archive, archive_obj = _state(pso)

    # A strictly dominating candidate for row 0 only.
    better = obj.copy()
    better[0] -= 1.0
    new_pop, new_obj = pso.update_pbest(pop * 2, better, pop, obj)

    np.testing.assert_allclose(new_obj[0], better[0])
    cp.testing.assert_array_equal(new_pop[0], (pop * 2)[0])
    # Every other row was mutually non-dominating, so the incumbent stands.
    np.testing.assert_array_equal(new_obj[1:], obj[1:])
    cp.testing.assert_array_equal(new_pop[1:], pop[1:])


@pytest.mark.parametrize("mode", ["layout", "yaw"])
def test_frozen_columns_never_move(mode):
    pso = _pso(optimize=mode, n_generations=4)
    pop, vel, aep, obj, archive, archive_obj = _state(pso)
    xy0 = cp.asnumpy(pop[:, :, :2]).copy()
    pbest, pbest_obj = pop, obj

    for _ in range(4):
        leaders = pso.select_leaders(archive, archive_obj, P)
        vel = pso.velocity(pop, vel, pbest, leaders)
        pop = pso.project(pso.advance(pop, vel))
        obj = pso.compute_objectives(pop, pso.evaluate(pop))
        archive, archive_obj = pso.archive_update(archive, archive_obj, pop, obj)

    if mode == "layout":
        assert float(cp.abs(pop[:, :, 2]).max()) == 0.0
    else:
        np.testing.assert_allclose(cp.asnumpy(pop[:, :, :2]), xy0, atol=1e-3)


def test_run_returns_the_uniform_four_tuple():
    with _pso(n_generations=4) as pso:
        best, history, front, best_last = pso.run(verbose=False)
    assert best.shape == (T, 3)
    assert len(history) == 4
    assert front.ndim == 2 and front.shape[1] == 2
    assert best_last.shape == (T, 3)
