"""
Selection pressure: binary tournament + elitist (mu + lambda) survival.

Before this was added the loop had essentially no selection pressure — select()
and pareto_select() were both called with n_select == pop_size, making them pure
reorders, and the only thing preserving good individuals was an elite blit that
landed on arbitrary slots because crossover shuffled the population first.
"""
import sys, os
# On Windows without a full CUDA toolkit, PyTorch ships the CUDA DLLs CuPy
# needs; register that directory before CuPy is imported. No-op elsewhere.
if sys.platform == "win32":
    _torch_lib = os.path.normpath(
        os.path.join(sys.executable, "..", "..", "Lib", "site-packages", "torch", "lib")
    )
    if os.path.isdir(_torch_lib):
        os.add_dll_directory(_torch_lib)

import numpy as np
import cupy as cp
import pytest

from gpuwfarm_core.config import (
    FarmConfig, TurbineConfig, WakeConfig, CostConfig, VisualImpactConfig,
)
from gpuwfarm_core.physics.farm_evaluator import FarmEvaluator
from gpuwfarm_core.physics.turbine.power_curve import TurbineData
from gpuwfarm_core.wind.wind_rose import WindRose
from gpuwfarm_opt.config import GAConfig
from gpuwfarm_opt.genetic import GeneticAlgorithm
from gpuwfarm_opt.projection.base import CompositeProjection
from gpuwfarm_opt.projection.boundary import BoundaryProjection
from gpuwfarm_opt.projection.spacing import PairwiseSpacingProjection


def _make_ga(pop_size=32, n_generations=15, n_turbines=6, multi_objective=False):
    farm_cfg = FarmConfig(n_turbines=n_turbines)
    ga_cfg   = GAConfig(pop_size=pop_size, n_generations=n_generations, optimize="both")
    ev = FarmEvaluator(farm_cfg, TurbineConfig(), WakeConfig(), TurbineData.nrel_5mw())
    proj = CompositeProjection([
        PairwiseSpacingProjection(farm_cfg, n_passes=5),
        BoundaryProjection(farm_cfg),
    ])
    return GeneticAlgorithm(
        farm_cfg, ga_cfg, ev, proj, WindRose.default_12sector(),
        cost_cfg=CostConfig(),
        vi_cfg=VisualImpactConfig() if multi_objective else None,
    )


# ──────────────────────────────────────────────────────────────────────
# The crowding-distance inf bug in pareto_select
# ──────────────────────────────────────────────────────────────────────

# Front 0 has four mutually non-dominating points, so its two interior members
# get a *finite* crowding distance, while the extremes of *every* front get inf.
# Scoring as `rank * 1e6 - distance` therefore ranked any front's extreme above
# the rank-0 interior (-inf beats any finite score): on this fixture it selects
# {0, 3, 4, 5}, i.e. it drops two rank-0 individuals in favour of two rank-1
# ones. Rank has to be the primary key.
_OBJ = np.array([
    [0.0, 3.0],   # 0  front 0, extreme  -> inf
    [1.0, 2.0],   # 1  front 0, interior -> finite
    [2.0, 1.0],   # 2  front 0, interior -> finite
    [3.0, 0.0],   # 3  front 0, extreme  -> inf
    [1.0, 4.0],   # 4  front 1
    [4.0, 1.0],   # 5  front 1
    [2.0, 3.0],   # 6  front 1
    [3.0, 4.0],   # 7  front 2 -> inf
    [4.0, 3.0],   # 8  front 2 -> inf
], dtype=np.float32)


def test_fixture_has_the_shape_the_bug_needs():
    ranks, distances = GeneticAlgorithm.fast_nondominated_sort(_OBJ)
    assert list(ranks) == [0, 0, 0, 0, 1, 1, 1, 2, 2]
    assert np.isfinite(distances[1]) and np.isfinite(distances[2])   # front-0 interior
    assert np.isinf(distances[7]) and np.isinf(distances[8])         # front-2 extremes


def test_selection_fills_whole_fronts_in_rank_order():
    ga = _make_ga(pop_size=4, n_turbines=3)
    pop = cp.zeros((len(_OBJ), 3, 3), dtype=cp.float32)

    _, idx = ga.pareto_select(pop, _OBJ, 4)
    assert set(idx.tolist()) == {0, 1, 2, 3}, (
        "an inf-crowding extreme from a worse front outranked the rank-0 interior"
    )


def test_partial_front_breaks_ties_on_crowding_distance():
    """Taking 6 of 9 must be all of front 0 plus the two most-spread of front 1."""
    ga = _make_ga(pop_size=4, n_turbines=3)
    pop = cp.zeros((len(_OBJ), 3, 3), dtype=cp.float32)
    ranks, distances = GeneticAlgorithm.fast_nondominated_sort(_OBJ)

    _, idx = ga.pareto_select(pop, _OBJ, 6, ranks=ranks, distances=distances)
    assert set(idx[:4].tolist()) == {0, 1, 2, 3}
    front1 = idx[4:]
    assert all(ranks[i] == 1 for i in front1)
    # The two picked from front 1 must be its highest-crowding members.
    front1_all = np.where(ranks == 1)[0]
    best_two = front1_all[np.argsort(-distances[front1_all])][:2]
    assert set(front1.tolist()) == set(best_two.tolist())


# ──────────────────────────────────────────────────────────────────────
# Tournament: does the mating pool actually favour better individuals?
# ──────────────────────────────────────────────────────────────────────

def test_tournament_aep_favours_high_fitness():
    ga = _make_ga(pop_size=64, n_turbines=3)
    cp.random.seed(0)
    P = 64
    # Fitness == individual index, encoded in the layout so we can read it back.
    pop = cp.zeros((P, 3, 3), dtype=cp.float32)
    pop[:, :, 0] = cp.arange(P, dtype=cp.float32)[:, None]
    aep = cp.arange(P, dtype=cp.float32)

    pool = ga.tournament_aep(pop, aep)
    assert pool.shape == pop.shape
    # Binary tournament on a uniform ladder has expected value ~2/3 of the range.
    mean_pool = float(pool[:, 0, 0].mean())
    assert mean_pool > float(aep.mean()), "tournament applied no pressure"
    assert mean_pool == pytest.approx(2.0 * (P - 1) / 3.0, rel=0.25)


def test_tournament_pareto_favours_low_rank():
    ga = _make_ga(pop_size=9, n_turbines=3)
    np.random.seed(0)
    ranks, distances = GeneticAlgorithm.fast_nondominated_sort(_OBJ)
    pop = cp.zeros((len(_OBJ), 3, 3), dtype=cp.float32)
    pop[:, :, 0] = cp.asarray(np.arange(len(_OBJ), dtype=np.float32))[:, None]

    pool = ga.tournament_pareto(pop, ranks, distances)
    assert pool.shape == pop.shape
    picked = cp.asnumpy(pool[:, 0, 0]).astype(int)
    assert ranks[picked].mean() < ranks.mean(), "tournament applied no pressure"


# ──────────────────────────────────────────────────────────────────────
# Survival is elitist
# ──────────────────────────────────────────────────────────────────────

def test_survive_aep_keeps_the_best_p():
    ga = _make_ga(pop_size=8, n_turbines=3)
    pop      = cp.zeros((8, 3, 3), dtype=cp.float32)
    children = cp.ones((8, 3, 3), dtype=cp.float32)
    aep   = cp.arange(8, dtype=cp.float32)          # 0..7
    aep_c = cp.arange(8, 16, dtype=cp.float32)      # 8..15  (all better)

    surv_pop, surv_aep = ga.survive_aep(pop, aep, children, aep_c)
    assert surv_pop.shape == (8, 3, 3)
    assert cp.allclose(surv_aep, cp.arange(15, 7, -1, dtype=cp.float32))
    assert cp.all(surv_pop == 1.0), "offspring should have displaced every parent"


def test_survive_pareto_keeps_pop_size_and_stays_consistent():
    ga = _make_ga(pop_size=4, n_turbines=3)
    pop      = cp.zeros((4, 3, 3), dtype=cp.float32)
    children = cp.ones((4, 3, 3), dtype=cp.float32)
    aep      = cp.arange(4, dtype=cp.float32)
    aep_c    = cp.arange(4, 8, dtype=cp.float32)
    obj      = np.array([[4.0, 4.0], [5.0, 5.0], [6.0, 6.0], [7.0, 7.0]], np.float32)
    obj_c    = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], np.float32)

    p, a, o = ga.survive_pareto(pop, aep, obj, children, aep_c, obj_c)
    assert p.shape == (4, 3, 3) and a.shape == (4,) and o.shape == (4, 2)
    # obj_c dominates obj throughout, so only offspring survive — and aep/obj
    # must still line up with the individuals they belong to.
    assert cp.all(p == 1.0)
    assert set(cp.asnumpy(a).tolist()) == {4.0, 5.0, 6.0, 7.0}
    assert o.max() <= 3.0


@pytest.mark.parametrize("multi_objective", [False, True])
def test_best_aep_never_regresses_over_a_run(multi_objective):
    """
    The (mu + lambda) merge cannot lose the incumbent, so the history must be
    monotonically non-decreasing. This is the end-to-end guard on the whole
    selection path.
    """
    cp.random.seed(1)
    np.random.seed(1)
    with _make_ga(multi_objective=multi_objective) as ga:
        _, history, _, _ = ga.run(verbose=False, multi_objective=multi_objective)

    assert len(history) == ga.ga_cfg.n_generations
    for prev, cur in zip(history, history[1:]):
        assert cur >= prev - 1.0, f"best AEP regressed: {prev:.6e} -> {cur:.6e}"
    assert history[-1] > history[0], "20 generations produced no improvement at all"
