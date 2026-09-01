"""GAConfig.optimize — "both" | "layout" | "yaw" decision-variable selection."""
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

from gpuwfarm_core.config import FarmConfig, TurbineConfig, WakeConfig
from gpuwfarm_core.physics.farm_evaluator import FarmEvaluator
from gpuwfarm_core.physics.turbine.power_curve import TurbineData
from gpuwfarm_core.wind.wind_rose import WindRose
from gpuwfarm_opt.config import GAConfig
from gpuwfarm_opt.genetic import GeneticAlgorithm
from gpuwfarm_opt.projection.base import CompositeProjection
from gpuwfarm_opt.projection.boundary import BoundaryProjection
from gpuwfarm_opt.projection.spacing import PairwiseSpacingProjection


def _make_ga(optimize: str, n_turbines: int = 4) -> GeneticAlgorithm:
    farm_cfg = FarmConfig(n_turbines=n_turbines)
    ga_cfg   = GAConfig(pop_size=16, n_generations=3, elite=2, optimize=optimize)
    ev       = FarmEvaluator(farm_cfg, TurbineConfig(), WakeConfig(), TurbineData.nrel_5mw())
    proj     = CompositeProjection([
        PairwiseSpacingProjection(farm_cfg, n_passes=5),
        BoundaryProjection(farm_cfg),
    ])
    return GeneticAlgorithm(farm_cfg, ga_cfg, ev, proj, WindRose.default_12sector())


def test_bad_mode_rejected():
    with pytest.raises(ValueError):
        GAConfig(optimize="colour")


def test_layout_mode_keeps_yaw_zero():
    ga = _make_ga("layout")
    pop = ga.init_population()
    assert cp.all(pop[:, :, 2] == 0), "layout mode must init yaw at 0"

    best, history, _, _ = ga.run(verbose=False)
    assert cp.all(best[:, 2] == 0), f"yaw mutated in layout mode: {best[:, 2]}"
    assert len(history) == 3


# Well spaced (1600 m apart, min_spacing 480) and inside the 2000×2000 domain,
# so the projection chain leaves it exactly as-is.
SEED = np.array([[200.0, 200.0], [1800.0, 200.0],
                 [200.0, 1800.0], [1800.0, 1800.0]], dtype=np.float32)


def test_yaw_mode_freezes_layout():
    ga = _make_ga("yaw")
    pop = ga.init_population()
    xy0 = cp.asnumpy(pop[0, :, :2])

    # every individual shares the one fixed layout
    assert np.allclose(cp.asnumpy(pop[:, :, :2]), xy0), "layout must be shared in yaw mode"
    assert cp.any(pop[:, :, 2] != 0), "yaw must be randomised in yaw mode"

    # mutation and projection leave positions untouched
    assert np.allclose(cp.asnumpy(ga.project(ga.mutate(pop))[:, :, :2]), xy0)


def test_yaw_mode_run_keeps_seed_layout():
    ga = _make_ga("yaw")
    pop = ga.init_population(seed_layout=SEED)
    assert np.allclose(cp.asnumpy(pop[:, :, :2]), SEED), "seed must fill the population"

    best, history, _, _ = ga.run(verbose=False, seed_layout=SEED)
    assert np.allclose(cp.asnumpy(best[:, :2]), SEED), "layout moved in yaw mode"
    assert len(history) == 3


def test_both_mode_moves_everything():
    ga = _make_ga("both")
    pop = ga.init_population()
    moved = ga.mutate(pop)
    assert cp.any(moved[:, :, :2] != pop[:, :, :2]), "positions must mutate"
    assert cp.any(moved[:, :, 2] != pop[:, :, 2]), "yaw must mutate"
