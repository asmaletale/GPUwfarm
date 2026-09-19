"""
Resume works the same way for every optimizer.

The history file is the checkpoint for all three -- the row count on disk is the
iteration to resume at, and the shape guard is inherited from Optimizer. What
each optimizer restores beyond the population differs (PSO restarts velocity
and archive, GD restarts Adam's moments but restores its running ideal point),
and those choices are asserted here so a future change has to be deliberate.
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
import h5py
import hdf5plugin  # noqa: F401  (registers the LZ4 codec for reading)

from gpuwfarm_core.config import (
    FarmConfig, TurbineConfig, WakeConfig, CostConfig, VisualImpactConfig,
)
from gpuwfarm_core.physics.farm_evaluator import FarmEvaluator
from gpuwfarm_core.physics.turbine.power_curve import TurbineData
from gpuwfarm_core.wind.wind_rose import WindRose
from gpuwfarm_opt.config import GDConfig, PSOConfig
from gpuwfarm_opt.gradient import GradientDescent
from gpuwfarm_opt.projection.base import CompositeProjection
from gpuwfarm_opt.projection.boundary import BoundaryProjection
from gpuwfarm_opt.projection.spacing import PairwiseSpacingProjection
from gpuwfarm_opt.swarm import ParticleSwarm

P, T = 8, 4
CASES = [(ParticleSwarm, PSOConfig), (GradientDescent, GDConfig)]
IDS = ["pso", "gd"]


def _build(cls, cfg_cls, hist, n_iter, resume=False):
    farm_cfg = FarmConfig(n_turbines=T)
    ev = FarmEvaluator(farm_cfg, TurbineConfig(), WakeConfig(), TurbineData.nrel_5mw())
    proj = CompositeProjection([
        PairwiseSpacingProjection(farm_cfg, n_passes=3),
        BoundaryProjection(farm_cfg),
    ])
    return cls(
        farm_cfg, cfg_cls(pop_size=P, n_generations=n_iter), ev, proj,
        WindRose.default_12sector(),
        cost_cfg=CostConfig(), vi_cfg=VisualImpactConfig(),
        history_file=hist, resume=resume,
    )


@pytest.mark.parametrize("cls,cfg_cls", CASES, ids=IDS)
def test_resume_extends_instead_of_restarting(cls, cfg_cls, tmp_path):
    hist = str(tmp_path / "h.h5")

    with _build(cls, cfg_cls, hist, 4) as opt:
        _, history_a, _, _ = opt.run(verbose=False)
    assert len(history_a) == 4

    with _build(cls, cfg_cls, hist, 10, resume=True) as opt:
        assert opt._start_gen == 4
        _, history_b, _, _ = opt.run(verbose=False)

    assert len(history_b) == 10
    # The rows already on disk are carried forward, not recomputed.
    np.testing.assert_allclose(history_b[:4], history_a, rtol=1e-5)

    with h5py.File(hist, "r") as f:
        assert f["genomes"].shape == (10, P, T * 3)
        assert f["objectives"].shape == (10, P, 2)


@pytest.mark.parametrize("cls,cfg_cls", CASES, ids=IDS)
def test_resume_of_missing_file_starts_fresh(cls, cfg_cls, tmp_path):
    with _build(cls, cfg_cls, str(tmp_path / "nope.h5"), 3, resume=True) as opt:
        assert opt._start_gen == 0
        _, history, _, _ = opt.run(verbose=False)
    assert len(history) == 3


@pytest.mark.parametrize("cls,cfg_cls", CASES, ids=IDS)
def test_resume_rejects_a_mismatched_population(cls, cfg_cls, tmp_path):
    hist = str(tmp_path / "h.h5")
    with _build(cls, cfg_cls, hist, 2) as opt:
        opt.run(verbose=False)

    farm_cfg = FarmConfig(n_turbines=T)
    ev = FarmEvaluator(farm_cfg, TurbineConfig(), WakeConfig(), TurbineData.nrel_5mw())
    proj = CompositeProjection([BoundaryProjection(farm_cfg)])
    with pytest.raises(ValueError):
        cls(
            farm_cfg, cfg_cls(pop_size=P * 2, n_generations=4), ev, proj,
            WindRose.default_12sector(), history_file=hist, resume=True,
        )


def test_gd_restores_its_running_ideal_point(tmp_path):
    """
    Without this the scalarisation would renormalise from scratch on resume and
    every row's objective would jump at the boundary.
    """
    hist = str(tmp_path / "h.h5")
    with _build(GradientDescent, GDConfig, hist, 3) as gd:
        gd.run(verbose=False)

    with _build(GradientDescent, GDConfig, hist, 6, resume=True) as gd:
        z_min, z_max = gd.init_ideal()

    assert np.isfinite(z_min).all() and np.isfinite(z_max).all()
    assert (z_max >= z_min).all()
