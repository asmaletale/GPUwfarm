"""
The evals history must contain every genome the evaluator ever saw — exactly
P * (n_generations + 1) of them — while the population history keeps only the
P survivors per generation.
"""
import numpy as np
import h5py
import hdf5plugin  # noqa: F401 — registers the LZ4 codec for reading

from gpuwfarm_core.config import WakeConfig, FarmConfig, TurbineConfig, VisualImpactConfig
from gpuwfarm_core.physics.farm_evaluator import FarmEvaluator
from gpuwfarm_core.physics.turbine.power_curve import TurbineData
from gpuwfarm_core.wind.wind_rose import WindRose
from gpuwfarm_opt.config import GAConfig
from gpuwfarm_opt.genetic import GeneticAlgorithm
from gpuwfarm_opt.projection.base import CompositeProjection
from gpuwfarm_opt.projection.spacing import PairwiseSpacingProjection
from gpuwfarm_opt.projection.boundary import BoundaryProjection


P, T, G = 8, 5, 4


def _run(tmp_path, multi_objective):
    farm_cfg = FarmConfig(n_turbines=T)
    ga_cfg = GAConfig(pop_size=P, n_generations=G)
    evaluator = FarmEvaluator(farm_cfg, TurbineConfig(), WakeConfig(), TurbineData.nrel_5mw())
    projection = CompositeProjection([
        PairwiseSpacingProjection(farm_cfg, n_passes=3),
        BoundaryProjection(farm_cfg),
    ])
    hist, evals = str(tmp_path / "hist.h5"), str(tmp_path / "evals.h5")
    with GeneticAlgorithm(
        farm_cfg, ga_cfg, evaluator, projection, WindRose.default_12sector(),
        vi_cfg=VisualImpactConfig() if multi_objective else None,
        history_file=hist, evals_file=evals,
    ) as ga:
        ga.run(verbose=False, multi_objective=multi_objective)
    return hist, evals


def test_evals_file_holds_every_evaluation(tmp_path):
    hist, evals = _run(tmp_path, multi_objective=False)

    with h5py.File(evals, "r") as f:
        genomes, fitnesses = f["genomes"][:], f["fitnesses"][:]
    with h5py.File(hist, "r") as f:
        survivors = f["genomes"][:]

    # One row per evaluated batch: initial population + one offspring batch per gen
    assert genomes.shape == (G + 1, P, T * 3)
    assert fitnesses.shape == (G + 1, P)

    # One row per evaluator call, duplicates kept: an offspring that neither
    # crossover nor mutation touched is a genuine repeat evaluation, not a
    # bookkeeping slip, and dropping it would understate the GPU cost.
    flat = genomes.reshape(-1, T * 3)
    assert len(flat) == P * (G + 1)

    # Survivors are drawn from the evaluated set, never the other way round
    seen = {row.tobytes() for row in flat}
    assert all(row.tobytes() in seen for row in survivors.reshape(-1, T * 3))
    assert survivors.shape[0] == G


def test_objectives_logged_for_discarded_offspring(tmp_path):
    _, evals = _run(tmp_path, multi_objective=True)
    with h5py.File(evals, "r") as f:
        assert f["objectives"].shape == (G + 1, P, 2)
        assert np.isfinite(f["objectives"][:]).all()
