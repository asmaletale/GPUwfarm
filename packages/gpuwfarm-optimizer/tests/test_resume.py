"""
An interrupted run must be resumable from its history file.

Two things are being checked: that resuming continues the search instead of
restarting it (and never loses the incumbent across the boundary), and that the
file on disk is usable after a kill that runs no cleanup at all.
"""
import subprocess
import sys
import textwrap
import time

import numpy as np
import h5py
import hdf5plugin  # noqa: F401 — registers the LZ4 codec for reading
import pytest

from gpuwfarm_core.config import WakeConfig, FarmConfig, TurbineConfig
from gpuwfarm_core.physics.farm_evaluator import FarmEvaluator
from gpuwfarm_core.physics.turbine.power_curve import TurbineData
from gpuwfarm_core.wind.wind_rose import WindRose
from gpuwfarm_opt.config import GAConfig
from gpuwfarm_opt.genetic import GeneticAlgorithm
from gpuwfarm_opt.projection.base import CompositeProjection
from gpuwfarm_opt.projection.spacing import PairwiseSpacingProjection
from gpuwfarm_opt.projection.boundary import BoundaryProjection

P, T = 8, 5


def _ga(hist, n_generations, resume=False, evals=None):
    farm_cfg = FarmConfig(n_turbines=T)
    ga_cfg = GAConfig(pop_size=P, n_generations=n_generations)
    evaluator = FarmEvaluator(farm_cfg, TurbineConfig(), WakeConfig(), TurbineData.nrel_5mw())
    projection = CompositeProjection([
        PairwiseSpacingProjection(farm_cfg, n_passes=3),
        BoundaryProjection(farm_cfg),
    ])
    return GeneticAlgorithm(
        farm_cfg, ga_cfg, evaluator, projection, WindRose.default_12sector(),
        history_file=hist, evals_file=evals, resume=resume,
    )


def test_resume_extends_instead_of_restarting(tmp_path):
    hist = str(tmp_path / "hist.h5")

    with _ga(hist, 4) as ga:
        _, history_a, _, _ = ga.run(verbose=False)
    assert len(history_a) == 4

    # Same file, more generations: should pick up at 4 and run through 9
    with _ga(hist, 10, resume=True) as ga:
        best, history_b, _, _ = ga.run(verbose=False)

    assert len(history_b) == 10
    assert history_b[:4] == pytest.approx(history_a, rel=1e-6), "history was rewritten"

    with h5py.File(hist, "r") as f:
        assert f["genomes"].shape == (10, P, T * 3)

    # Elitist merge: the incumbent must survive the resume boundary
    assert np.all(np.diff(history_b) >= -1e-3), "best AEP regressed"
    assert best.shape == (T, 3)


def test_resume_of_missing_file_starts_fresh(tmp_path):
    with _ga(str(tmp_path / "nope.h5"), 3, resume=True) as ga:
        _, history, _, _ = ga.run(verbose=False)
    assert len(history) == 3


def test_resume_rejects_mismatched_population(tmp_path):
    hist = str(tmp_path / "hist.h5")
    with _ga(hist, 2) as ga:
        ga.run(verbose=False)

    farm_cfg = FarmConfig(n_turbines=T)
    evaluator = FarmEvaluator(farm_cfg, TurbineConfig(), WakeConfig(), TurbineData.nrel_5mw())
    projection = CompositeProjection([BoundaryProjection(farm_cfg)])
    with pytest.raises(ValueError, match="pop_size"):
        GeneticAlgorithm(
            farm_cfg, GAConfig(pop_size=P * 2, n_generations=4), evaluator,
            projection, WindRose.default_12sector(), history_file=hist, resume=True,
        )


CHILD = textwrap.dedent("""
    import os, sys, time
    _t = os.path.join(os.path.dirname(sys.executable), '..', 'Lib', 'site-packages', 'torch', 'lib')
    if os.path.isdir(_t): os.add_dll_directory(os.path.normpath(_t))
    from gpuwfarm_core.config import WakeConfig, FarmConfig, TurbineConfig
    from gpuwfarm_core.physics.farm_evaluator import FarmEvaluator
    from gpuwfarm_core.physics.turbine.power_curve import TurbineData
    from gpuwfarm_core.wind.wind_rose import WindRose
    from gpuwfarm_opt.config import GAConfig
    from gpuwfarm_opt.genetic import GeneticAlgorithm
    from gpuwfarm_opt.projection.base import CompositeProjection
    from gpuwfarm_opt.projection.boundary import BoundaryProjection

    farm = FarmConfig(n_turbines={T})
    ev = FarmEvaluator(farm, TurbineConfig(), WakeConfig(), TurbineData.nrel_5mw())
    proj = CompositeProjection([BoundaryProjection(farm)])
    ga = GeneticAlgorithm(farm, GAConfig(pop_size={P}, n_generations=100000),
                          ev, proj, WindRose.default_12sector(),
                          history_file=r"{hist}")

    # The stage methods spelled out, so the child can announce each flushed
    # generation on stdout -- the parent cannot poll the HDF5 file itself while
    # this process holds the writer's lock on it.
    pop = ga.project(ga.init_population())
    aep = ga.evaluate(pop)
    for g in range(100000):
        children = ga.project(ga.mutate(ga.crossover(ga.tournament_aep(pop, aep))))
        pop, aep = ga.survive_aep(pop, aep, children, ga.evaluate(children))
        ga.log(g, pop, aep)
        print(g, flush=True)
""")


def test_file_survives_a_kill_with_no_cleanup(tmp_path):
    """SIGKILL/TerminateProcess runs no finally block — only the per-row flush saves us."""
    hist = tmp_path / "hist.h5"
    script = tmp_path / "child.py"
    script.write_text(CHILD.format(T=T, P=P, hist=str(hist).replace("\\", "\\\\")))

    proc = subprocess.Popen(
        [sys.executable, str(script)], stdout=subprocess.PIPE, text=True
    )
    try:
        deadline = time.time() + 180
        while time.time() < deadline:
            line = proc.stdout.readline()
            if not line:
                pytest.fail("child exited before logging any generation")
            if int(line) >= 5:
                break
        else:
            pytest.fail("child never reached 5 generations")
    finally:
        proc.kill()      # no cleanup, no close(), no finally
        proc.wait()

    # Windows releases the killed process's file lock asynchronously, so the
    # first open can still fail with ERROR_LOCK_VIOLATION.
    for _ in range(40):
        try:
            with h5py.File(hist, "r") as f:
                n_done = f["genomes"].shape[0]
                assert np.isfinite(f["fitnesses"][:n_done]).all()
            break
        except OSError:
            time.sleep(0.25)
    else:
        pytest.fail("history file never became readable after the kill")

    # log() is async, so the writer may be a row or two behind what the child
    # printed -- the contract is that whatever reached disk is intact and
    # resumable, not that nothing was in flight.
    assert n_done >= 1

    with _ga(str(hist), n_done + 3, resume=True) as ga:
        _, history, _, _ = ga.run(verbose=False)
    assert len(history) == n_done + 3
