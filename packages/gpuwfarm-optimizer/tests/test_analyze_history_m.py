"""
The analysis helpers must not assume two objectives.

The Monte-Carlo hypervolume is validated against the exact 2-D sweep it stands
in for above M=2 -- checking an estimator against the thing it replaces is the
only way to know the sample count is high enough to be useful.
"""
import numpy as np

from gpuwfarm_opt.scripts.analyze_history import (
    _hypervolume, _hypervolume_mc, _pareto_mask,
)


def test_pareto_mask_handles_three_objectives():
    obj = np.array([
        [1.0, 1.0, 1.0],   # dominates everything below
        [2.0, 2.0, 2.0],
        [1.0, 3.0, 1.0],
        [3.0, 1.0, 1.0],
    ])
    np.testing.assert_array_equal(_pareto_mask(obj), [True, False, False, False])


def test_pareto_mask_keeps_mutually_nondominated_rows():
    obj = np.array([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]])
    assert _pareto_mask(obj).all()


def test_monte_carlo_hypervolume_tracks_the_exact_sweep():
    front = np.array([[0.0, 3.0], [1.0, 1.0], [3.0, 0.0]])
    ref = np.array([4.0, 4.0])

    exact = _hypervolume(front, ref)
    assert exact == 11.0   # hand-computed L-strips

    mc = _hypervolume_mc(front, ref, n_samples=400_000)
    assert abs(mc - exact) / exact < 0.02


def test_hypervolume_dispatches_on_objective_count():
    """M > 2 must go to the estimator instead of the 2-D sweep, not crash."""
    front = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    ref = np.array([2.0, 2.0, 2.0])
    hv = _hypervolume(front, ref)
    assert 0.0 < hv < 8.0        # inside the 2x2x2 reference box


def test_empty_front_is_zero_volume():
    assert _hypervolume(np.empty((0, 2)), np.array([1.0, 1.0])) == 0.0
    assert _hypervolume(np.empty((0, 3)), np.array([1.0, 1.0, 1.0])) == 0.0
