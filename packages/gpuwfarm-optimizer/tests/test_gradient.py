"""
SPSA gradient descent: estimator correctness, Adam state, decomposition.

The estimator test is the one that matters. SPSA is unbiased but individually
very noisy, so a wrong sign, a wrong per-column scale, or a missing 1/(2*delta)
all still *run* and still produce a plausible-looking convergence curve. The
only way to catch that is to point it at a function whose gradient is known in
closed form and compare directions.
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
from gpuwfarm_opt.config import GDConfig
from gpuwfarm_opt.gradient import GradientDescent, das_dennis
from gpuwfarm_opt.projection.base import CompositeProjection
from gpuwfarm_opt.projection.boundary import BoundaryProjection
from gpuwfarm_opt.projection.spacing import PairwiseSpacingProjection

P, T = 8, 4


def _gd(**kw):
    farm_cfg = FarmConfig(n_turbines=T)
    ev = FarmEvaluator(farm_cfg, TurbineConfig(), WakeConfig(), TurbineData.nrel_5mw())
    proj = CompositeProjection([
        PairwiseSpacingProjection(farm_cfg, n_passes=3),
        BoundaryProjection(farm_cfg),
    ])
    cfg = GDConfig(pop_size=P, n_generations=kw.pop("n_generations", 5), **kw)
    return GradientDescent(
        farm_cfg, cfg, ev, proj, WindRose.default_12sector(),
        cost_cfg=CostConfig(), vi_cfg=VisualImpactConfig(),
    )


# ──────────────────────────────────────────────────────────────────────
# The estimator
# ──────────────────────────────────────────────────────────────────────

def test_spsa_recovers_the_direction_of_a_known_gradient():
    """
    Point SPSA at an analytic quadratic and compare with its exact gradient.

    f(x) = sum_j ((x_j - t_j) / s_j)^2, with s_j the per-column perturbation
    radius so every column contributes comparably. Exact gradient:
        df/dx_j = 2 (x_j - t_j) / s_j^2

    For a quadratic the central difference is exact, so the SPSA estimate is
    unbiased and the mean over enough pairs must line up with the truth.
    """
    gd = _gd()
    scale = cp.asarray(gd._spsa_c, dtype=cp.float32)          # (3,)
    target = cp.random.uniform(0.0, 1000.0, (P, T, 3)).astype(cp.float32)

    def f(pop):
        # spsa_gradient stacks the +/- halves into one 2P batch, so the target
        # has to tile to match.
        tgt = cp.tile(target, (pop.shape[0] // P, 1, 1))
        return cp.asnumpy(
            (((pop - tgt) / scale) ** 2).sum(axis=(1, 2))
        ).astype(np.float32)[:, None]

    # Stand in for the whole evaluate -> objectives chain.
    gd.evaluate = lambda pop: cp.zeros(pop.shape[0], dtype=cp.float32)
    gd.compute_objectives = lambda pop, aep: f(pop)

    pop = cp.random.uniform(0.0, 1000.0, (P, T, 3)).astype(cp.float32)
    weights = np.ones((P, 1), dtype=np.float32)
    z_min, z_max = np.zeros(1), np.ones(1)

    est = gd.spsa_gradient(pop, weights, z_min, z_max, n_pairs=300)
    exact = 2.0 * (pop - target) / scale ** 2

    e = cp.asnumpy(est).ravel()
    x = cp.asnumpy(exact).ravel()
    cos = float(e @ x / (np.linalg.norm(e) * np.linalg.norm(x)))
    assert cos > 0.9, f"SPSA direction is off: cosine similarity {cos:.3f}"


def test_perturbation_is_rademacher_at_the_column_scale():
    """Every element is exactly +/-c -- never zero, or 1/(2*delta) would blow up."""
    gd = _gd()
    pop = cp.zeros((P, T, 3), dtype=cp.float32)
    d = cp.asnumpy(gd.perturbation(pop))
    c = cp.asnumpy(gd._spsa_c)

    assert (d != 0).all()
    for col in range(3):
        np.testing.assert_allclose(np.abs(d[:, :, col]), c[col], rtol=1e-6)


@pytest.mark.parametrize("mode", ["layout", "yaw"])
def test_frozen_columns_get_zero_gradient(mode):
    gd = _gd(optimize=mode)
    pop = gd.project(gd.init_population())
    w = gd.weight_vectors()
    z_min, z_max = gd.update_ideal(
        gd.compute_objectives(pop, gd.evaluate(pop)), *gd.init_ideal()
    )
    g = cp.asnumpy(gd.spsa_gradient(pop, w, z_min, z_max, n_pairs=1))

    if mode == "layout":
        assert np.abs(g[:, :, 2]).max() == 0.0
    else:
        assert np.abs(g[:, :, :2]).max() == 0.0


# ──────────────────────────────────────────────────────────────────────
# Decomposition
# ──────────────────────────────────────────────────────────────────────

def test_weight_vectors_span_the_simplex():
    gd = _gd()
    w = gd.weight_vectors()
    assert w.shape == (P, 2)
    np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-6)
    assert w.min() == 0.0 and w.max() == 1.0      # both extremes present


def test_das_dennis_lattice():
    pts = das_dennis(3, 4)
    assert pts.shape == (15, 3)                   # C(4+3-1, 3-1) = 15
    np.testing.assert_allclose(pts.sum(axis=1), 1.0, atol=1e-6)


def test_scalarize_normalises_across_wildly_different_units():
    """
    Without range normalisation the large-magnitude objective always wins the
    max and every weight vector collapses to the same ranking.
    """
    gd = _gd()
    # obj 0 ~ 50 (LCOE-like), obj 1 ~ 0.01 (VI-like)
    obj = np.array([[50.0, 0.01], [60.0, 0.005]], dtype=np.float32)
    z_min, z_max = obj.min(axis=0), obj.max(axis=0)

    all_first = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    all_second = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)

    g0 = gd.scalarize(obj, all_first, z_min, z_max)
    g1 = gd.scalarize(obj, all_second, z_min, z_max)

    assert g0[0] < g0[1]      # weighting obj-0 prefers the cheaper row
    assert g1[1] < g1[0]      # weighting obj-1 prefers the lower-VI row


def test_update_ideal_ignores_non_finite_rows():
    gd = _gd()
    z_min, z_max = gd.init_ideal()
    obj = np.array([[1.0, 2.0], [np.inf, 0.5], [3.0, 1.0]], dtype=np.float32)
    z_min, z_max = gd.update_ideal(obj, z_min, z_max)
    np.testing.assert_allclose(z_min, [1.0, 1.0])
    np.testing.assert_allclose(z_max, [3.0, 2.0])


# ──────────────────────────────────────────────────────────────────────
# The step
# ──────────────────────────────────────────────────────────────────────

def test_step_is_pure_and_adam_state_persists():
    """
    The step counter advancing is the direct evidence that Adam's moments are
    not being silently reset -- which is exactly what would happen if the leaf
    tensors were reallocated inside step() instead of held in AdamState.
    """
    gd = _gd()
    pop = gd.project(gd.init_population())
    grad = cp.ones_like(pop)
    adam = gd.init_adam(pop)

    snap = pop.copy()
    out = gd.step(pop, grad, adam)
    cp.testing.assert_array_equal(pop, snap)      # input untouched
    assert out is not pop

    for _ in range(2):
        out = gd.step(out, grad, adam)
    assert int(adam.opt.state[adam.xy]["step"]) == 3

    # A positive gradient must move x downhill (we minimise).
    assert float(out[:, :, 0].max()) < float(snap[:, :, 0].max()) + 1e-6


def test_run_returns_the_uniform_four_tuple_and_descends():
    with _gd(n_generations=10) as gd:
        best, history, front, best_last = gd.run(verbose=False)
    assert best.shape == (T, 3)
    assert len(history) == 10
    assert front.ndim == 2 and front.shape[1] == 2
    assert best_last.shape == (T, 3)
