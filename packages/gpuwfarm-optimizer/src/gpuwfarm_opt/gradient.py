"""
Gradient descent by SPSA, with Chebyshev decomposition for the Pareto front.

Two ideas make this work on a physics core that has no autodiff.

**SPSA.** Simultaneous Perturbation Stochastic Approximation (Spall, 1992)
estimates a gradient from two evaluations, regardless of how many variables
there are. Perturb *every* coordinate at once with a random +/-1 vector d:

    g_i = (f(x + c*d) - f(x - c*d)) / (2 * c * d_i)

A single draw is a very noisy estimate, but it is unbiased to O(c^2), so the
averaging happens across iterations rather than within one. Classical finite
differences would need 2*T*3 evaluations per gradient; this needs 2.

Being derivative-free is not just a convenience here, it is the whole reason
the method is usable: the objective chain contains a sorted sweep-line union
area (visual impact), lookup-table interpolation (the power curve), and hard
distance gates in the turbulence model. None of that has a useful derivative.
SPSA never differentiates anything -- it only evaluates -- so all of it is
irrelevant. The same argument makes the non-smooth `max` inside the Chebyshev
scalarisation free.

**Decomposition.** Row p of the batch descends its *own* scalarised objective,
built from its own weight vector. P rows sweep out P points of the Pareto
front in a single run, so the batch *is* the front -- no archive needed.

The scalarisation is augmented Chebyshev, not a weighted sum, because a
weighted sum provably cannot reach points on a non-convex region of a front and
LCOE-against-visual-impact has no reason to be convex.

Every stage is pure and individually callable, as everywhere else in this
package. Adam's momentum is the one piece of genuinely order-dependent state;
it lives in a caller-owned ``AdamState`` threaded through the loop, exactly the
way the swarm threads its velocity -- never on the instance.
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
from typing import Sequence, TYPE_CHECKING

import numpy as np
import cupy as cp

from gpuwfarm_core.config import FarmConfig, CostConfig, VisualImpactConfig
from gpuwfarm_core.physics.farm_evaluator import FarmEvaluator
from gpuwfarm_core.wind.wind_rose import WindRose
from gpuwfarm_opt.base import Optimizer
from gpuwfarm_opt.config import GDConfig
from gpuwfarm_opt.objectives import ObjectiveSet, Objective
from gpuwfarm_opt.projection.base import ProjectionOperator

if TYPE_CHECKING:  # pragma: no cover
    import torch


def das_dennis(n_obj: int, n_div: int) -> np.ndarray:
    """
    Das-Dennis simplex lattice: every point with non-negative coordinates that
    are multiples of 1/n_div and sum to 1.

    Returns (C(n_div + n_obj - 1, n_obj - 1), n_obj) float32. This is the same
    structured spread of weight vectors NSGA-III and MOEA/D use.
    """
    pts = []
    for cut in combinations(range(n_div + n_obj - 1), n_obj - 1):
        prev, row = -1, []
        for c in cut:
            row.append(c - prev - 1)
            prev = c
        row.append(n_div + n_obj - 2 - prev)
        pts.append(row)
    return (np.asarray(pts, dtype=np.float32) / float(n_div)).astype(np.float32)


@dataclass
class AdamState:
    """
    The torch optimiser and the leaf tensors its momentum is keyed to.

    Held by the caller's loop, not by the optimizer instance. That is not
    ceremony: torch keys its moment buffers by *tensor identity*, so an
    implementation that allocated fresh leaves inside step() would silently
    reset (m, v, t) every iteration and quietly degrade Adam to plain SGD --
    no exception, and barely visible in a convergence curve. Creating the
    leaves once, here, makes that failure impossible.

    xy and yaw are separate parameter groups because Adam's step size is
    approximately lr regardless of gradient magnitude, and x/y are metres on a
    ~2 km farm while yaw is radians in +/-0.5. One learning rate cannot serve
    both.
    """

    xy:  "torch.Tensor"   # (P, T, 2) float32 cuda leaf
    yaw: "torch.Tensor"   # (P, T)    float32 cuda leaf
    opt: "torch.optim.Optimizer"


class GradientDescent(Optimizer):
    """
    Batched SPSA gradient descent with Chebyshev decomposition.

    Population tensor: (P, T, 3). Unlike the GA and the swarm, rows do not
    compete -- each descends its own scalarised objective, so the final batch is
    a set of Pareto-front candidates rather than a converged population.

    Use as a context manager when a history_file is given:

        with GradientDescent(..., history_file="gd.h5") as gd:
            ...
    """

    def __init__(
        self,
        farm_cfg:   FarmConfig,
        gd_cfg:     GDConfig,
        evaluator:  FarmEvaluator,
        projection: ProjectionOperator,
        wind_rose:  WindRose,
        cost_cfg:        CostConfig | None = None,
        vi_cfg:          VisualImpactConfig | None = None,
        objectives_mode: str = "lcoe_vi",
        objectives:      ObjectiveSet | Sequence[Objective] | None = None,
        history_file:    str | None = None,
        evals_file:      str | None = None,
        resume:          bool = False,
    ) -> None:
        super().__init__(
            farm_cfg, gd_cfg, evaluator, projection, wind_rose,
            cost_cfg=cost_cfg, vi_cfg=vi_cfg, objectives_mode=objectives_mode,
            objectives=objectives, history_file=history_file,
            evals_file=evals_file, resume=resume,
        )
        self.gd_cfg = gd_cfg

        # Perturbation radius per column: metres, metres, radians. A single
        # scalar c would be meaningless across a 2 km position and a 0.5 rad
        # yaw -- this is the single most common way to get SPSA wrong.
        self._spsa_c = cp.asarray(
            [gd_cfg.spsa_c_xy, gd_cfg.spsa_c_xy, np.deg2rad(gd_cfg.spsa_c_yaw_deg)],
            dtype=cp.float32,
        )
        self._free = cp.asarray(
            [float(self.opt_layout), float(self.opt_layout), float(self.opt_yaw)],
            dtype=cp.float32,
        )

    # ──────────────────────────────────────────────────────────────────
    # Decomposition
    # ──────────────────────────────────────────────────────────────────

    def weight_vectors(self) -> np.ndarray:
        """
        (P, M) weight vectors on the unit simplex, one per batch row.

        M == 2 is an exact linspace, so the rows are ordered along the front and
        P is honoured exactly. For M > 2 it is the largest Das-Dennis lattice
        that fits in P rows, with the remainder filled by Dirichlet samples.

        ponytail: the Dirichlet tail keeps pop_size free instead of forcing it
        to be a binomial coefficient. If the spread ever matters more than the
        convenience, validate P against the lattice and raise instead.
        """
        P, M = self.gd_cfg.pop_size, len(self.objectives)
        if M == 1:
            return np.ones((P, 1), dtype=np.float32)
        if M == 2:
            a = np.linspace(0.0, 1.0, P, dtype=np.float32)
            return np.column_stack([a, 1.0 - a]).astype(np.float32)

        n_div, lattice = 1, das_dennis(M, 1)
        while True:
            nxt = das_dennis(M, n_div + 1)
            if len(nxt) > P:
                break
            n_div, lattice = n_div + 1, nxt
        if len(lattice) < P:
            pad = np.random.dirichlet(np.ones(M), size=P - len(lattice))
            lattice = np.vstack([lattice, pad.astype(np.float32)])
        return lattice[:P].astype(np.float32)

    def init_ideal(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Empty running ideal/nadir estimates, or the ones implied by a resumed
        history so the scalarisation is continuous across the boundary.

        Returns:
            (z_min (M,), z_max (M,)) float64
        """
        M = len(self.objectives)
        if self._resume_obj is not None and self._resume_obj.size:
            flat = self._resume_obj.reshape(-1, self._resume_obj.shape[-1])
            flat = flat[np.isfinite(flat).all(axis=1)]
            if len(flat):
                return flat.min(axis=0).astype(np.float64), flat.max(axis=0).astype(np.float64)
        return np.full(M, np.inf), np.full(M, -np.inf)

    @staticmethod
    def update_ideal(
        obj: np.ndarray, z_min: np.ndarray, z_max: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Fold one batch into the running per-objective min and max.

        Rows with a non-finite objective (LCOE is infinite at zero AEP) are
        skipped rather than poisoning the normalisation. Pure.
        """
        finite = obj[np.isfinite(obj).all(axis=1)]
        if not len(finite):
            return z_min, z_max
        return (
            np.minimum(z_min, finite.min(axis=0)),
            np.maximum(z_max, finite.max(axis=0)),
        )

    def scalarize(
        self,
        obj:     np.ndarray,
        weights: np.ndarray,
        z_min:   np.ndarray,
        z_max:   np.ndarray,
    ) -> np.ndarray:
        """
        Augmented Chebyshev scalarisation, one scalar per row.

            zn   = (obj - z_min) / (z_max - z_min)
            g[p] = max_m( w[p,m] * zn[p,m] ) + rho * sum_m( w[p,m] * zn[p,m] )

        The range normalisation is not optional: LCOE runs around 50 EUR/MWh and
        visual impact around 0.01, so without it the max_m is always the LCOE
        term and every row collapses onto the same point. The rho term is what
        excludes weakly-dominated solutions.

        Args:
            obj:     (N, M) objectives, minimisation
            weights: (N, M) one weight vector per row
            z_min:   (M,) running ideal
            z_max:   (M,) running nadir estimate

        Returns:
            (N,) float32
        """
        rng = np.maximum(z_max - z_min, 1e-12)
        zn = (obj - z_min) / rng
        zn = np.nan_to_num(zn, nan=1e6, posinf=1e6, neginf=0.0)
        wz = weights * zn
        return (wz.max(axis=1) + self.gd_cfg.rho * wz.sum(axis=1)).astype(np.float32)

    # ──────────────────────────────────────────────────────────────────
    # SPSA
    # ──────────────────────────────────────────────────────────────────

    def perturbation(self, pop: cp.ndarray) -> cp.ndarray:
        """
        (P, T, 3) signed Rademacher perturbation, scaled per column.

        Every element is exactly +c or -c and never zero, which is what makes
        the elementwise 1/(2*delta) in spsa_gradient well defined. Rademacher
        specifically, not Gaussian: SPSA's convergence proof needs finite
        inverse moments of the perturbation, and a Gaussian draw near zero blows
        the estimator up.
        """
        signs = cp.random.randint(0, 2, pop.shape).astype(cp.float32) * 2.0 - 1.0
        return signs * self._spsa_c

    def spsa_gradient(
        self,
        pop:     cp.ndarray,
        weights: np.ndarray,
        z_min:   np.ndarray,
        z_max:   np.ndarray,
        n_pairs: int | None = None,
    ) -> cp.ndarray:
        """
        (P, T, 3) gradient estimate of each row's own scalarised objective, in
        per-metre (x, y) and per-radian (yaw) units.

        Both halves of each +/- pair go to the evaluator in a single (2P, T, 3)
        batch, so one pair costs one evaluator call, not two. Cost is
        2*n_pairs evaluations per iteration *independent of turbine count* --
        the reason SPSA is used here instead of 2*T*3 coordinate-wise
        differences.

        The perturbed points are evaluated **unprojected**, deliberately: if
        spacing repair ran on them it could cancel the perturbation outright and
        flatten the difference to zero. Only the accepted step is projected.

        Frozen columns are not perturbed and their gradient is zeroed, so
        "layout" and "yaw" modes cannot drift.
        """
        cfg = self.gd_cfg
        n_pairs = cfg.spsa_pairs if n_pairs is None else n_pairs
        P = pop.shape[0]
        grad = cp.zeros_like(pop)

        w2 = np.vstack([weights, weights])
        for _ in range(n_pairs):
            delta = self.perturbation(pop)          # +/- c, never zero
            step = delta * self._free               # frozen columns unperturbed

            both = cp.concatenate([pop + step, pop - step], axis=0)
            obj = self.compute_objectives(both, self.evaluate(both))
            f = self.scalarize(obj, w2, z_min, z_max)

            diff = cp.asarray(f[:P] - f[P:], dtype=cp.float32)
            grad += diff[:, None, None] / (2.0 * delta)

        return ((grad / cp.float32(n_pairs)) * self._free).astype(cp.float32)

    # ──────────────────────────────────────────────────────────────────
    # The step
    # ──────────────────────────────────────────────────────────────────

    def init_adam(self, pop: cp.ndarray) -> AdamState:
        """
        Create the leaf tensors and the torch optimiser, once per run.

        Returned to the caller rather than stored on self -- see AdamState.
        torch is imported here, not at module scope, so this module still
        imports when the optional `gd` extra is not installed.
        """
        import torch

        P, T, _ = pop.shape
        dev = "cuda"
        xy = torch.zeros((P, T, 2), dtype=torch.float32, device=dev)
        yaw = torch.zeros((P, T), dtype=torch.float32, device=dev)
        xy.grad = torch.zeros_like(xy)
        yaw.grad = torch.zeros_like(yaw)

        cls = getattr(torch.optim, self.gd_cfg.torch_optimizer, None)
        if cls is None:
            cls = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD,
                   "rmsprop": torch.optim.RMSprop, "adamw": torch.optim.AdamW}.get(
                self.gd_cfg.torch_optimizer.lower()
            )
        if cls is None:
            raise ValueError(
                f"unknown torch_optimizer {self.gd_cfg.torch_optimizer!r}"
            )

        opt = cls([
            {"params": [xy],  "lr": self.gd_cfg.lr_xy},
            {"params": [yaw], "lr": float(np.deg2rad(self.gd_cfg.lr_yaw_deg))},
        ])
        return AdamState(xy=xy, yaw=yaw, opt=opt)

    def step(
        self, pop: cp.ndarray, grad: cp.ndarray, adam: AdamState
    ) -> cp.ndarray:
        """
        One optimiser step. Returns a new (P, T, 3); `pop` is not modified.

        `pop` is the source of truth, the leaf tensors are scratch: the
        projection chain runs between steps and moves the population, so leaves
        that were canonical would silently discard that repair. Only Adam's
        (m, v, t) persists in `adam`, keyed to leaves created once by
        init_adam().

        The CuPy <-> torch handoffs go through dlpack and share memory outright;
        ascontiguousarray is needed because pop[:, :, :2] is a strided view.
        """
        import torch

        adam.xy.data.copy_(torch.from_dlpack(cp.ascontiguousarray(pop[:, :, :2])))
        adam.yaw.data.copy_(torch.from_dlpack(cp.ascontiguousarray(pop[:, :, 2])))
        adam.xy.grad.copy_(torch.from_dlpack(cp.ascontiguousarray(grad[:, :, :2])))
        adam.yaw.grad.copy_(torch.from_dlpack(cp.ascontiguousarray(grad[:, :, 2])))

        adam.opt.step()

        out = pop.copy()
        out[:, :, :2] = cp.from_dlpack(adam.xy.detach())
        out[:, :, 2] = cp.from_dlpack(adam.yaw.detach())
        return self._clip_to_bounds(out)

    # ──────────────────────────────────────────────────────────────────
    # Main optimisation loop
    # ──────────────────────────────────────────────────────────────────

    def run(
        self,
        verbose: bool = True,
        seed_layout: np.ndarray | None = None,
        **kwargs,
    ) -> tuple[cp.ndarray, list[float], np.ndarray | None, cp.ndarray | None]:
        """
        Descend to completion. This is exactly the loop written out in
        example_optimizer_gd.py.

        The aep/objectives logged at row `it` are those of the point the
        gradient was taken at, i.e. *before* that iteration's step -- the only
        honest pairing without a third evaluator call per iteration.
        """
        cfg = self.gd_cfg
        history = list(self._resume_history)

        try:
            if self._resume_pop is not None:
                pop = self._resume_pop
                if verbose:
                    print(f"Resuming at iteration {self._start_gen}")
            else:
                pop = self.project(self.init_population(seed_layout=seed_layout))

            weights = self.weight_vectors()
            z_min, z_max = self.init_ideal()
            adam = self.init_adam(pop)
            obj = None

            for it in range(self._start_gen, cfg.n_generations):
                aep = self.evaluate(pop)
                obj = self.compute_objectives(pop, aep)
                z_min, z_max = self.update_ideal(obj, z_min, z_max)

                self.log_evals(it, pop, aep, objectives=obj)
                self.log(it, pop, aep, objectives=obj)
                history.append(float(cp.max(aep).item()))

                grad = self.spsa_gradient(pop, weights, z_min, z_max)
                pop = self.project(self.step(pop, grad, adam))

                if verbose and it % 10 == 0:
                    g = self.scalarize(obj, weights, z_min, z_max)
                    print(
                        f"Iter {it:4d}  Best AEP: {history[-1]:.4e} kWh  "
                        f"Mean Chebyshev: {float(g.mean()):.4f}"
                    )

            aep = self.evaluate(pop)
            obj = self.compute_objectives(pop, aep)
            best_ind, pareto_obj, best_last = self.best(pop, aep, obj)
            return best_ind, history, pareto_obj, best_last
        finally:
            self.close()
