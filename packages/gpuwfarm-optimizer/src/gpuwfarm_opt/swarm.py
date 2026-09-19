"""
Multi-objective particle swarm optimisation (MOPSO).

Same shapes and the same stage-by-stage contract as the GA: a (P, T, 3)
population in, a (P, T, 3) population out, every stage pure and individually
callable.

PSO carries state the GA does not -- velocity, each particle's personal best,
and an external Pareto archive. None of it is stored on the instance. Each
stage takes the state it reads as an argument and hands the new value back,
exactly the way ``survive_pareto(pop, aep, obj, ...) -> (pop, aep, obj)``
already does. The state lives in the caller's loop as a named CuPy array, so it
can be inspected, logged, or swapped for something of your own:

    vel = pso.velocity(pop, vel, pbest, leaders)
    vel = my_velocity_damping(vel)               # drop your own step in
    pop = pso.project(pso.advance(pop, vel))

What makes the swarm spread along a front instead of collapsing onto one point
is the leader: the social pull is toward a member drawn from the external
archive, chosen with probability proportional to crowding distance, so sparse
regions of the front attract particles.

Reference: Coello, Pulido & Lechuga (2004), "Handling multiple objectives with
particle swarm optimization", IEEE Trans. Evolutionary Computation 8(3).
"""
from __future__ import annotations
from typing import Sequence

import numpy as np
import cupy as cp

from gpuwfarm_core.config import FarmConfig, CostConfig, VisualImpactConfig
from gpuwfarm_core.physics.farm_evaluator import FarmEvaluator
from gpuwfarm_core.wind.wind_rose import WindRose
from gpuwfarm_opt.base import Optimizer
from gpuwfarm_opt.config import PSOConfig
from gpuwfarm_opt.objectives import ObjectiveSet, Objective
from gpuwfarm_opt.projection.base import ProjectionOperator


class ParticleSwarm(Optimizer):
    """
    Batched MOPSO for joint layout + yaw optimisation.

    Population tensor: (P, T, 3) -- [x, y, yaw_rad], P particles evaluated
    simultaneously on GPU. ``PSOConfig.optimize`` picks the decision variables
    exactly as it does for the GA.

    Use as a context manager when a history_file is given, so the HDF5 writer
    thread is shut down:

        with ParticleSwarm(..., history_file="pso.h5") as pso:
            ...
    """

    def __init__(
        self,
        farm_cfg:   FarmConfig,
        pso_cfg:    PSOConfig,
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
            farm_cfg, pso_cfg, evaluator, projection, wind_rose,
            cost_cfg=cost_cfg, vi_cfg=vi_cfg, objectives_mode=objectives_mode,
            objectives=objectives, history_file=history_file,
            evals_file=evals_file, resume=resume,
        )
        self.pso_cfg = pso_cfg
        self.archive_size = pso_cfg.archive_size or pso_cfg.pop_size

        # Per-column velocity cap: x and y are metres on a ~2 km farm, yaw is
        # radians in +/-0.5. One scalar cap cannot serve both, so it is a (3,)
        # broadcast over the last axis. float32, or it promotes the batch.
        self._v_max = cp.asarray(
            [pso_cfg.v_max_xy, pso_cfg.v_max_xy, np.deg2rad(pso_cfg.v_max_yaw_deg)],
            dtype=cp.float32,
        )
        # Zeroes the columns that are not decision variables, so a frozen yaw
        # or a frozen layout can never drift through the velocity update.
        self._free = cp.asarray(
            [float(self.opt_layout), float(self.opt_layout), float(self.opt_yaw)],
            dtype=cp.float32,
        )

    # ──────────────────────────────────────────────────────────────────
    # Swarm operators
    # ──────────────────────────────────────────────────────────────────

    def init_swarm(self, pop: cp.ndarray) -> cp.ndarray:
        """
        Return the initial (P, T, 3) velocity: uniform in +/- v_max per column.

        Frozen columns come back exactly 0, so they stay frozen for the whole
        run without any further gating.
        """
        vel = cp.random.uniform(-1.0, 1.0, pop.shape).astype(cp.float32)
        return vel * self._v_max * self._free

    def velocity(
        self,
        pop:     cp.ndarray,
        vel:     cp.ndarray,
        pbest:   cp.ndarray,
        leaders: cp.ndarray,
    ) -> cp.ndarray:
        """
        One velocity update, clipped per column.

            v' = w*v + c1*r1*(pbest - pop) + c2*r2*(leader - pop)

        r1 and r2 are drawn per component, not per particle -- the standard
        form, and the one that keeps the swarm from moving along a single
        direction in lockstep.

        Args:
            pop:     (P, T, 3) current positions
            vel:     (P, T, 3) current velocity
            pbest:   (P, T, 3) each particle's own best-so-far position
            leaders: (P, T, 3) archive member each particle is pulled toward

        Returns:
            (P, T, 3) new velocity. `vel` is not modified.
        """
        cfg = self.pso_cfg
        r1 = cp.random.rand(*pop.shape).astype(cp.float32)
        r2 = cp.random.rand(*pop.shape).astype(cp.float32)

        new_vel = (
            cp.float32(cfg.inertia) * vel
            + cp.float32(cfg.c1) * r1 * (pbest - pop)
            + cp.float32(cfg.c2) * r2 * (leaders - pop)
        )
        new_vel = cp.clip(new_vel, -self._v_max, self._v_max)
        return (new_vel * self._free).astype(cp.float32)

    def advance(self, pop: cp.ndarray, vel: cp.ndarray) -> cp.ndarray:
        """
        Move the swarm one step and clip to the box bounds.

        Feasibility repair is deliberately *not* done here -- call project()
        afterwards, so the projection chain stays one visible line in the loop
        rather than something hidden inside the position update.

        Returns a new (P, T, 3); `pop` is not modified.
        """
        return self._clip_to_bounds((pop + vel).astype(cp.float32))

    @staticmethod
    def _dominates(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Row-wise Pareto dominance for two (N, M) minimisation matrices."""
        return np.all(a <= b, axis=1) & np.any(a < b, axis=1)

    def update_pbest(
        self,
        pop:       cp.ndarray,
        obj:       np.ndarray,
        pbest:     cp.ndarray,
        pbest_obj: np.ndarray,
    ) -> tuple[cp.ndarray, np.ndarray]:
        """
        Per-particle personal best, by Pareto dominance.

        A particle's new position replaces its personal best only when it
        dominates it. On mutual non-domination the incumbent is kept, which
        makes the update deterministic -- the usual alternative, a coin flip,
        makes runs irreproducible for no measurable benefit.

        Args:
            pop:       (P, T, 3) new positions
            obj:       (P, M) their objectives, minimisation
            pbest:     (P, T, 3) incumbent personal bests
            pbest_obj: (P, M) their objectives

        Returns:
            (pbest (P, T, 3) cupy, pbest_obj (P, M) numpy) -- new arrays.
        """
        better = self._dominates(obj, pbest_obj)          # (P,) numpy bool
        new_obj = np.where(better[:, None], obj, pbest_obj)
        mask = cp.asarray(better)[:, None, None]
        return cp.where(mask, pop, pbest), new_obj.astype(np.float32)

    def archive_update(
        self,
        archive:     cp.ndarray | None,
        archive_obj: np.ndarray | None,
        pop:         cp.ndarray,
        obj:         np.ndarray,
    ) -> tuple[cp.ndarray, np.ndarray]:
        """
        Fold the swarm into the bounded external Pareto archive.

        Concatenate archive + swarm, keep the non-dominated set, and if that
        exceeds archive_size prune by *ascending* crowding distance -- the
        densest members go first, which is what preserves spread along the
        front rather than letting it clump.

        Pass archive=None, archive_obj=None on the first call.

        Args:
            archive:     (A, T, 3) or None
            archive_obj: (A, M) or None
            pop:         (P, T, 3) current swarm
            obj:         (P, M) its objectives

        Returns:
            (archive (A', T, 3) cupy, archive_obj (A', M) numpy), A' <= archive_size
        """
        if archive is None or archive_obj is None or len(archive_obj) == 0:
            merged, merged_obj = pop, obj
        else:
            merged = cp.concatenate([archive, pop], axis=0)
            merged_obj = np.vstack([archive_obj, obj])

        ranks, distances = self.fast_nondominated_sort(merged_obj)
        front = np.where(ranks == 0)[0]

        if len(front) > self.archive_size:
            # Densest first: ascending crowding distance, keep the sparse tail.
            front = front[np.argsort(distances[front])][-self.archive_size:]

        return merged[cp.asarray(front)], merged_obj[front].astype(np.float32)

    def select_leaders(
        self,
        archive:     cp.ndarray,
        archive_obj: np.ndarray,
        n:           int,
    ) -> cp.ndarray:
        """
        Draw n leaders from the archive, favouring sparse regions of the front.

        Roulette with probability proportional to crowding distance, so a
        particle is most often pulled toward an under-populated part of the
        front. Infinite distances (the front's extremes) are mapped to twice the
        largest finite value: that keeps the extremes attractive without letting
        them take all the probability mass.

        Flip the sense here if you want leaders drawn from dense regions instead
        -- it is the one line that decides whether the swarm spreads or converges.

        Returns:
            (n, T, 3) -- one leader per particle, gathered on the GPU.
        """
        n_arch = archive_obj.shape[0]
        # The archive is all rank 0 by construction, so it is a single front.
        d = self._crowding_distance(archive_obj, np.zeros(n_arch, dtype=np.int32))

        finite = d[np.isfinite(d)]
        cap = 2.0 * finite.max() if finite.size and finite.max() > 0 else 1.0
        d = np.where(np.isfinite(d), d, cap)

        total = d.sum()
        p = d / total if total > 0 else np.full(n_arch, 1.0 / n_arch)

        idx = np.random.choice(n_arch, size=n, replace=True, p=p)
        return archive[cp.asarray(idx)]

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
        Run the swarm to completion. This is exactly the loop written out in
        example_optimizer_pso.py.

        MOPSO is inherently multi-objective, so unlike the GA there is no
        single-objective switch: objectives are always computed and the archive
        is always maintained.

        The result is taken from the *archive*, not the final swarm -- the
        archive is what the search actually found, and particles keep moving
        after passing through good positions.

        ponytail: on resume the velocity, personal bests and archive all restart
        from the checkpointed swarm rather than being restored; only the
        population is checkpointed, in the same spirit as the RNG. The archive
        could be rebuilt from the logged rows (self._resume_obj holds them) if a
        resumed run ever turns out to lose too much front.
        """
        cfg = self.pso_cfg
        history = list(self._resume_history)

        try:
            if self._resume_pop is not None:
                pop = self._resume_pop
                if verbose:
                    print(f"Resuming at iteration {self._start_gen}")
            else:
                pop = self.project(self.init_population(seed_layout=seed_layout))

            vel = self.init_swarm(pop)
            aep = self.evaluate(pop)
            obj = self.compute_objectives(pop, aep)
            if self._resume_pop is None:
                self.log_evals(0, pop, aep, objectives=obj)

            pbest, pbest_obj = pop, obj
            archive, archive_obj = self.archive_update(None, None, pop, obj)

            for it in range(self._start_gen, cfg.n_generations):
                leaders = self.select_leaders(archive, archive_obj, pop.shape[0])
                vel = self.velocity(pop, vel, pbest, leaders)
                pop = self.project(self.advance(pop, vel))

                aep = self.evaluate(pop)
                obj = self.compute_objectives(pop, aep)
                self.log_evals(it + 1, pop, aep, objectives=obj)

                pbest, pbest_obj = self.update_pbest(pop, obj, pbest, pbest_obj)
                archive, archive_obj = self.archive_update(
                    archive, archive_obj, pop, obj
                )

                self.log(it, pop, aep, objectives=obj)
                history.append(float(cp.max(aep).item()))

                if verbose and it % 10 == 0:
                    print(
                        f"Iter {it:4d}  Best AEP: {history[-1]:.4e} kWh  "
                        f"Archive: {len(archive_obj):3d}"
                    )

            best_idx = int(cp.argmax(self.evaluate(archive)).item())
            return (
                archive[best_idx],
                history,
                archive_obj,
                archive[int(np.argmin(archive_obj[:, -1]))],
            )
        finally:
            self.close()
