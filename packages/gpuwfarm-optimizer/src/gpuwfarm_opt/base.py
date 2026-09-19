"""
Shared machinery for every optimizer in this package.

``Optimizer`` holds what genetic search, particle swarms and gradient descent
all need and none of them should re-implement: candidate initialisation,
feasibility projection, evaluation, the objective matrix, Pareto ranking and
selection, the HDF5 history/checkpoint lifecycle, and pulling results out of a
finished population. Each algorithm subclasses it and adds only its own
operators.

The contract every subclass keeps -- and the reason the algorithms are pleasant
to drive by hand:

  * The instance holds **configuration only**. No stage method assigns to
    ``self`` (the logger lifecycle is the one exception, and it is a resource,
    not search state).
  * Algorithm state that has to survive between iterations -- a swarm's
    velocity, an optimiser's momentum, a Pareto archive -- is passed *in* as an
    argument and handed *back* as a return value, exactly the way ``pop`` and
    ``aep`` already are. It lives in the caller's loop as a named array, where
    it can be inspected, logged or replaced.
  * Every stage returns a real CuPy array. Nothing is lazy, there is no graph.

So ``run()`` is never more than its own stage methods called in order, and you
can write that loop out yourself whenever you want the intermediates or a step
of your own. See example_optimizer*.py at the repo root.
"""
from __future__ import annotations
import os
from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
import cupy as cp
import h5py

from gpuwfarm_core.config import FarmConfig, CostConfig, VisualImpactConfig
from gpuwfarm_core.objectives import ObjectiveEvaluator
from gpuwfarm_core.physics.farm_evaluator import FarmEvaluator
from gpuwfarm_core.wind.wind_rose import WindRose
from gpuwfarm_opt.config import OptimizerConfig
from gpuwfarm_opt.objectives import ObjectiveSet, Objective, default_objective_set
from gpuwfarm_opt.population_logger import AsyncPopulationLogger
from gpuwfarm_opt.projection.base import ProjectionOperator


class Optimizer(ABC):
    """
    Base for the batched optimizers. Population tensor is (P, T, 3) -- x, y, yaw_rad.

    ``cfg.optimize`` picks the decision variables: "both" (default), "layout"
    (yaw pinned to 0) or "yaw" (one fixed layout shared by the whole batch).

    When a history_file is given, use the instance as a context manager (or call
    close()) so the HDF5 writer thread is shut down:

        with ParticleSwarm(..., history_file="hist.h5") as pso:
            ...
    """

    def __init__(
        self,
        farm_cfg:   FarmConfig,
        cfg:        OptimizerConfig,
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
        self.farm_cfg   = farm_cfg
        self.cfg        = cfg
        self.evaluator  = evaluator
        self.projection = projection
        self.wind_rose  = wind_rose

        # float32: a float64 scalar here would promote the yaw column in the
        # cp.random.uniform / cp.clip calls downstream.
        self._max_yaw = np.float32(np.deg2rad(cfg.max_yaw_deg))

        # Which decision variables are free (OptimizerConfig.optimize)
        self.opt_layout = cfg.optimize in ("both", "layout")
        self.opt_yaw    = cfg.optimize in ("both", "yaw")

        # Multi-objective. The turbine config comes off the evaluator so the
        # objectives cannot silently disagree with the physics about rotor size.
        self.cost_cfg        = cost_cfg or CostConfig()
        self.objectives_mode = objectives_mode
        self.obj_eval = ObjectiveEvaluator(
            farm_cfg, evaluator.turbine_cfg, self.cost_cfg, vi_cfg=vi_cfg
        )
        # Which objectives are in play is configuration. Passing `objectives=`
        # overrides objectives_mode entirely; otherwise the named built-in pair
        # is used, reproducing the pre-pluggable columns exactly.
        self.objectives: ObjectiveSet = (
            objectives if isinstance(objectives, ObjectiveSet)
            else ObjectiveSet(objectives) if objectives is not None
            else default_objective_set(objectives_mode, self.obj_eval)
        )

        # Async HDF5 loggers (None when the corresponding file is not given).
        # _logger      — one row per iteration: the population that survived it.
        # _eval_logger — one row per evaluation batch: every genome the evaluator
        #                ever saw. Survivors are a subset of these, so this file
        #                alone is the complete search history.
        genome_size = farm_cfg.n_turbines * 3

        # Read the checkpoint *before* the loggers open the file for writing —
        # HDF5 will not hand out a read handle while the writer thread holds it.
        self._resume_pop, self._start_gen, self._resume_history = (
            self._load_checkpoint(history_file, cfg.pop_size, genome_size)
            if resume and history_file else (None, 0, [])
        )
        # Objectives of every logged row, for subclasses that need to restore
        # more than the population (a Pareto archive, a running ideal point).
        # Same pre-logger ordering constraint as _load_checkpoint.
        self._resume_obj = (
            self._load_objectives(history_file) if resume and history_file else None
        )

        # Append rather than truncate when continuing an existing history.
        append = self._resume_pop is not None
        self._logger: AsyncPopulationLogger | None = (
            AsyncPopulationLogger(history_file, cfg.pop_size, genome_size, append)
            if history_file else None
        )
        self._eval_logger: AsyncPopulationLogger | None = (
            AsyncPopulationLogger(evals_file, cfg.pop_size, genome_size, append)
            if evals_file else None
        )

    # ──────────────────────────────────────────────────────────────────
    # Checkpointing
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _load_checkpoint(
        history_file: str, pop_size: int, genome_size: int
    ) -> tuple[cp.ndarray | None, int, list[float]]:
        """
        Read the last surviving population out of an existing history file.

        The history file *is* the checkpoint — the logger flushes every row, so
        whatever generations reached disk before an interruption are resumable,
        even after a SIGKILL that ran no cleanup. A missing or empty file simply
        starts from scratch, which is what makes `resume=True` safe to leave on.

        Returns:
            pop:       (P, T, 3) last logged population, or None to start fresh
            start_gen: the generation to resume at (= number of rows on disk)
            history:   best AEP per already-completed generation
        """
        if not os.path.exists(history_file):
            return None, 0, []

        with h5py.File(history_file, "r") as f:
            if "genomes" not in f or f["genomes"].shape[0] == 0:
                return None, 0, []
            stored_pop, stored_genome = f["genomes"].shape[1:]
            if (stored_pop, stored_genome) != (pop_size, genome_size):
                raise ValueError(
                    f"'{history_file}' holds (pop={stored_pop}, genome={stored_genome}) "
                    f"but this GA is configured for (pop={pop_size}, genome={genome_size}). "
                    "Resume needs matching pop_size and n_turbines; point --history-file "
                    "at a different path to start a new run."
                )
            start_gen = int(f["genomes"].shape[0])
            pop = f["genomes"][-1].reshape(pop_size, -1, 3)
            history = f["fitnesses"][:].max(axis=1).tolist()

        return cp.asarray(pop, dtype=cp.float32), start_gen, history

    @staticmethod
    def _load_objectives(history_file: str | None) -> np.ndarray | None:
        """
        Read every logged row's objective matrix out of an existing history.

        Subclasses use this to restore state that the population alone does not
        carry — a Pareto archive, or a running ideal point for scalarisation.
        Like _load_checkpoint, it must run *before* the loggers open the file.

        Returns:
            (n_rows, P, M) float32, or None when the file or dataset is absent
            (a single-objective run writes no "objectives" dataset at all).
        """
        if not history_file or not os.path.exists(history_file):
            return None
        with h5py.File(history_file, "r") as f:
            if "objectives" not in f or f["objectives"].shape[0] == 0:
                return None
            return f["objectives"][:]

    # ──────────────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────────────

    def close(self) -> None:
        """
        Flush and close the history loggers. Idempotent, so it is safe to call
        after run() (which closes them itself) or twice from nested scopes.
        """
        if self._logger is not None:
            self._logger.close()
            self._logger = None
        if self._eval_logger is not None:
            self._eval_logger.close()
            self._eval_logger = None

    def __enter__(self) -> "Optimizer":
        return self

    def __exit__(self, *exc_info) -> bool:
        self.close()
        return False


    # ──────────────────────────────────────────────────────────────────
    # Initialisation
    # ──────────────────────────────────────────────────────────────────

    def init_population(self, seed_layout: np.ndarray | None = None) -> cp.ndarray:
        """
        Return (P, T, 3) initial population.

        If seed_layout (N, 2) is provided the first individual is initialised
        from those positions with zero yaw; the rest are randomised as usual.

        In "yaw" mode the layout is not a decision variable: every individual
        shares one fixed layout — the seed layout if given, otherwise a single
        random (feasibility-repaired) one. In "layout" mode yaw stays 0.
        """
        P, T = self.cfg.pop_size, self.farm_cfg.n_turbines
        pop  = cp.zeros((P, T, 3), dtype=cp.float32)

        pop[:, :, 0] = cp.random.uniform(0, self.farm_cfg.area_width,  (P, T))
        pop[:, :, 1] = cp.random.uniform(0, self.farm_cfg.area_height, (P, T))
        if self.opt_yaw:
            pop[:, :, 2] = cp.random.uniform(-self._max_yaw, self._max_yaw, (P, T))

        if seed_layout is not None:
            pop[0, :, :2] = cp.asarray(seed_layout[:T].astype(np.float32))
            pop[0, :,  2] = 0.0

        if not self.opt_layout:
            # Fixed layout: repair it once here, since project() is a no-op after this.
            pop[:, :, :2] = self.projection.project(pop[:1, :, :2])

        return pop


    def project(self, pop: cp.ndarray) -> cp.ndarray:
        """Apply the feasibility projection chain to positions."""
        if not self.opt_layout:
            return pop  # positions never move — already repaired in init_population
        xy = pop[:, :, :2]
        xy = self.projection.project(xy)
        pop = pop.copy()
        pop[:, :, :2] = xy
        return pop

    def evaluate(self, pop: cp.ndarray) -> cp.ndarray:
        """Return AEP (P,) in kWh for the full population."""
        # In "layout" mode the yaw column is 0 by construction, which lets the
        # evaluator skip the wake-deflection model entirely.
        return self.evaluator.evaluate(pop, self.wind_rose, zero_yaw=not self.opt_yaw)

    def _clip_to_bounds(self, pop: cp.ndarray) -> cp.ndarray:
        """
        Clip positions into the farm rectangle and yaw to +/- max_yaw, in place.

        Box bounds only — minimum spacing is not a box constraint and is
        enforced by the projection chain, which every optimizer runs as its own
        visible step. Frozen columns are left alone: in "layout" mode the yaw
        column is 0 by construction, and in "yaw" mode positions never move.

        Mutates and returns `pop`; callers pass a population they already own.
        """
        if self.opt_layout:
            pop[:, :, 0] = cp.clip(pop[:, :, 0], 0, self.farm_cfg.area_width)
            pop[:, :, 1] = cp.clip(pop[:, :, 1], 0, self.farm_cfg.area_height)
        if self.opt_yaw:
            pop[:, :, 2] = cp.clip(pop[:, :, 2], -self._max_yaw, self._max_yaw)
        return pop

    # ──────────────────────────────────────────────────────────────────
    # Objectives
    # ──────────────────────────────────────────────────────────────────

    def compute_objectives(self, pop: cp.ndarray, aep: cp.ndarray) -> np.ndarray:
        """
        Compute the objective matrix for each individual, minimisation convention.

        Thin delegate to the ObjectiveSet -- which objectives are in play, and
        how many, is configuration, not GA logic. See gpuwfarm_opt.objectives.

        Args:
            pop:  (P, T, 3) population [x, y, yaw]
            aep:  (P,) AEP values in kWh

        Returns:
            (P, M) float32 NumPy. With the default objectives_mode="lcoe_vi",
            column 0 is LCOE in EUR/MWh (or -AEP in GWh for "aep_vi") and column
            1 is visual impact -- unchanged from before objectives were pluggable.
        """
        return self.objectives(pop, aep, self.wind_rose)


    # ──────────────────────────────────────────────────────────────────
    # Pareto ranking and selection
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def fast_nondominated_sort(
        objectives: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Fast non-dominated sorting (Deb et al., 2002).

        Args:
            objectives: (P, M) array — minimisation assumed

        Returns:
            ranks:     (P,) rank of each individual (0 = Pareto front)
            distances: (P,) crowding distance
        """
        P, M = objectives.shape

        # Vectorized (P, P) dominance matrix — replaces O(P²) Python nested loop
        obj_i = objectives[:, np.newaxis, :]   # (P, 1, M)
        obj_j = objectives[np.newaxis, :, :]   # (1, P, M)
        dominates = np.all(obj_i <= obj_j, axis=2) & np.any(obj_i < obj_j, axis=2)
        np.fill_diagonal(dominates, False)

        domination_count = dominates.sum(axis=0).astype(np.int32)  # (P,)
        ranks     = np.full(P, -1, dtype=np.int32)
        remaining = np.ones(P, dtype=bool)
        current_rank = 0

        while remaining.any():
            # Gate on `remaining` so already-ranked individuals are never re-selected
            # (fixes rank-overwrite bug: without this, ranks were overwritten each iter)
            front = remaining & (domination_count == 0)
            if not front.any():
                ranks[remaining] = current_rank  # degenerate: mutually non-dominating
                break
            ranks[front] = current_rank
            remaining[front] = False
            # Vectorized decrement: count front members that dominate each j
            domination_count -= dominates[front, :].sum(axis=0)
            current_rank += 1

        distances = Optimizer._crowding_distance(objectives, ranks)
        return ranks, distances

    @staticmethod
    def _crowding_distance(objectives: np.ndarray, ranks: np.ndarray) -> np.ndarray:
        """
        Calculate crowding distance for each individual.

        Note the extremes of *every* front get np.inf, not just the rank-0 ones.
        Any consumer must therefore treat rank as the primary key and crowding
        distance only as a within-front tiebreak — see pareto_select.

        Args:
            objectives: (P, M) objective values
            ranks:      (P,) domination rank

        Returns:
            distances: (P,) crowding distance
        """
        P, M = objectives.shape
        distances = np.zeros(P)

        for rank in np.unique(ranks):
            front_idx = np.where(ranks == rank)[0]
            if len(front_idx) <= 2:
                distances[front_idx] = np.inf
                continue

            front_objs = objectives[front_idx]

            for m in range(M):
                sorted_local = np.argsort(front_objs[:, m])
                sorted_front = front_idx[sorted_local]

                distances[sorted_front[0]]  = np.inf
                distances[sorted_front[-1]] = np.inf

                obj_range = front_objs[sorted_local[-1], m] - front_objs[sorted_local[0], m]
                if obj_range > 1e-10:
                    # Vectorized neighbor-difference — replaces inner Python loop
                    numerator = front_objs[sorted_local[2:], m] - front_objs[sorted_local[:-2], m]
                    distances[sorted_front[1:-1]] += numerator / obj_range

        return distances

    def pareto_select(
        self,
        pop: cp.ndarray,
        objectives: np.ndarray,
        n_select: int,
        ranks: np.ndarray | None = None,
        distances: np.ndarray | None = None,
    ) -> tuple[cp.ndarray, np.ndarray]:
        """
        NSGA-II environmental selection: keep n_select individuals by filling
        whole fronts in rank order, breaking the last (partial) front by
        descending crowding distance.

        The ordering is a single lexsort with rank as the primary key. Scoring it
        instead as `rank * 1e6 - distance` looks equivalent but is not: crowding
        distance is np.inf for the extremes of *every* front, so any front's
        extreme point scored -inf and outranked the rank-0 interior. That was
        latent while n_select == len(pop) made this a pure reorder; it is
        load-bearing now that environmental selection truncates 2P down to P.

        Args:
            pop:         (P, T, 3) population
            objectives:  (P, M) objective values
            n_select:    number to select
            ranks:       precomputed ranks (avoids a second sort call when provided)
            distances:   precomputed crowding distances

        Returns:
            selected_pop:     (n_select, T, 3)
            selected_obj_idx: (n_select,) indices into the original population
        """
        if ranks is None or distances is None:
            ranks, distances = self.fast_nondominated_sort(objectives)

        # Last key is primary: rank ascending, then crowding distance descending.
        order = np.lexsort((-distances, ranks))
        selected_idx = order[:min(n_select, len(order))]

        # Index directly on GPU — no D2H/H2D round-trip
        return pop[cp.asarray(selected_idx)], selected_idx


    # ──────────────────────────────────────────────────────────────────
    # Logging and results
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _write(
        logger:     AsyncPopulationLogger | None,
        row:        int,
        pop:        cp.ndarray,
        aep:        cp.ndarray,
        objectives: np.ndarray | None,
    ) -> None:
        """D2H copy + queue one row. No-op when the logger is None."""
        if logger is None:
            return
        logger.log(
            row,
            cp.asnumpy(pop).reshape(pop.shape[0], -1),
            cp.asnumpy(aep).astype(np.float32),
            objectives=objectives,
        )

    def log(
        self,
        generation: int,
        pop:        cp.ndarray,
        aep:        cp.ndarray,
        objectives: np.ndarray | None = None,
    ) -> None:
        """
        Write one generation's surviving population to the HDF5 history. No-op
        when the GA was built without a history_file.

        The generation number is the dataset row address, so a hand-written loop
        must pass a monotonically increasing value. This is the only place the
        full (P, T, 3) population crosses to the host — by far the largest
        transfer in the loop, which is why it is opt-in.
        """
        self._write(self._logger, generation, pop, aep, objectives)

    def log_evals(
        self,
        row:        int,
        pop:        cp.ndarray,
        aep:        cp.ndarray,
        objectives: np.ndarray | None = None,
    ) -> None:
        """
        Write one *evaluated batch* to the evals history — the complete search
        record, including the offspring that lose the survival merge and are
        therefore absent from log(). No-op without an evals_file.

        run() uses row 0 for the initial population and row g+1 for the offspring
        of generation g; a hand-written loop owns the row counter itself.
        """
        self._write(self._eval_logger, row, pop, aep, objectives)

    def best(
        self,
        pop: cp.ndarray,
        aep: cp.ndarray,
        objectives: np.ndarray | None = None,
    ) -> tuple[cp.ndarray, np.ndarray | None, cp.ndarray | None]:
        """
        Pull the results out of a finished population.

        Args:
            pop:        (P, T, 3) final population
            aep:        (P,) its AEP
            objectives: (P, M) its objectives, or None for single-objective runs

        Returns:
            best_aep_individual: (T, 3) highest-AEP individual
            pareto_objectives:   (n_pareto, M) rank-0 objectives, or None
            best_last_objective: (T, 3) Pareto member minimising the *last*
                                 objective, or None

        With the default two-objective set the last objective is visual impact,
        so the third element is the lowest-VI Pareto member exactly as before.
        best_per_objective() gives one representative per objective when M > 2.
        """
        best_ind = pop[int(cp.argmax(aep).item())]
        if objectives is None:
            return best_ind, None, None

        ranks, _ = self.fast_nondominated_sort(objectives)
        pareto_idx = np.where(ranks == 0)[0]
        pareto_obj = objectives[pareto_idx]

        best_last_idx = int(pareto_idx[int(np.argmin(pareto_obj[:, -1]))])
        return best_ind, pareto_obj, pop[best_last_idx]

    def best_per_objective(
        self, pop: cp.ndarray, objectives: np.ndarray
    ) -> dict[str, cp.ndarray]:
        """
        The Pareto-front member that minimises each objective, by name.

        The general-M companion to best(): with three objectives you get three
        entries, so nothing downstream has to assume a two-column layout.

        Args:
            pop:        (P, T, 3) population
            objectives: (P, M) its objectives, minimisation convention

        Returns:
            {objective_name: (T, 3)} — one representative per objective
        """
        ranks, _ = self.fast_nondominated_sort(objectives)
        pareto_idx = np.where(ranks == 0)[0]
        front = objectives[pareto_idx]
        return {
            name: pop[int(pareto_idx[int(np.argmin(front[:, m]))])]
            for m, name in enumerate(self.objectives.names)
        }

    # ──────────────────────────────────────────────────────────────────
    # Main optimisation loop
    # ──────────────────────────────────────────────────────────────────

    @abstractmethod
    def run(
        self,
        verbose: bool = True,
        seed_layout: np.ndarray | None = None,
        **kwargs,
    ) -> tuple[cp.ndarray, list[float], np.ndarray | None, cp.ndarray | None]:
        """
        Run the search to completion.

        Every optimizer returns the same 4-tuple, so a caller can swap one for
        another without touching the code that consumes the result:

            best_individual:   (T, 3) highest-AEP individual
            history:           list[float] best AEP per iteration
            pareto_objectives: (n_pareto, M) rank-0 objectives, or None
            best_last_obj:     (T, 3) Pareto member minimising the last
                               objective, or None
        """
        ...
