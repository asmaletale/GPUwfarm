"""
Genetic algorithm for wind farm layout and yaw optimisation.

The GA is purely a search operator — it contains no physics. All farm
evaluation is delegated to FarmEvaluator, and everything it shares with the
other optimizers (initialisation, projection, evaluation, objectives, Pareto
ranking, logging, checkpointing) lives on the Optimizer base class.

Every stage is a public, self-free method returning a real CuPy array, and
``run()`` is nothing but those methods called in a loop. Write the loop out
yourself when you want the intermediates or an extra step of your own — see
example_optimizer.py / example_optimizer_mo.py at the repo root.

Operators this class adds, in the order run() calls them:
    tournament_pareto    → binary tournament on (rank, crowding) → mating pool
    tournament_aep       → binary tournament on AEP → mating pool
    crossover            → whole-turbine uniform crossover
    mutate               → Gaussian perturbation with clipping
    survive_pareto       → (mu + lambda) merge, NSGA-II environmental selection
    survive_aep          → (mu + lambda) merge, truncation on AEP

Inherited from Optimizer: init_population, project, evaluate,
compute_objectives, fast_nondominated_sort, pareto_select, log, log_evals, best.

Selection is elitist through the merge in survive_*: parents and offspring are
concatenated to 2P and truncated back to P, so the best individual can never be
lost and no separate elite-reinsertion step is needed.
"""
from __future__ import annotations
from typing import Sequence

import numpy as np
import cupy as cp

from gpuwfarm_core.config import FarmConfig, CostConfig, VisualImpactConfig
from gpuwfarm_core.physics.farm_evaluator import FarmEvaluator
from gpuwfarm_core.wind.wind_rose import WindRose
from gpuwfarm_opt.base import Optimizer
from gpuwfarm_opt.config import GAConfig
from gpuwfarm_opt.objectives import ObjectiveSet, Objective
from gpuwfarm_opt.projection.base import ProjectionOperator


class GeneticAlgorithm(Optimizer):
    """
    Batched GA for joint layout + yaw optimisation with multi-objective support.

    Population tensor: (P, T, 3) — [x, y, yaw_rad]. All P individuals are
    evaluated simultaneously on GPU.

    ``GAConfig.optimize`` picks the decision variables: "both" (default),
    "layout" (yaw pinned to 0) or "yaw" (one fixed layout for all individuals).

    Multi-objective: NSGA-II — non-dominated sorting, crowding distance, binary
    tournament and an elitist (mu + lambda) merge.

    The instance holds configuration only; no method assigns to self (apart from
    the logger's lifecycle), so the stage methods can be called in any order
    from a script. When a history_file is given, use the instance as a context
    manager (or call close()) so the HDF5 writer thread is shut down:

        with GeneticAlgorithm(..., history_file="hist.h5") as ga:
            ...
    """

    def __init__(
        self,
        farm_cfg:   FarmConfig,
        ga_cfg:     GAConfig,
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
            farm_cfg, ga_cfg, evaluator, projection, wind_rose,
            cost_cfg=cost_cfg, vi_cfg=vi_cfg, objectives_mode=objectives_mode,
            objectives=objectives, history_file=history_file,
            evals_file=evals_file, resume=resume,
        )
        # Readable alias for the base's generic `cfg`; same object.
        self.ga_cfg = ga_cfg

        # float32: a float64 scalar would promote the yaw column in mutate().
        # GA-only — it is the mutation step size, not a bound.
        self._sigma_yaw = np.float32(np.deg2rad(ga_cfg.sigma_yaw_deg))

    # ──────────────────────────────────────────────────────────────────
    # GA operators
    # ──────────────────────────────────────────────────────────────────

    def crossover(self, pop: cp.ndarray) -> cp.ndarray:
        """
        Whole-turbine uniform crossover over adjacent pairs.

        Each even/odd pair crosses over with probability crossover_rate. Within a
        crossing pair, each turbine is independently drawn from either parent with
        50 % probability — (x, y, yaw) is kept together so spatial coherence is
        preserved. An odd individual (when P is odd) passes through unchanged.

        Pairing is positional, not shuffled: the mating pool comes out of
        tournament_* in random order already, so an extra permutation here would
        only cost a gather.
        """
        P, T, _ = pop.shape
        rate      = self.ga_cfg.crossover_rate
        gene_rate = self.ga_cfg.gene_swap_rate or (1.0 / T)  # default: 1/T → ~1 swap/pair

        n_pairs = P // 2
        idx_a = cp.arange(0, 2 * n_pairs, 2)  # even positions
        idx_b = cp.arange(1, 2 * n_pairs, 2)  # odd positions

        # (n_pairs, 1) bool — which pairs actually cross over
        do_cross = (cp.random.rand(n_pairs) < rate)[:, cp.newaxis]

        # (n_pairs, T) bool — which turbines swap between parents
        # gene_rate controls how many turbines exchange per pair;
        # 1/T keeps spatial structure largely intact (≈1 turbine swapped on average)
        swap = (cp.random.rand(n_pairs, T) < gene_rate) & do_cross
        swap3 = swap[:, :, cp.newaxis]  # (n_pairs, T, 1) — broadcasts over dim 3

        a = pop[idx_a]  # (n_pairs, T, 3)
        b = pop[idx_b]  # (n_pairs, T, 3)

        pop = pop.copy()
        pop[idx_a] = cp.where(swap3, b, a)
        pop[idx_b] = cp.where(swap3, a, b)

        return pop

    def mutate(self, pop: cp.ndarray) -> cp.ndarray:
        """
        Gaussian mutation on positions and yaw angles.

        Step sizes are GAConfig.sigma_xy (metres) and GAConfig.sigma_yaw_deg —
        sigma_xy is farm-scale dependent, so it wants setting per case study.
        """
        rate = cp.float32(self.ga_cfg.mutation_rate)
        P, T, _ = pop.shape

        pop = pop.copy()

        # Position mutation
        if self.opt_layout:
            noise_xy  = cp.random.normal(0, self.ga_cfg.sigma_xy, (P, T, 2)).astype(cp.float32)
            mask_xy   = (cp.random.rand(P, T, 2) < rate).astype(cp.float32)
            pop[:, :, :2] += mask_xy * noise_xy
            pop[:, :, 0] = cp.clip(pop[:, :, 0], 0, self.farm_cfg.area_width)
            pop[:, :, 1] = cp.clip(pop[:, :, 1], 0, self.farm_cfg.area_height)

        # Yaw mutation
        if self.opt_yaw:
            noise_yaw = cp.random.normal(0, self._sigma_yaw, (P, T)).astype(cp.float32)
            mask_yaw  = (cp.random.rand(P, T) < rate).astype(cp.float32)
            pop[:, :, 2] += mask_yaw * noise_yaw
            pop[:, :, 2] = cp.clip(pop[:, :, 2], -self._max_yaw, self._max_yaw)

        return pop


    # ──────────────────────────────────────────────────────────────────
    # Selection
    # ──────────────────────────────────────────────────────────────────

    def tournament_pareto(
        self, pop: cp.ndarray, ranks: np.ndarray, distances: np.ndarray
    ) -> cp.ndarray:
        """
        Binary tournament on the crowded-comparison operator: lower rank wins,
        ties broken by larger crowding distance.

        Runs on NumPy because ranks/distances are already host arrays; only the
        winning index vector goes back to the GPU, for one gather.

        Args:
            pop:       (P, T, 3) population
            ranks:     (P,) from fast_nondominated_sort
            distances: (P,) from fast_nondominated_sort

        Returns:
            (P, T, 3) mating pool, in random order
        """
        n = pop.shape[0]
        a = np.random.randint(0, n, n)
        b = np.random.randint(0, n, n)
        a_wins = (ranks[a] < ranks[b]) | (
            (ranks[a] == ranks[b]) & (distances[a] > distances[b])
        )
        return pop[cp.asarray(np.where(a_wins, a, b))]

    def tournament_aep(self, pop: cp.ndarray, aep: cp.ndarray) -> cp.ndarray:
        """
        Binary tournament on AEP (higher wins). Stays entirely on the GPU.

        Args:
            pop: (P, T, 3) population
            aep: (P,) fitness in kWh

        Returns:
            (P, T, 3) mating pool, in random order
        """
        n = pop.shape[0]
        a = cp.random.randint(0, n, n)
        b = cp.random.randint(0, n, n)
        return pop[cp.where(aep[a] > aep[b], a, b)]


    # ──────────────────────────────────────────────────────────────────
    # Survival
    # ──────────────────────────────────────────────────────────────────

    def survive_pareto(
        self,
        pop:      cp.ndarray,
        aep:      cp.ndarray,
        obj:      np.ndarray,
        children: cp.ndarray,
        aep_c:    cp.ndarray,
        obj_c:    np.ndarray,
    ) -> tuple[cp.ndarray, cp.ndarray, np.ndarray]:
        """
        NSGA-II survival: concatenate parents and offspring to 2P, then keep P by
        rank and crowding distance.

        The merge *is* the elitism — the whole parent Pareto front competes for
        survival, so nothing needs copying aside and blitting back in.

        Returns:
            (pop, aep, obj) for the surviving P individuals, all three consistent
            with one another.
        """
        merged     = cp.concatenate([pop, children])          # (2P, T, 3)
        merged_aep = cp.concatenate([aep, aep_c])             # (2P,)
        merged_obj = np.vstack([obj, obj_c])                  # (2P, M)

        selected, idx = self.pareto_select(merged, merged_obj, self.ga_cfg.pop_size)
        return selected, merged_aep[cp.asarray(idx)], merged_obj[idx]

    def survive_aep(
        self,
        pop:      cp.ndarray,
        aep:      cp.ndarray,
        children: cp.ndarray,
        aep_c:    cp.ndarray,
    ) -> tuple[cp.ndarray, cp.ndarray]:
        """
        Single-objective survival: merge parents and offspring to 2P and keep the
        best P by AEP. Elitist by construction, and never leaves the GPU.

        Returns:
            (pop, aep) for the surviving P individuals.
        """
        merged     = cp.concatenate([pop, children])
        merged_aep = cp.concatenate([aep, aep_c])

        idx = cp.argsort(merged_aep)[::-1][:self.ga_cfg.pop_size]
        return merged[idx], merged_aep[idx]


    # ──────────────────────────────────────────────────────────────────
    # Main optimisation loop
    # ──────────────────────────────────────────────────────────────────

    def run(
        self,
        verbose: bool = True,
        seed_layout: np.ndarray | None = None,
        multi_objective: bool = False,
    ) -> tuple[cp.ndarray, list[float], np.ndarray | None, cp.ndarray | None]:
        """
        Run the genetic algorithm to completion.

        This is the reference loop — the stage methods above called in order.
        Copy it into your own script when you want to inspect or extend it;
        example_optimizer.py and example_optimizer_mo.py are exactly this.

        Args:
            verbose:          print progress every 10 generations
            seed_layout:      (N, 2) initial layout for the first individual
            multi_objective:  NSGA-II on (LCOE, VI) if True, else AEP only

        Returns:
            best_individual:   (T, 3) highest-AEP layout of the final population
            history:           best AEP after each generation (n_generations long)
            pareto_objectives: (n_pareto, 2) final Pareto front, None if single-objective
            best_vi_individual:(T, 3) lowest-VI Pareto member, None if single-objective
        """
        cfg = self.ga_cfg
        history: list[float] = list(self._resume_history)

        try:
            if self._resume_pop is not None:
                # Resumed: the checkpointed population is re-evaluated rather
                # than read back from the file, so aep/objectives always agree
                # with the *current* wind rose and configs. One generation's
                # worth of work, against a whole run saved.
                pop = self._resume_pop
                aep = self.evaluate(pop)
                obj = self.compute_objectives(pop, aep) if multi_objective else None
                if verbose:
                    print(f"Resuming at generation {self._start_gen} "
                          f"(best AEP so far: {history[-1]:.4e} kWh)")
            else:
                pop = self.project(self.init_population(seed_layout=seed_layout))
                aep = self.evaluate(pop)
                obj = self.compute_objectives(pop, aep) if multi_objective else None
                self.log_evals(0, pop, aep, objectives=obj)

            for g in range(self._start_gen, cfg.n_generations):
                # Mating pool
                if multi_objective:
                    ranks, distances = self.fast_nondominated_sort(obj)
                    parents = self.tournament_pareto(pop, ranks, distances)
                else:
                    parents = self.tournament_aep(pop, aep)

                # Offspring: crossover → mutation → feasibility repair
                children = self.project(self.mutate(self.crossover(parents)))
                aep_c    = self.evaluate(children)

                # Every genome the evaluator saw, before the merge discards half
                obj_c = self.compute_objectives(children, aep_c) if multi_objective else None
                self.log_evals(g + 1, children, aep_c, objectives=obj_c)

                # Survival (elitist merge of parents + offspring)
                if multi_objective:
                    pop, aep, obj = self.survive_pareto(
                        pop, aep, obj, children, aep_c, obj_c
                    )
                else:
                    pop, aep = self.survive_aep(pop, aep, children, aep_c)

                self.log(g, pop, aep, objectives=obj)
                history.append(float(cp.max(aep).item()))

                if verbose and g % 10 == 0:
                    if multi_objective:
                        print(
                            f"Gen {g:4d}  Best AEP: {history[-1]:.4e} kWh  "
                            f"Best VI: {float(obj[:, 1].min()):.4f}  "
                            f"Pareto: {int((ranks == 0).sum())}"
                        )
                    else:
                        print(f"Gen {g:4d}  Best AEP: {history[-1]:.4e} kWh")

            best_ind, pareto_obj, best_vi_ind = self.best(pop, aep, obj)
            return best_ind, history, pareto_obj, best_vi_ind

        finally:
            self.close()
