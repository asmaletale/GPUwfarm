"""
Genetic algorithm for wind farm layout and yaw optimisation.

The GA is purely a search operator — it contains no physics.
All farm evaluation is delegated to FarmEvaluator.

Every stage is a public, self-free method returning a real CuPy array, and
``run()`` is nothing but those methods called in a loop. Write the loop out
yourself when you want the intermediates or an extra step of your own — see
example_optimizer.py / example_optimizer_mo.py at the repo root.

Operators, in the order run() calls them:
    init_population      → uniform random initialisation (or seeded)
    project              → feasibility repair (projection chain)
    evaluate             → FarmEvaluator.evaluate() → AEP (P,)
    compute_objectives   → LCOE/-AEP and visual impact → (P, 2)
    fast_nondominated_sort → Pareto ranks + crowding distance
    tournament_pareto    → binary tournament on (rank, crowding) → mating pool
    tournament_aep       → binary tournament on AEP → mating pool
    crossover            → whole-turbine uniform crossover
    mutate               → Gaussian perturbation with clipping
    survive_pareto       → (mu + lambda) merge, NSGA-II environmental selection
    survive_aep          → (mu + lambda) merge, truncation on AEP
    log                  → one generation to HDF5 via AsyncPopulationLogger
    best                 → pull the result out of a finished population

Selection is elitist through the merge in survive_*: parents and offspring are
concatenated to 2P and truncated back to P, so the best individual can never be
lost and no separate elite-reinsertion step is needed.
"""
from __future__ import annotations
import numpy as np
import cupy as cp

from gpuwfarm_core.config import FarmConfig, CostConfig, VisualImpactConfig
from gpuwfarm_opt.config import GAConfig
from gpuwfarm_core.physics.farm_evaluator import FarmEvaluator
from gpuwfarm_core.objectives import ObjectiveEvaluator
from gpuwfarm_opt.projection.base import ProjectionOperator
from gpuwfarm_core.wind.wind_rose import WindRose
from gpuwfarm_opt.population_logger import AsyncPopulationLogger


class GeneticAlgorithm:
    """
    Batched GA for joint layout + yaw optimisation with multi-objective support.

    Population tensor: (P, T, 3) — [x, y, yaw_rad] per turbine.
    All P individuals are evaluated simultaneously on GPU.

    ``GAConfig.optimize`` picks the decision variables: "both" (default),
    "layout" (yaw pinned to 0) or "yaw" (one fixed layout for all individuals).

    Multi-objective: NSGA-II — non-dominated sorting, crowding distance, binary
    tournament and an elitist (mu + lambda) merge.

    The instance holds configuration only; no method assigns to self (apart from
    the logger's lifecycle), so the stage methods can be called in any order from
    a script. When a history_file is given, use the instance as a context manager
    (or call close()) so the HDF5 writer thread is shut down:

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
        history_file:    str | None = None,
        evals_file:      str | None = None,
    ) -> None:
        self.farm_cfg   = farm_cfg
        self.ga_cfg     = ga_cfg
        self.evaluator  = evaluator
        self.projection = projection
        self.wind_rose  = wind_rose

        # float32: a float64 scalar here would promote the yaw column in the
        # cp.random.uniform / cp.clip calls below.
        self._max_yaw = np.float32(np.deg2rad(ga_cfg.max_yaw_deg))
        self._sigma_yaw = np.float32(np.deg2rad(ga_cfg.sigma_yaw_deg))

        # Which decision variables are free (GAConfig.optimize)
        self.opt_layout = ga_cfg.optimize in ("both", "layout")
        self.opt_yaw    = ga_cfg.optimize in ("both", "yaw")

        # Multi-objective. The turbine config comes off the evaluator so the
        # objectives cannot silently disagree with the physics about rotor size.
        self.cost_cfg        = cost_cfg or CostConfig()
        self.objectives_mode = objectives_mode
        self.obj_eval = ObjectiveEvaluator(
            farm_cfg, evaluator.turbine_cfg, self.cost_cfg, vi_cfg=vi_cfg
        )

        # Async HDF5 loggers (None when the corresponding file is not given).
        # _logger      — one row per generation: the surviving population.
        # _eval_logger — one row per evaluation batch: every genome the evaluator
        #                ever saw (row 0 = initial population, row g+1 = the
        #                offspring of generation g). Survivors are a subset of
        #                these, so this file alone is the complete search history.
        genome_size = farm_cfg.n_turbines * 3
        self._logger: AsyncPopulationLogger | None = (
            AsyncPopulationLogger(history_file, ga_cfg.pop_size, genome_size)
            if history_file else None
        )
        self._eval_logger: AsyncPopulationLogger | None = (
            AsyncPopulationLogger(evals_file, ga_cfg.pop_size, genome_size)
            if evals_file else None
        )

    # ──────────────────────────────────────────────────────────────────
    # Logger lifecycle
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

    def __enter__(self) -> "GeneticAlgorithm":
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
        P, T = self.ga_cfg.pop_size, self.farm_cfg.n_turbines
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

    # ──────────────────────────────────────────────────────────────────
    # GA operators
    # ──────────────────────────────────────────────────────────────────

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
    # Multi-objective evaluation
    # ──────────────────────────────────────────────────────────────────

    def compute_objectives(self, pop: cp.ndarray, aep: cp.ndarray) -> np.ndarray:
        """
        Compute the objective matrix for each individual, minimisation convention.

        Everything through obj1/vi_vals stays GPU-resident (cable length, LCOE,
        and VI are all computed with CuPy) -- the only D2H transfer is the final
        (P, 2) matrix below, needed because fast_nondominated_sort/pareto_select
        do small-matrix Pareto ranking on NumPy. That is a tiny transfer (2*P
        floats) versus a per-generation cp.asnumpy() of the full (P, T, 3)
        population.

        Args:
            pop:  (P, T, 3) population [x, y, yaw]
            aep:  (P,) AEP values in kWh

        Returns:
            (P, 2) float32 NumPy — column 0 is LCOE in EUR/MWh (or -AEP in GWh
            when objectives_mode == "aep_vi"), column 1 is visual impact.
        """
        P, T, _ = pop.shape

        x = pop[:, :, 0]  # (P, T) cupy
        y = pop[:, :, 1]

        # Vectorized cable length: sum of turbine distances from farm centroid
        center_x = x.mean(axis=1, keepdims=True)  # (P, 1)
        center_y = y.mean(axis=1, keepdims=True)
        cable_length_km = cp.sqrt((x - center_x) ** 2 + (y - center_y) ** 2).sum(axis=1) / cp.float32(1000.0)

        aep_gwh = aep / cp.float32(1e6)  # kWh → GWh
        vi_vals = self.obj_eval.compute_vi_batch(x, y, self.wind_rose)

        if self.objectives_mode == "aep_vi":
            # Minimise -AEP (= maximise AEP) and minimise VI
            obj1 = (-aep_gwh).astype(cp.float32)
        else:
            obj1 = self.obj_eval.compute_lcoe_batch(T, aep_gwh, cable_length_km)

        return np.column_stack([
            cp.asnumpy(obj1).astype(np.float32),
            cp.asnumpy(vi_vals).astype(np.float32),
        ])

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

        distances = GeneticAlgorithm._crowding_distance(objectives, ranks)
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
        load-bearing now that survive_pareto truncates 2P down to P.

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
    # Survival: (mu + lambda) elitist merge
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
    # History logging and results
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
            aep:        (P,) its AEP, as returned by survive_*
            objectives: (P, 2) its objectives, or None for single-objective runs

        Returns:
            best_aep_individual: (T, 3) highest-AEP individual
            pareto_objectives:   (n_pareto, 2) rank-0 objectives, or None
            best_vi_individual:  (T, 3) lowest-VI Pareto member, or None
        """
        best_ind = pop[int(cp.argmax(aep).item())]
        if objectives is None:
            return best_ind, None, None

        ranks, _ = self.fast_nondominated_sort(objectives)
        pareto_idx = np.where(ranks == 0)[0]
        pareto_obj = objectives[pareto_idx]

        # Best-VI individual: Pareto member with minimum VI
        best_vi_idx = int(pareto_idx[int(np.argmin(pareto_obj[:, 1]))])
        return best_ind, pareto_obj, pop[best_vi_idx]

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
        history: list[float] = []

        try:
            pop = self.project(self.init_population(seed_layout=seed_layout))
            aep = self.evaluate(pop)
            obj = self.compute_objectives(pop, aep) if multi_objective else None
            self.log_evals(0, pop, aep, objectives=obj)

            for g in range(cfg.n_generations):
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
