"""
Genetic algorithm for wind farm layout and yaw optimisation.

The GA is purely a search operator — it contains no physics.
All farm evaluation is delegated to FarmEvaluator.

Multi-objective support: Pareto-based selection with crowding distance.
Full generation history logged for post-processing and convergence analysis.

Operators:
    init   → uniform random initialisation
    project → feasibility repair (projection chain)
    evaluate → FarmEvaluator.evaluate()
    evaluate_objectives → ObjectiveEvaluator.compute_lcoe_batch()
    select  → Pareto-based (rank + crowding distance)
    mutate  → Gaussian perturbation with clipping
    elitism → preserve top-k Pareto-front individuals
"""
from __future__ import annotations
import numpy as np
import cupy as cp
from typing import List

from config import FarmConfig, GAConfig, CostConfig, TurbineConfig, VisualImpactConfig
from physics.farm_evaluator import FarmEvaluator
from physics.objectives import ObjectiveEvaluator
from physics.base import ProjectionOperator
from wind.wind_rose import WindRose
from optimizer.population_logger import AsyncPopulationLogger


class GeneticAlgorithm:
    """
    Batched GA for joint layout + yaw optimisation with multi-objective support.

    Population tensor: (P, T, 3) — [x, y, yaw_rad] per turbine.
    All P individuals are evaluated simultaneously on GPU.

    Multi-objective: Pareto-based selection using rank + crowding distance.
    Full history saved to HDF5 via AsyncPopulationLogger for post-processing.
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
    ) -> None:
        self.farm_cfg   = farm_cfg
        self.ga_cfg     = ga_cfg
        self.evaluator  = evaluator
        self.projection = projection
        self.wind_rose  = wind_rose

        self._max_yaw = np.deg2rad(ga_cfg.max_yaw_deg)

        # Multi-objective
        self.cost_cfg        = cost_cfg or CostConfig()
        self.objectives_mode = objectives_mode
        turb_cfg = TurbineConfig()
        self.obj_eval = ObjectiveEvaluator(farm_cfg, turb_cfg, self.cost_cfg, vi_cfg=vi_cfg)

        # Async HDF5 population logger (None when no history_file given)
        self._logger: AsyncPopulationLogger | None = None
        if history_file:
            genome_size = farm_cfg.n_turbines * 3
            self._logger = AsyncPopulationLogger(
                history_file, ga_cfg.pop_size, genome_size
            )

    # ──────────────────────────────────────────────────────────────────
    # Initialisation
    # ──────────────────────────────────────────────────────────────────

    def init_population(self, seed_layout: np.ndarray | None = None) -> cp.ndarray:
        """
        Return (P, T, 3) initial population.

        If seed_layout (N, 2) is provided the first individual is initialised
        from those positions with zero yaw; the rest are randomised as usual.
        """
        P, T = self.ga_cfg.pop_size, self.farm_cfg.n_turbines
        pop  = cp.zeros((P, T, 3), dtype=cp.float32)

        pop[:, :, 0] = cp.random.uniform(0, self.farm_cfg.area_width,  (P, T))
        pop[:, :, 1] = cp.random.uniform(0, self.farm_cfg.area_height, (P, T))
        pop[:, :, 2] = cp.random.uniform(-self._max_yaw, self._max_yaw, (P, T))

        if seed_layout is not None:
            pop[0, :, :2] = cp.asarray(seed_layout[:T].astype(np.float32))
            pop[0, :,  2] = 0.0

        return pop

    # ──────────────────────────────────────────────────────────────────
    # GA operators
    # ──────────────────────────────────────────────────────────────────

    def project(self, pop: cp.ndarray) -> cp.ndarray:
        """Apply the feasibility projection chain to positions."""
        xy = pop[:, :, :2]
        xy = self.projection.project(xy)
        pop = pop.copy()
        pop[:, :, :2] = xy
        return pop

    def evaluate(self, pop: cp.ndarray) -> cp.ndarray:
        """Return AEP (P,) for the full population."""
        return self.evaluator.evaluate(pop, self.wind_rose)

    def select(self, pop: cp.ndarray, fitness: cp.ndarray) -> cp.ndarray:
        """Rank truncation: keep top POP_SIZE individuals."""
        P = self.ga_cfg.pop_size
        idx = cp.argsort(fitness)[-P:]
        return pop[idx]

    def mutate(self, pop: cp.ndarray) -> cp.ndarray:
        """Gaussian mutation on positions and yaw angles."""
        rate = cp.float32(self.ga_cfg.mutation_rate)
        P, T, _ = pop.shape

        pop = pop.copy()

        # Position mutation
        noise_xy  = cp.random.normal(0, 50.0, (P, T, 2)).astype(cp.float32)
        mask_xy   = (cp.random.rand(P, T, 2) < rate).astype(cp.float32)
        pop[:, :, :2] += mask_xy * noise_xy

        # Yaw mutation (σ = 3°)
        noise_yaw = cp.random.normal(0, np.deg2rad(3), (P, T)).astype(cp.float32)
        mask_yaw  = (cp.random.rand(P, T) < rate).astype(cp.float32)
        pop[:, :, 2] += mask_yaw * noise_yaw

        # Clip to domain
        pop[:, :, 0] = cp.clip(pop[:, :, 0], 0, self.farm_cfg.area_width)
        pop[:, :, 1] = cp.clip(pop[:, :, 1], 0, self.farm_cfg.area_height)
        pop[:, :, 2] = cp.clip(pop[:, :, 2], -self._max_yaw, self._max_yaw)

        return pop

    # ──────────────────────────────────────────────────────────────────
    # Multi-objective evaluation
    # ──────────────────────────────────────────────────────────────────

    def compute_objectives(
        self, pop: cp.ndarray, aep: cp.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute LCOE and VI objectives for each individual.

        Args:
            pop:  (P, T, 3) population [x, y, yaw]
            aep:  (P,) AEP values in kWh

        Returns:
            lcoe: (P,) LCOE in EUR/MWh
            vi:   (P,) visual impact (currently 0 for all)
        """
        P, T, _ = pop.shape

        # Single D2H transfer — no per-individual transfers
        pop_np = cp.asnumpy(pop)
        aep_np = cp.asnumpy(aep) if isinstance(aep, cp.ndarray) else aep

        x = pop_np[:, :, 0]  # (P, T)
        y = pop_np[:, :, 1]

        # Vectorized cable length: sum of turbine distances from farm centroid
        center_x = x.mean(axis=1, keepdims=True)  # (P, 1)
        center_y = y.mean(axis=1, keepdims=True)
        cable_length_km = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2).sum(axis=1) / 1000

        aep_gwh = aep_np / 1e6  # kWh → GWh
        vi_vals = self.obj_eval.compute_vi_batch(x, y, self.wind_rose)

        if self.objectives_mode == "aep_vi":
            # Minimise -AEP (= maximise AEP) and minimise VI
            obj1 = (-aep_gwh).astype(np.float32)
        else:
            obj1 = self.obj_eval.compute_lcoe_batch(T, aep_gwh, cable_length_km)

        return obj1, vi_vals

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
        Select top n_select individuals using Pareto rank + crowding distance.

        Args:
            pop:         (P, T, 3) population
            objectives:  (P, M) objective values
            n_select:    number to select
            ranks:       precomputed ranks (avoids a second sort call when provided)
            distances:   precomputed crowding distances

        Returns:
            selected_pop:     (n_select, T, 3)
            selected_obj_idx: (n_select,) indices into original population
        """
        if ranks is None or distances is None:
            ranks, distances = self.fast_nondominated_sort(objectives)

        scores = ranks.astype(np.float32) * 1e6 - distances.astype(np.float32)
        selected_idx = np.argsort(scores)[:min(n_select, len(scores))]

        # Index directly on GPU — no D2H/H2D round-trip
        return pop[cp.asarray(selected_idx)], selected_idx

    # ──────────────────────────────────────────────────────────────────
    # Main optimisation loop
    # ──────────────────────────────────────────────────────────────────

    def run(
        self,
        verbose: bool = True,
        seed_layout: np.ndarray | None = None,
        multi_objective: bool = False,
    ) -> tuple[cp.ndarray, list[float], np.ndarray | None]:
        """
        Run the genetic algorithm with multi-objective support.

        Args:
            verbose:          print progress
            seed_layout:      initial layout for first individual
            multi_objective:  use Pareto selection (True) or single-objective AEP (False)

        Returns:
            best_individual:   (T, 3) Pareto-optimal or AEP-best layout
            history:           list of best AEP per generation
            pareto_objectives: (n_pareto, 2) LCOE and VI of final Pareto front
        """
        pop = self.init_population(seed_layout=seed_layout)
        history: List[float] = []

        # Project once before the loop so elites are never re-projected.
        # Projection is moved to after mutation each generation — offspring are
        # repaired, then elites (already valid) are reinserted without touching them.
        pop = self.project(pop)

        try:
            for g in range(self.ga_cfg.n_generations):
                # pop is already projected from the previous iteration (or init above)

                # Fitness evaluation
                aep = self.evaluate(pop)

                n_elites_to_keep = self.ga_cfg.elite
                elites = None

                if multi_objective:
                    lcoe, vi = self.compute_objectives(pop, aep)
                    objectives = np.column_stack([lcoe, vi])

                    # Log with objectives so analyze_history can compute VI/HV convergence
                    if self._logger is not None:
                        pop_np = cp.asnumpy(pop).reshape(pop.shape[0], -1)
                        aep_np = cp.asnumpy(aep).astype(np.float32)
                        self._logger.log(g, pop_np, aep_np, objectives=objectives)

                    best_aep = float(cp.max(aep).item())
                    history.append(best_aep)

                    # Sort once — reuse for elites and selection
                    ranks, distances = self.fast_nondominated_sort(objectives)

                    # Preserve the entire rank-0 Pareto front as elites.
                    # Saving only ga_cfg.elite (e.g. 6) individuals lets the other
                    # front members get mutated away each generation, causing the
                    # front to collapse and fluctuate rather than monotonically grow.
                    pareto_idx = np.where(ranks == 0)[0]
                    # Cap at half the population to leave room for exploration.
                    max_elites = min(len(pareto_idx), self.ga_cfg.pop_size // 2)
                    if len(pareto_idx) > max_elites:
                        # When over the cap, prefer the most diverse (highest crowding distance).
                        crowd_order = np.argsort(-distances[pareto_idx])[:max_elites]
                        elites_np_idx = pareto_idx[crowd_order]
                    else:
                        elites_np_idx = pareto_idx
                    n_elites_to_keep = len(elites_np_idx)
                    elites = pop[cp.asarray(elites_np_idx)].copy()

                    # Pass precomputed ranks — avoids a second sort inside pareto_select
                    pop, _ = self.pareto_select(
                        pop, objectives, self.ga_cfg.pop_size,
                        ranks=ranks, distances=distances,
                    )

                    front_size = int((ranks == 0).sum())

                else:
                    if self._logger is not None:
                        pop_np = cp.asnumpy(pop).reshape(pop.shape[0], -1)
                        aep_np = cp.asnumpy(aep).astype(np.float32)
                        self._logger.log(g, pop_np, aep_np)

                    best_aep = float(cp.max(aep).item())
                    history.append(best_aep)

                    elite_idx = cp.argsort(aep)[-self.ga_cfg.elite:]
                    elites = pop[elite_idx].copy()

                    pop = self.select(pop, aep)

                # Mutation
                pop = self.mutate(pop)

                # Project mutated offspring — elites are reinserted after this so
                # they are never re-projected (preserves their AEP exactly).
                pop = self.project(pop)

                # Reinsert elites
                if elites is not None:
                    pop[:n_elites_to_keep] = elites[:n_elites_to_keep]

                if verbose and g % 10 == 0:
                    if multi_objective:
                        best_vi = float(objectives[:, 1].min())
                        print(
                            f"Gen {g:4d}  Best AEP: {best_aep:.4e} kWh  "
                            f"Best VI: {best_vi:.4f}  "
                            f"Pareto: {front_size}"
                        )
                    else:
                        print(f"Gen {g:4d}  Best AEP: {best_aep:.4e} kWh")

            # Final evaluation
            pop = self.project(pop)
            aep = self.evaluate(pop)

            if multi_objective:
                lcoe, vi = self.compute_objectives(pop, aep)
                objectives = np.column_stack([lcoe, vi])

                best_aep_idx = int(cp.argmax(aep).item())
                ranks, _ = self.fast_nondominated_sort(objectives)
                pareto_idx = np.where(ranks == 0)[0]
                pareto_obj = objectives[pareto_idx]

                # Best-VI individual: Pareto member with minimum VI
                best_vi_local = int(np.argmin(pareto_obj[:, 1]))
                best_vi_idx   = pareto_idx[best_vi_local]
                best_vi_ind   = pop[cp.asarray([best_vi_idx])][0]

                return pop[best_aep_idx], history, pareto_obj, best_vi_ind
            else:
                best_idx = int(cp.argmax(aep).item())
                return pop[best_idx], history, None, None

        finally:
            if self._logger is not None:
                self._logger.close()
