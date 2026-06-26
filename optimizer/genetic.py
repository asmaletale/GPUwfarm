"""
Genetic algorithm for wind farm layout and yaw optimisation.

The GA is purely a search operator — it contains no physics.
All farm evaluation is delegated to FarmEvaluator.

Operators:
    init   → uniform random initialisation
    project → feasibility repair (projection chain)
    evaluate → FarmEvaluator.evaluate()
    select  → rank truncation (keep top POP_SIZE)
    mutate  → Gaussian perturbation with clipping
    elitism → preserve top-k individuals across generations
"""
from __future__ import annotations
import numpy as np
import cupy as cp
from typing import List

from config import FarmConfig, GAConfig
from physics.farm_evaluator import FarmEvaluator
from physics.base import ProjectionOperator
from wind.wind_rose import WindRose


class GeneticAlgorithm:
    """
    Batched GA for joint layout + yaw optimisation.

    Population tensor: (P, T, 3) — [x, y, yaw_rad] per turbine.
    All P individuals are evaluated simultaneously on GPU.
    """

    def __init__(
        self,
        farm_cfg:   FarmConfig,
        ga_cfg:     GAConfig,
        evaluator:  FarmEvaluator,
        projection: ProjectionOperator,
        wind_rose:  WindRose,
    ) -> None:
        self.farm_cfg   = farm_cfg
        self.ga_cfg     = ga_cfg
        self.evaluator  = evaluator
        self.projection = projection
        self.wind_rose  = wind_rose

        self._max_yaw = np.deg2rad(ga_cfg.max_yaw_deg)

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
    # Main optimisation loop
    # ──────────────────────────────────────────────────────────────────

    def run(
        self,
        verbose: bool = True,
        seed_layout: np.ndarray | None = None,
    ) -> tuple[cp.ndarray, list[float]]:
        """
        Run the genetic algorithm.

        Returns:
            best_individual: (T, 3) best layout found
            history:         list of best AEP per generation (Python floats)
        """
        pop     = self.init_population(seed_layout=seed_layout)
        history: List[float] = []

        for g in range(self.ga_cfg.n_generations):
            # Feasibility repair
            pop = self.project(pop)

            # Fitness evaluation
            fit = self.evaluate(pop)

            # Record best
            best_val = float(cp.max(fit).item())
            history.append(best_val)

            # Elitism: preserve top-k before selection/mutation
            elite_idx  = cp.argsort(fit)[-self.ga_cfg.elite:]
            elites     = pop[elite_idx].copy()

            # Selection
            pop = self.select(pop, fit)

            # Mutation
            pop = self.mutate(pop)

            # Reinsert elites
            pop[:self.ga_cfg.elite] = elites

            if verbose and g % 10 == 0:
                print(f"Gen {g:4d}  Best AEP: {best_val:.4e} kWh")

        # Final evaluation to get the best individual
        pop = self.project(pop)
        fit = self.evaluate(pop)
        best_idx = int(cp.argmax(fit).item())

        return pop[best_idx], history
