"""
The NSGA-II generation loop written out — reference for gpuwfarm_opt,
multi-objective (LCOE vs. visual impact).

Same shape as example_optimizer.py; only three lines differ, marked <<< below.
The objective matrix `obj` is (P, 2) NumPy, minimisation convention:
    column 0 = LCOE in EUR/MWh   (or -AEP in GWh when objectives_mode="aep_vi")
    column 1 = visual impact

Textbook NSGA-II, in the order the loop runs it:
    fast_nondominated_sort  → ranks (0 = Pareto front) + crowding distance
    tournament_pareto       → binary tournament on (rank, crowding)
    crossover / mutate      → offspring
    survive_pareto          → merge 2P, fill whole fronts by rank, break the
                              last partial front by descending crowding distance

Run it directly:
    python example_optimizer_mo.py
"""
import numpy as np
import cupy as cp

from gpuwfarm_core import (
    FarmEvaluator, WindRose, WakeConfig, FarmConfig, TurbineConfig, TurbineData,
    CostConfig, VisualImpactConfig,
)
from gpuwfarm_opt.config import GAConfig
from gpuwfarm_opt.genetic import GeneticAlgorithm
from gpuwfarm_opt.projection.base import CompositeProjection
from gpuwfarm_opt.projection.spacing import PairwiseSpacingProjection
from gpuwfarm_opt.projection.boundary import BoundaryProjection

N_TURBINES = 6

farm_cfg     = FarmConfig(n_turbines=N_TURBINES)
wake_cfg     = WakeConfig(combination="SOSFS")
turbine_cfg  = TurbineConfig()
turbine_data = TurbineData.nrel_5mw()
ga_cfg       = GAConfig(pop_size=32, n_generations=20, optimize="both")
wind_rose    = WindRose.default_12sector()

evaluator = FarmEvaluator(farm_cfg, turbine_cfg, wake_cfg, turbine_data)
projection = CompositeProjection([
    PairwiseSpacingProjection(farm_cfg, n_passes=10),
    BoundaryProjection(farm_cfg),
])

# vi_cfg is what switches visual impact on; without it column 1 is all zeros
# and every individual ends up on the same front.
ga = GeneticAlgorithm(
    farm_cfg, ga_cfg, evaluator, projection, wind_rose,
    cost_cfg=CostConfig(), vi_cfg=VisualImpactConfig(),
    objectives_mode="lcoe_vi",          # or "aep_vi" to trade AEP against VI
)

with ga:
    # ══════════════════════════════════════════════════════════════════
    # Initial population
    # ══════════════════════════════════════════════════════════════════
    pop = ga.project(ga.init_population())      # (P, T, 3) float32
    aep = ga.evaluate(pop)                      # (P,)      float32 kWh
    obj = ga.compute_objectives(pop, aep)       # (P, 2)    float32 numpy   <<<

    history, front_sizes = [], []

    # ══════════════════════════════════════════════════════════════════
    # Generations
    # ══════════════════════════════════════════════════════════════════
    for g in range(ga_cfg.n_generations):

        ranks, crowding = ga.fast_nondominated_sort(obj)      # (P,) int32, (P,) float  <<<
        parents  = ga.tournament_pareto(pop, ranks, crowding)  # (P, T, 3)               <<<
        children = ga.crossover(parents)                       # (P, T, 3)
        children = ga.mutate(children)                         # (P, T, 3)
        children = ga.project(children)                        # (P, T, 3)

        # ─── Add your own step here ──────────────────────────────────
        # Takes the previous return, gives back a (P, T, 3) array:
        #   children = my_local_search(children, evaluator, wind_rose)
        # ─────────────────────────────────────────────────────────────

        aep_c = ga.evaluate(children)                          # (P,)   float32
        obj_c = ga.compute_objectives(children, aep_c)          # (P, 2) numpy

        # Elitist (mu + lambda): 2P compete, best P survive by rank + crowding.
        pop, aep, obj = ga.survive_pareto(
            pop, aep, obj, children, aep_c, obj_c
        )                                                       #                        <<<

        ga.log(g, pop, aep, objectives=obj)
        history.append(float(cp.max(aep).item()))
        front_sizes.append(int((ga.fast_nondominated_sort(obj)[0] == 0).sum()))

        if g % 5 == 0:
            print(f"Gen {g:4d}  Best AEP: {history[-1]:.4e} kWh  "
                  f"LCOE: {obj[:, 0].min():.2f} EUR/MWh  "
                  f"VI: {obj[:, 1].min():.4f}  "
                  f"Pareto: {front_sizes[-1]}")

    best_aep_ind, pareto_obj, best_vi_ind = ga.best(pop, aep, obj)

print()
print(f"Best AEP:      {history[-1]:.4e} kWh  (generations: {len(history)})")
print(f"Pareto front:  {len(pareto_obj)} individuals")
print(f"  LCOE range:  {pareto_obj[:, 0].min():.2f} .. {pareto_obj[:, 0].max():.2f} EUR/MWh")
print(f"  VI range:    {pareto_obj[:, 1].min():.4f} .. {pareto_obj[:, 1].max():.4f}")
print(f"Best-AEP layout (x, y, yaw):\n{cp.asnumpy(best_aep_ind)}")
print(f"Best-VI layout (x, y, yaw):\n{cp.asnumpy(best_vi_ind)}")

# Elitist merge: the best AEP seen can never be lost.
assert all(b >= a - 1.0 for a, b in zip(history, history[1:])), \
    "best-so-far went backwards: survival is not elitist"
print("\nHistory is monotonically non-decreasing (elitist selection works).")
