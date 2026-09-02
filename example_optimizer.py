"""
The GA generation loop written out — reference for gpuwfarm_opt, single objective.

`GeneticAlgorithm.run()` is exactly the loop below. Writing it out yourself
gives you the population, the fitness, the mating pool and the offspring as
named CuPy arrays at every step, so you can inspect them, log them, or drop your
own operator in between two existing ones.

The contract: every call takes a (P, T, 3) population and returns one. Nothing
is lazy. To add a procedure, add a line that takes the previous return:

    children = my_local_search(children, evaluator, wind_rose)

To replace a stage, just call your own function instead of ga.<stage> — no
subclassing, no registration, no config flag.

Selection is elitist: survive_aep merges the P parents with the P offspring and
keeps the best P, so the best individual can never be lost and history is
monotonically non-decreasing.

See example_optimizer_mo.py for the multi-objective (NSGA-II) loop, which
differs only in the three marked lines.
"""
import numpy as np
import cupy as cp

from gpuwfarm_core import (
    FarmEvaluator, WindRose, WakeConfig, FarmConfig, TurbineConfig, TurbineData,
)
from gpuwfarm_opt.config import GAConfig
from gpuwfarm_opt.genetic import GeneticAlgorithm
from gpuwfarm_opt.projection.base import CompositeProjection
from gpuwfarm_opt.projection.spacing import PairwiseSpacingProjection
from gpuwfarm_opt.projection.boundary import BoundaryProjection

N_TURBINES = 6

# Decision variables: "both", "layout" (yaw pinned to 0), or
# "yaw" (layout frozen — seed_layout if given, else one random layout shared
# by the whole population).
OPTIMIZE = "both"

farm_cfg     = FarmConfig(n_turbines=N_TURBINES)
wake_cfg     = WakeConfig(combination="SOSFS")
turbine_cfg  = TurbineConfig()
turbine_data = TurbineData.nrel_5mw()
ga_cfg       = GAConfig(pop_size=32, n_generations=20, optimize=OPTIMIZE)
wind_rose    = WindRose.default_12sector()

# Only meaningful for OPTIMIZE = "yaw" / "both": a starting grid layout.
seed_layout = np.stack(
    np.meshgrid(np.linspace(300, 1700, 3), np.linspace(600, 1400, 2)), -1
).reshape(-1, 2).astype(np.float32)

evaluator = FarmEvaluator(farm_cfg, turbine_cfg, wake_cfg, turbine_data)
projection = CompositeProjection([
    PairwiseSpacingProjection(farm_cfg, n_passes=10),
    BoundaryProjection(farm_cfg),
])

# ga holds configuration only — every method below is a pure operator on arrays.
# Used as a context manager so the HDF5 history writer is always shut down;
# without history_file that is a no-op, but it costs nothing to be consistent.
with GeneticAlgorithm(farm_cfg, ga_cfg, evaluator, projection, wind_rose) as ga:

    # ══════════════════════════════════════════════════════════════════
    # Initial population
    # ══════════════════════════════════════════════════════════════════
    pop = ga.init_population(seed_layout=seed_layout)   # (P, T, 3) float32
    pop = ga.project(pop)                               # (P, T, 3) feasibility repair
    aep = ga.evaluate(pop)                              # (P,)      float32 kWh

    history = []

    # ══════════════════════════════════════════════════════════════════
    # Generations
    # ══════════════════════════════════════════════════════════════════
    for g in range(ga_cfg.n_generations):

        parents  = ga.tournament_aep(pop, aep)          # (P, T, 3)  <- MO: tournament_pareto
        children = ga.crossover(parents)                # (P, T, 3)
        children = ga.mutate(children)                  # (P, T, 3)
        children = ga.project(children)                 # (P, T, 3)

        # ─── Add your own step here ──────────────────────────────────
        # Takes the previous return, gives back a (P, T, 3) array:
        #   children = my_local_search(children, evaluator, wind_rose)
        # ─────────────────────────────────────────────────────────────

        aep_c = ga.evaluate(children)                   # (P,) float32 kWh

        # Elitist (mu + lambda): 2P compete, best P survive.
        pop, aep = ga.survive_aep(pop, aep, children, aep_c)   # <- MO: survive_pareto

        ga.log(g, pop, aep)                             # no-op without history_file
        history.append(float(cp.max(aep).item()))

        if g % 5 == 0:
            print(f"Gen {g:4d}  Best AEP: {history[-1]:.4e} kWh  "
                  f"Mean: {float(cp.mean(aep)):.4e} kWh")

    best, _, _ = ga.best(pop, aep)                      # (T, 3)

print()
print(f"Optimised: {OPTIMIZE}")
print(f"Best AEP:  {history[-1]:.4e} kWh  (generations: {len(history)})")
print(f"Gain over generation 0: {100 * (history[-1] / history[0] - 1):+.2f} %")
print(f"Best layout (x, y, yaw):\n{cp.asnumpy(best)}")

# The merge is elitist, so this must hold. If it ever fails, a survive step or a
# projection is losing the incumbent.
assert all(b >= a - 1.0 for a, b in zip(history, history[1:])), \
    "best-so-far went backwards: survival is not elitist"
print("\nHistory is monotonically non-decreasing (elitist selection works).")
