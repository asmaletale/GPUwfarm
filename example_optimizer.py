"""
Full gpuwfarm_opt usage — GeneticAlgorithm wired up with a gpuwfarm_core
FarmEvaluator and a feasibility-repair projection chain, per the optimizer's
own composition pattern (gpuwfarm_opt/main.py).

Set OPTIMIZE below to pick what the GA searches over: layout + yaw jointly,
layout only, or yaw only on a fixed layout.

Small pop/generations on purpose: this is meant to be opened in VSCode and
stepped through (e.g. breakpoint inside GeneticAlgorithm.run or
FarmEvaluator.evaluate), not to produce a converged layout.
"""
import numpy as np
import cupy as cp

from gpuwfarm_core import FarmEvaluator, WindRose, WakeConfig, FarmConfig, TurbineConfig, TurbineData
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

farm_cfg = FarmConfig(n_turbines=N_TURBINES)
wake_cfg = WakeConfig(combination="SOSFS")
turbine_cfg = TurbineConfig()
turbine_data = TurbineData.nrel_5mw()
ga_cfg = GAConfig(pop_size=32, n_generations=10, optimize=OPTIMIZE)
wind_rose = WindRose.default_12sector()

# Only meaningful for OPTIMIZE = "yaw" / "both": a starting grid layout.
seed_layout = np.stack(
    np.meshgrid(np.linspace(300, 1700, 3), np.linspace(600, 1400, 2)), -1
).reshape(-1, 2).astype(np.float32)

evaluator = FarmEvaluator(farm_cfg, turbine_cfg, wake_cfg, turbine_data)
projection = CompositeProjection([
    PairwiseSpacingProjection(farm_cfg, n_passes=10),
    BoundaryProjection(farm_cfg),
])

ga = GeneticAlgorithm(farm_cfg, ga_cfg, evaluator, projection, wind_rose)
best, history, pareto, _ = ga.run(
    verbose=True, multi_objective=False, seed_layout=seed_layout
)

print(f"Optimised: {OPTIMIZE}")
print(f"Best AEP: {history[-1]:.4e} kWh (generations: {len(history)})")
print(f"Best layout (x, y, yaw):\n{cp.asnumpy(best)}")
