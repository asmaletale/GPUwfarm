"""
Full gpuwfarm_opt usage — GeneticAlgorithm wired up with a gpuwfarm_core
FarmEvaluator and a feasibility-repair projection chain, per the optimizer's
own composition pattern (gpuwfarm_opt/main.py).

Small pop/generations on purpose: this is meant to be opened in VSCode and
stepped through (e.g. breakpoint inside GeneticAlgorithm.run or
FarmEvaluator.evaluate), not to produce a converged layout.
"""
import cupy as cp

from gpuwfarm_core import FarmEvaluator, WindRose, WakeConfig, FarmConfig, TurbineConfig, TurbineData
from gpuwfarm_opt.config import GAConfig
from gpuwfarm_opt.genetic import GeneticAlgorithm
from gpuwfarm_opt.projection.base import CompositeProjection
from gpuwfarm_opt.projection.spacing import PairwiseSpacingProjection
from gpuwfarm_opt.projection.boundary import BoundaryProjection

N_TURBINES = 6

farm_cfg = FarmConfig(n_turbines=N_TURBINES)
wake_cfg = WakeConfig(combination="SOSFS")
turbine_cfg = TurbineConfig()
turbine_data = TurbineData.nrel_5mw()
ga_cfg = GAConfig(pop_size=32, n_generations=10)
wind_rose = WindRose.default_12sector()

evaluator = FarmEvaluator(farm_cfg, turbine_cfg, wake_cfg, turbine_data)
projection = CompositeProjection([
    PairwiseSpacingProjection(farm_cfg, n_passes=10),
    BoundaryProjection(farm_cfg),
])

ga = GeneticAlgorithm(farm_cfg, ga_cfg, evaluator, projection, wind_rose)
best, history, pareto, _ = ga.run(verbose=True, multi_objective=False)

print(f"Best AEP: {history[-1]:.4e} kWh (generations: {len(history)})")
print(f"Best layout (x, y, yaw):\n{cp.asnumpy(best)}")
