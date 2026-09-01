"""
Standalone gpuwfarm_core usage — no optimizer involved.

Mirrors the "Standalone evaluation" snippet in CLAUDE.md: build the physics
config, a wind rose, and a population of layouts, then evaluate AEP directly.
This is the shape you'd use to plug FarmEvaluator into something like an RL
loop. Open this file in VSCode and set breakpoints on the evaluate() call to
step through the physics pipeline.
"""
import numpy as np
import cupy as cp

from gpuwfarm_core import FarmEvaluator, WindRose, WakeConfig, FarmConfig, TurbineConfig, TurbineData

N_TURBINES = 3
POP_SIZE = 4

farm_cfg = FarmConfig(n_turbines=N_TURBINES, area_width=2000, area_height=2000)
wake_cfg = WakeConfig(combination="SOSFS")
turbine_cfg = TurbineConfig()
turbine_data = TurbineData.nrel_5mw()

evaluator = FarmEvaluator(farm_cfg, turbine_cfg, wake_cfg, turbine_data)

# Simple aligned row, repeated across the population.
xs = np.linspace(200, 1800, N_TURBINES)
ys = np.full(N_TURBINES, 1000.0)
pop = cp.zeros((POP_SIZE, N_TURBINES, 3), dtype=cp.float32)
pop[:, :, 0] = cp.asarray(xs)
pop[:, :, 1] = cp.asarray(ys)

wind_rose = WindRose.default_12sector()

aep = evaluator.evaluate(pop, wind_rose)                    # (P,)
aep_per_turbine = evaluator.evaluate(pop, wind_rose, per_turbine=True)  # (P, T)

print(f"Farm AEP per individual (kWh): {cp.asnumpy(aep)}")
print(f"Per-turbine AEP, individual 0 (kWh): {cp.asnumpy(aep_per_turbine)[0]}")
