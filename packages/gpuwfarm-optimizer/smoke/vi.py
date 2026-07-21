"""Smoke-test for the Visual Impact implementation."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from gpuwfarm_core.config import FarmConfig, TurbineConfig, CostConfig, VisualImpactConfig
from gpuwfarm_core.wind.wind_rose import WindRose
from gpuwfarm_core.objectives import ObjectiveEvaluator

farm_cfg    = FarmConfig(n_turbines=3)
turbine_cfg = TurbineConfig(rotor_diameter=120.0, hub_height=90.0)
cost_cfg    = CostConfig()
vi_cfg      = VisualImpactConfig(
    obs_coords  = [[-1000.0, -1000.0]],
    obs_heights = [1.77],
    obs_weights = [1.0],
)
obj = ObjectiveEvaluator(farm_cfg, turbine_cfg, cost_cfg, vi_cfg=vi_cfg)

wr = WindRose.default_12sector()
x  = np.array([0.0, 500.0, 1000.0], dtype=np.float32)
y  = np.array([0.0, 500.0,    0.0], dtype=np.float32)

vi = obj.compute_visual_impact(x, y, wr)
print(f"Single-layout VI = {vi:.6f}")
assert vi > 0, "VI should be positive for non-trivial layout"

# vi=0 when no vi_cfg
obj_no_vi = ObjectiveEvaluator(farm_cfg, turbine_cfg, cost_cfg, vi_cfg=None)
assert obj_no_vi.compute_visual_impact(x, y, wr) == 0.0

# Batch: P=4 identical layouts → same VI repeated
x_batch = np.tile(x, (4, 1))
y_batch = np.tile(y, (4, 1))
vi_batch = obj.compute_vi_batch(x_batch, y_batch, wr)
print(f"Batch VI = {vi_batch}")
assert vi_batch.shape == (4,), "wrong batch shape"
assert np.allclose(vi_batch, vi_batch[0]), "identical layouts should give equal VI"

# Multi-observer: sum of weights=1 equals same as single observer with weight=1
vi_cfg2 = VisualImpactConfig(
    obs_coords  = [[-1000.0, -1000.0], [-1000.0, -1000.0]],
    obs_heights = [1.77, 1.77],
    obs_weights = [0.5, 0.5],
)
obj2 = ObjectiveEvaluator(farm_cfg, turbine_cfg, cost_cfg, vi_cfg=vi_cfg2)
vi2  = obj2.compute_visual_impact(x, y, wr)
assert abs(vi2 - vi) < 1e-5, f"split-weight mismatch: {vi2} vs {vi}"

print("All assertions passed.")
