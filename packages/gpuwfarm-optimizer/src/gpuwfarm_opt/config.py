"""
Optimizer-only configuration dataclasses for the GPUwfarm genetic algorithm.

The physics / evaluation configs (WakeConfig, FarmConfig, TurbineConfig,
CostConfig, VisualImpactConfig) now live in the evaluation core package,
``gpuwfarm_core.config``.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class GAConfig:
    pop_size:       int   = 256
    n_generations:  int   = 150
    mutation_rate:  float = 0.15
    crossover_rate:      float = 0.7   # probability a parent pair undergoes crossover
    gene_swap_rate:      float = 0.0   # per-turbine swap probability (0 = use 1/T)
    sigma_xy:            float = 50.0  # position mutation step, metres (farm-scale dependent)
    sigma_yaw_deg:       float = 3.0   # yaw mutation step, degrees
    elite:          int   = 6     # unused: survival is an elitist (mu + lambda) merge
    max_yaw_deg:    float = 30.0  # degrees
    optimize:       str   = "both"  # "both" | "layout" (yaw fixed at 0) | "yaw" (layout fixed)

    def __post_init__(self) -> None:
        if self.optimize not in ("both", "layout", "yaw"):
            raise ValueError(
                f"optimize must be 'both', 'layout' or 'yaw', got {self.optimize!r}"
            )
