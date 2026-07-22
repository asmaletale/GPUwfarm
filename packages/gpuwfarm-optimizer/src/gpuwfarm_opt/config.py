"""
Optimizer-only configuration dataclasses for the GPUwfarm genetic algorithm.

The physics / evaluation configs (WakeConfig, FarmConfig, TurbineConfig,
CostConfig, VisualImpactConfig) now live in the evaluation core package,
``gpuwfarm_core.config``.
"""
from __future__ import annotations
from dataclasses import dataclass


# Decision variables the GA is allowed to search over.
#   "both"   → optimise turbine positions (x, y) *and* yaw angles (default)
#   "layout" → optimise positions only; yaw is held fixed at 0
#   "yaw"    → optimise yaw only; positions are held fixed (needs a seed layout)
OPTIMIZE_MODES = ("both", "layout", "yaw")


@dataclass
class GAConfig:
    pop_size:       int   = 256
    n_generations:  int   = 150
    mutation_rate:  float = 0.15
    crossover_rate:      float = 0.7   # probability a parent pair undergoes crossover
    gene_swap_rate:      float = 0.0   # per-turbine swap probability (0 = use 1/T)
    elite:          int   = 6
    max_yaw_deg:    float = 30.0  # degrees
    optimize:       str   = "both"  # one of OPTIMIZE_MODES: "both" | "layout" | "yaw"

    def __post_init__(self) -> None:
        if self.optimize not in OPTIMIZE_MODES:
            raise ValueError(
                f"GAConfig.optimize must be one of {OPTIMIZE_MODES}, "
                f"got {self.optimize!r}"
            )
