"""
Optimizer-only configuration dataclasses.

The physics / evaluation configs (WakeConfig, FarmConfig, TurbineConfig,
CostConfig, VisualImpactConfig) live in the evaluation core package,
``gpuwfarm_core.config``.

``OptimizerConfig`` holds what every search algorithm needs — how many
candidates, how many iterations, and which decision variables are free. Each
algorithm's own config inherits from it and adds only its own operators'
parameters. Every field has a default, so subclass field ordering never matters
at a call site (all construction is keyword-based).
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class OptimizerConfig:
    """Fields shared by every optimizer in this package."""

    pop_size:       int   = 256
    n_generations:  int   = 150
    max_yaw_deg:    float = 30.0  # degrees
    optimize:       str   = "both"  # "both" | "layout" (yaw fixed at 0) | "yaw" (layout fixed)

    def __post_init__(self) -> None:
        if self.optimize not in ("both", "layout", "yaw"):
            raise ValueError(
                f"optimize must be 'both', 'layout' or 'yaw', got {self.optimize!r}"
            )

    @property
    def n_iterations(self) -> int:
        """
        Alias for n_generations.

        "Generation" is the GA's word for it, but the number is really just the
        HDF5 row count — the loop bound is the same quantity for a swarm or a
        gradient batch, so PSO/GD code reads better through this name.
        """
        return self.n_generations


@dataclass
class GAConfig(OptimizerConfig):
    mutation_rate:  float = 0.15
    crossover_rate: float = 0.7   # probability a parent pair undergoes crossover
    gene_swap_rate: float = 0.0   # per-turbine swap probability (0 = use 1/T)
    sigma_xy:       float = 50.0  # position mutation step, metres (farm-scale dependent)
    sigma_yaw_deg:  float = 3.0   # yaw mutation step, degrees
    elite:          int   = 6     # unused: survival is an elitist (mu + lambda) merge


@dataclass
class PSOConfig(OptimizerConfig):
    """
    Multi-objective particle swarm (MOPSO).

    Velocity is the usual inertia + cognitive + social blend; the social pull is
    toward a leader drawn from the external Pareto archive rather than a single
    global best, which is what makes the swarm spread along a front instead of
    collapsing onto one point.
    """

    inertia:       float = 0.4
    c1:            float = 1.5   # cognitive weight (pull toward personal best)
    c2:            float = 1.5   # social weight (pull toward archive leader)
    v_max_xy:      float = 100.0  # max position step, metres per iteration
    v_max_yaw_deg: float = 5.0    # max yaw step, degrees per iteration
    archive_size:  int   = 0      # external Pareto archive cap; 0 means "use pop_size"


@dataclass
class GDConfig(OptimizerConfig):
    """
    SPSA gradient descent with Chebyshev decomposition.

    Row p of the batch descends its own scalarised objective, so P rows sweep
    out P points of the Pareto front in a single run.

    Gradients come from SPSA (simultaneous perturbation), which needs only
    2*spsa_pairs evaluations per iteration regardless of turbine count -- and,
    being derivative-free, is indifferent to the sort/max/lookup-table
    discontinuities in the visual-impact objective and the power curve.

    Note the metres/radians split: x/y live on a ~2000 m farm while yaw lives in
    +/-0.5 rad, so a single learning rate or perturbation radius cannot serve
    both columns.
    """

    lr_xy:          float = 20.0  # Adam step, metres
    lr_yaw_deg:     float = 1.0   # Adam step, degrees
    spsa_c_xy:      float = 5.0   # SPSA perturbation radius, metres
    spsa_c_yaw_deg: float = 0.5   # SPSA perturbation radius, degrees
    spsa_pairs:     int   = 1     # +/- pairs averaged per iteration (2*pairs evaluations)
    rho:            float = 1e-3  # augmented-Chebyshev term; kills weakly-dominated points
    torch_optimizer: str  = "adam"  # any torch.optim class name: adam, sgd, rmsprop, adamw
