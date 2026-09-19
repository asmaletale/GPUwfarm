"""
gpuwfarm_opt — optimizers for wind-farm layout and yaw.

This package is the optimization layer. It depends on the evaluation core
(``gpuwfarm_core``) for all physics: it injects a ``FarmEvaluator`` into an
optimizer and never contains physics itself. It adds the search operators, the
feasibility-repair projection chain, and the CLI entry point.

Three optimizers, all subclasses of ``Optimizer`` and all sharing its stage
methods, its HDF5 history format and its checkpoint/resume behaviour:

    GeneticAlgorithm   NSGA-II genetic search
    ParticleSwarm      MOPSO with an external, crowding-pruned Pareto archive
    GradientDescent    SPSA gradients + torch.optim, Chebyshev decomposition

They are interchangeable: each takes the same constructor arguments and each
``run()`` returns the same 4-tuple. Which objectives are optimised is
configuration -- see ``gpuwfarm_opt.objectives``.
"""
from __future__ import annotations

import os
import sys

# PyTorch ships the CUDA runtime DLLs (curand, cublas, ...) that CuPy needs on
# Windows. Register that directory before anything imports CuPy, so
# cuda-pathfinder can find them without a full CUDA toolkit installed.
if sys.platform == "win32":  # pragma: no cover - platform specific
    _torch_lib = os.path.normpath(
        os.path.join(os.path.dirname(sys.executable), "..", "Lib",
                     "site-packages", "torch", "lib")
    )
    if os.path.isdir(_torch_lib):
        os.add_dll_directory(_torch_lib)

from gpuwfarm_opt.base import Optimizer
from gpuwfarm_opt.config import OptimizerConfig, GAConfig, PSOConfig, GDConfig
from gpuwfarm_opt.genetic import GeneticAlgorithm
from gpuwfarm_opt.gradient import GradientDescent
from gpuwfarm_opt.objectives import (
    LCOE, NegativeAEP, Objective, ObjectiveSet, VisualImpact,
    default_objective_set,
)
from gpuwfarm_opt.swarm import ParticleSwarm

__all__ = [
    # optimizers
    "Optimizer", "GeneticAlgorithm", "ParticleSwarm", "GradientDescent",
    # configs
    "OptimizerConfig", "GAConfig", "PSOConfig", "GDConfig",
    # objectives
    "Objective", "ObjectiveSet", "LCOE", "VisualImpact", "NegativeAEP",
    "default_objective_set",
]
