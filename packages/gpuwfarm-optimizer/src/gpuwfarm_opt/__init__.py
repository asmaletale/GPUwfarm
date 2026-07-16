"""
gpuwfarm_opt — genetic-algorithm optimizer for wind-farm layout and yaw.

This package is the optimization layer. It depends on the evaluation core
(``gpuwfarm_core``) for all physics: it injects a ``FarmEvaluator`` into the
genetic algorithm and never contains physics itself. It adds the GA search
operators, the feasibility-repair projection chain, and the CLI entry point.
"""
from __future__ import annotations

from gpuwfarm_opt.config import GAConfig
from gpuwfarm_opt.genetic import GeneticAlgorithm

__all__ = ["GAConfig", "GeneticAlgorithm"]
