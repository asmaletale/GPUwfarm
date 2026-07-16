"""
gpuwfarm_core — GPU-batched, FLORIS-traceable wind-farm evaluation core.

This package is the fast evaluation/simulation layer, independent of any
optimizer. It exposes the batched AEP evaluator, the wind rose, the turbine
power model, the wake-model abstract base classes, and the LCOE / visual-impact
objectives — everything needed to score a population of farm layouts without
pulling in the genetic algorithm.

Typical standalone use (e.g. inside a reinforcement-learning loop)::

    from gpuwfarm_core import (
        FarmEvaluator, WindRose, WakeConfig, FarmConfig, TurbineConfig, TurbineData,
    )

    evaluator = FarmEvaluator(farm_cfg, turbine_cfg, wake_cfg, turbine_data)
    aep = evaluator.evaluate(pop, wind_rose)   # (P,) cupy array

The FLORIS-YAML loader is intentionally NOT eagerly imported here so that a bare
``import gpuwfarm_core`` does not require PyYAML; reach it explicitly via
``gpuwfarm_core.loaders.floris_yaml.load_floris_yaml``.
"""
from __future__ import annotations

from gpuwfarm_core.config import (
    WakeConfig,
    FarmConfig,
    TurbineConfig,
    CostConfig,
    VisualImpactConfig,
)
from gpuwfarm_core.wind.wind_rose import WindRose
from gpuwfarm_core.physics.farm_evaluator import FarmEvaluator
from gpuwfarm_core.physics.turbine.power_curve import TabulatedPowerCurve, TurbineData
from gpuwfarm_core.objectives import ObjectiveEvaluator
from gpuwfarm_core.physics.base import (
    BaseWakeVelocity,
    BaseWakeTurbulence,
    BaseWakeDeflection,
    BaseWakeCombination,
)

__all__ = [
    # evaluation
    "FarmEvaluator",
    "WindRose",
    "TabulatedPowerCurve",
    "TurbineData",
    "ObjectiveEvaluator",
    # config
    "WakeConfig",
    "FarmConfig",
    "TurbineConfig",
    "CostConfig",
    "VisualImpactConfig",
    # extension points
    "BaseWakeVelocity",
    "BaseWakeTurbulence",
    "BaseWakeDeflection",
    "BaseWakeCombination",
]
