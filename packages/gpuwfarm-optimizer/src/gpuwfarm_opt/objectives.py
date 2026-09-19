"""
Pluggable objective functions for the optimizers.

An objective is anything that scores a whole population in one batched call:

    (pop, aep, wind_rose) -> (P,) CuPy array

``ObjectiveSet`` stacks several of them into the ``(P, M)`` minimisation matrix
that Pareto ranking and selection consume. Adding a new objective is appending
one object to that list -- nothing in the optimizers, the non-dominated sort,
the crowding distance, the HDF5 logger or the analysis scripts needs to know how
many objectives there are or what they mean.

    class Noise:
        name      = "noise"
        direction = "min"
        def __call__(self, pop, aep, wind_rose):
            return my_batched_noise_model(pop)      # (P,) cupy

    objectives = ObjectiveSet([LCOE(obj_eval), VisualImpact(obj_eval), Noise()])
    GeneticAlgorithm(..., objectives=objectives)

``direction`` is the only bookkeeping: "max" columns are negated on the way out,
so every consumer downstream can assume "smaller is better" without a flag.
"""
from __future__ import annotations
from typing import Protocol, Sequence, runtime_checkable

import numpy as np
import cupy as cp

from gpuwfarm_core.objectives import ObjectiveEvaluator
from gpuwfarm_core.wind.wind_rose import WindRose


@runtime_checkable
class Objective(Protocol):
    """One scalar score per individual, computed for the whole batch at once."""

    name: str
    direction: str  # "min" | "max"

    def __call__(
        self, pop: cp.ndarray, aep: cp.ndarray, wind_rose: WindRose
    ) -> cp.ndarray:
        """(P, T, 3) population + (P,) AEP in kWh -> (P,) CuPy score."""
        ...


def cable_length_km(pop: cp.ndarray) -> cp.ndarray:
    """
    Cable-length proxy: total turbine distance from the farm's own centroid, km.

    Cheap stand-in for a real collection-network layout -- it captures "spread
    out costs more" without solving a minimum spanning tree per individual per
    generation. Stays on the GPU.

    Args:
        pop: (P, T, 3) population [x, y, yaw]

    Returns:
        (P,) CuPy, kilometres
    """
    x = pop[:, :, 0]
    y = pop[:, :, 1]
    cx = x.mean(axis=1, keepdims=True)
    cy = y.mean(axis=1, keepdims=True)
    return cp.sqrt((x - cx) ** 2 + (y - cy) ** 2).sum(axis=1) / cp.float32(1000.0)


class LCOE:
    """Levelised cost of energy, EUR/MWh. Needs AEP and the cable-length proxy."""

    name = "lcoe"
    direction = "min"

    def __init__(self, obj_eval: ObjectiveEvaluator) -> None:
        self.obj_eval = obj_eval

    def __call__(self, pop, aep, wind_rose) -> cp.ndarray:
        n_turbines = pop.shape[1]
        aep_gwh = aep / cp.float32(1e6)  # kWh -> GWh
        return self.obj_eval.compute_lcoe_batch(
            n_turbines, aep_gwh, cable_length_km(pop)
        )


class VisualImpact:
    """
    Visual impact: observer-weighted union of the turbines' angular footprints.

    Deliberately *not* differentiable (a sweep-line union area built on a sort,
    a hard coverage test and a max reduction), which is one reason the gradient
    optimizer estimates gradients by evaluation rather than autodiff.
    """

    name = "vi"
    direction = "min"

    def __init__(self, obj_eval: ObjectiveEvaluator) -> None:
        self.obj_eval = obj_eval

    def __call__(self, pop, aep, wind_rose) -> cp.ndarray:
        return self.obj_eval.compute_vi_batch(pop[:, :, 0], pop[:, :, 1], wind_rose)


class NegativeAEP:
    """
    Negated annual energy production, GWh -- minimising this maximises AEP.

    Stored negative rather than declared ``direction="max"`` so the sign of the
    column written to HDF5 matches every history file produced before objectives
    became pluggable.
    """

    name = "neg_aep_gwh"
    direction = "min"

    def __call__(self, pop, aep, wind_rose) -> cp.ndarray:
        return (-aep / cp.float32(1e6)).astype(cp.float32)


class ObjectiveSet:
    """
    An ordered list of objectives, evaluated together into one (P, M) matrix.

    Everything stays GPU-resident until a single (P, M) device-to-host copy at
    the end -- the Pareto ranking downstream runs on NumPy, and M floats per
    individual is a far smaller transfer than the (P, T, 3) population.
    """

    def __init__(self, objectives: Sequence[Objective]) -> None:
        if not objectives:
            raise ValueError("ObjectiveSet needs at least one objective")
        for o in objectives:
            if o.direction not in ("min", "max"):
                raise ValueError(
                    f"objective {o.name!r} has direction {o.direction!r}, "
                    "expected 'min' or 'max'"
                )
        self.objectives = list(objectives)

    def __len__(self) -> int:
        return len(self.objectives)

    @property
    def names(self) -> list[str]:
        return [o.name for o in self.objectives]

    @property
    def directions(self) -> list[str]:
        return [o.direction for o in self.objectives]

    def __call__(
        self, pop: cp.ndarray, aep: cp.ndarray, wind_rose: WindRose
    ) -> np.ndarray:
        """
        Args:
            pop:       (P, T, 3) population
            aep:       (P,) AEP in kWh
            wind_rose: conditions, passed through to each objective

        Returns:
            (P, M) float32 NumPy, minimisation convention ("max" columns negated)
        """
        cols = []
        for o in self.objectives:
            col = o(pop, aep, wind_rose).astype(cp.float32)
            cols.append(-col if o.direction == "max" else col)
        return cp.asnumpy(cp.stack(cols, axis=1)).astype(np.float32)


def default_objective_set(mode: str, obj_eval: ObjectiveEvaluator) -> ObjectiveSet:
    """
    Build the built-in objective pairs by name.

    "lcoe_vi" -> [LCOE, VisualImpact]      (column 0 EUR/MWh, column 1 VI)
    "aep_vi"  -> [NegativeAEP, VisualImpact] (column 0 -GWh,   column 1 VI)

    Both reproduce, column for column and sign for sign, what the GA produced
    before objectives were pluggable.
    """
    if mode == "lcoe_vi":
        return ObjectiveSet([LCOE(obj_eval), VisualImpact(obj_eval)])
    if mode == "aep_vi":
        return ObjectiveSet([NegativeAEP(), VisualImpact(obj_eval)])
    raise ValueError(
        f"objectives_mode must be 'lcoe_vi' or 'aep_vi', got {mode!r}"
    )
