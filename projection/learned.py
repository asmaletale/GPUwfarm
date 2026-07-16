"""
Learned projection interface — stub for future ML-based projectors.

Provides the interface for a machine-learning feasibility projector
(e.g. a neural network trained to map infeasible layouts to the
nearest feasible point on the constraint manifold).

Architecture:
    LearnedProjectionInterface
        .project(pop) → pop  (delegates to self.model.forward(pop))

Plug in any model that implements forward(x: cp.ndarray) -> cp.ndarray.
"""
from __future__ import annotations
import cupy as cp
from projection.base import ProjectionOperator
from typing import Protocol, runtime_checkable


@runtime_checkable
class MLProjectorModel(Protocol):
    """Protocol: any object with a forward(x) → x method."""
    def forward(self, x: cp.ndarray) -> cp.ndarray: ...


class LearnedProjectionInterface(ProjectionOperator):
    """
    Wraps a trained ML model as a ProjectionOperator.

    The model must accept (P, T*2) flattened positions and return
    (P, T*2) corrected positions.
    """

    def __init__(self, model: MLProjectorModel, n_turbines: int) -> None:
        assert isinstance(model, MLProjectorModel), \
            "model must implement forward(x: cp.ndarray) -> cp.ndarray"
        self.model      = model
        self.n_turbines = n_turbines

    def project(self, pop: cp.ndarray) -> cp.ndarray:
        """
        Args:
            pop: (P, T, 2)
        Returns:
            (P, T, 2)
        """
        P, T, _ = pop.shape
        flat = pop.reshape(P, T * 2)
        out  = self.model.forward(flat)
        return out.reshape(P, T, 2)
