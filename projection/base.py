"""
Abstract base class for all projection operators.

Each operator maps (P, T, 2) positions to (P, T, 2) feasible positions.
Operators are composable via CompositeProjection.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
import cupy as cp

from physics.base import ProjectionOperator


class CompositeProjection(ProjectionOperator):
    """
    Chains multiple ProjectionOperator instances in order.

    Example:
        proj = CompositeProjection([
            PairwiseSpacingProjection(cfg),
            BoundaryProjection(cfg),
        ])
        pop_xy = proj.project(pop_xy)
    """

    def __init__(self, operators: List[ProjectionOperator]) -> None:
        self.operators = operators

    def project(self, pop: cp.ndarray) -> cp.ndarray:
        for op in self.operators:
            pop = op.project(pop)
        return pop
