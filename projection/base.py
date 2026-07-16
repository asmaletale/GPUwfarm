"""
Abstract base class for all projection operators.

Each operator maps (P, T, 2) positions to (P, T, 2) feasible positions.
Operators are composable via CompositeProjection.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
import cupy as cp


class ProjectionOperator(ABC):
    """Maps a population of layouts to the feasible set."""

    @abstractmethod
    def project(self, pop: cp.ndarray) -> cp.ndarray:
        """
        Args:
            pop: (P, T, 2) positions [x, y] on GPU
        Returns:
            (P, T, 2) corrected positions
        """
        ...


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
