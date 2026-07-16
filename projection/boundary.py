"""
Boundary projection — clips turbine positions to a rectangular domain.

Extends the original main.py boundary clipping to the ProjectionOperator
interface, supporting arbitrary rectangular boundaries.
"""
from __future__ import annotations
import cupy as cp
from projection.base import ProjectionOperator
from config import FarmConfig


class BoundaryProjection(ProjectionOperator):
    """
    Clips all turbine positions to the rectangular farm boundary.

    Positions are clipped to [x_min, x_max] × [y_min, y_max].
    Default boundary is [0, area_width] × [0, area_height] from FarmConfig.
    """

    def __init__(
        self,
        farm_cfg: FarmConfig,
        x_min: float = 0.0,
        y_min: float = 0.0,
        x_max: float | None = None,
        y_max: float | None = None,
        setback: float = 0.0,
    ) -> None:
        self.x_min = cp.float32(x_min + setback)
        self.y_min = cp.float32(y_min + setback)
        self.x_max = cp.float32((x_max if x_max is not None else farm_cfg.area_width) - setback)
        self.y_max = cp.float32((y_max if y_max is not None else farm_cfg.area_height) - setback)

    def project(self, pop: cp.ndarray) -> cp.ndarray:
        """
        Args:
            pop: (P, T, 2) — x, y positions
        Returns:
            (P, T, 2) — positions clipped to boundary
        """
        pop = pop.copy()
        pop[:, :, 0] = cp.clip(pop[:, :, 0], self.x_min, self.x_max)
        pop[:, :, 1] = cp.clip(pop[:, :, 1], self.y_min, self.y_max)
        return pop
