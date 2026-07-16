"""
Pairwise spacing projection — GPU port of the original main.py project().

Iterative repulsion to enforce minimum turbine-to-turbine distance.
Each pass moves overlapping turbines apart symmetrically.

This refactors the existing main.py logic (lines 65-92) into the
ProjectionOperator interface with configurable passes.
"""
from __future__ import annotations
import cupy as cp
from gpuwfarm_opt.projection.base import ProjectionOperator
from gpuwfarm_core.config import FarmConfig


class PairwiseSpacingProjection(ProjectionOperator):
    """
    Repulsion-based minimum spacing constraint.

    For each pair of turbines that are too close, both turbines are pushed
    apart by half the overlap distance, in the direction of their separation.
    """

    def __init__(self, farm_cfg: FarmConfig, n_passes: int = 10) -> None:
        self.min_spacing = farm_cfg.min_spacing
        self.n_passes    = n_passes

    def project(self, pop: cp.ndarray) -> cp.ndarray:
        """
        Args:
            pop: (P, T, 2) — x, y positions
        Returns:
            (P, T, 2) — positions with minimum spacing enforced
        """
        x = pop[:, :, 0].copy()
        y = pop[:, :, 1].copy()
        D = cp.float32(self.min_spacing)

        for _ in range(self.n_passes):
            dx = x[:, :, None] - x[:, None, :]   # (P, T, T)
            dy = y[:, :, None] - y[:, None, :]

            dist = cp.sqrt(dx ** 2 + dy ** 2 + cp.float32(1e-9))
            mask = (dist < D) & (dist > cp.float32(0.0))

            overlap = cp.where(mask, D - dist, cp.float32(0.0))
            ux = dx / dist
            uy = dy / dist

            x += cp.sum(cp.float32(0.5) * overlap * ux, axis=2)
            y += cp.sum(cp.float32(0.5) * overlap * uy, axis=2)

        pop = pop.copy()
        pop[:, :, 0] = x
        pop[:, :, 1] = y
        return pop
