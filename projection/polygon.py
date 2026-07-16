"""
Polygon exclusion zone projection.

Moves turbines that fall inside exclusion polygons to the nearest
boundary point of that polygon.

This projection supports arbitrary convex or non-convex exclusion zones
(e.g. wetlands, shipping lanes, setback areas).

GPU note: For the initial implementation, polygon containment testing
is done on CPU (numpy) and results broadcast to the population.
A full GPU implementation using barycentric/winding-number tests
is deferred to a future version.
"""
from __future__ import annotations
import numpy as np
import cupy as cp
from typing import List
from projection.base import ProjectionOperator


def _point_in_polygon(px: float, py: float, poly: np.ndarray) -> bool:
    """Ray-casting test for point-in-polygon."""
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def _nearest_boundary_point(px: float, py: float, poly: np.ndarray):
    """Find the nearest point on polygon boundary edges."""
    min_dist = np.inf
    bx, by = px, py
    n = len(poly)
    for i in range(n):
        ax, ay = poly[i]
        bx2, by2 = poly[(i + 1) % n]
        dx, dy = bx2 - ax, by2 - ay
        t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx**2 + dy**2 + 1e-12)))
        cx, cy = ax + t * dx, ay + t * dy
        d = np.hypot(px - cx, py - cy)
        if d < min_dist:
            min_dist = d
            bx, by = cx, cy
    return bx, by


class PolygonProjection(ProjectionOperator):
    """
    Exclusion zone enforcement via polygon containment.

    Args:
        polygons: list of (N, 2) numpy arrays defining exclusion polygon vertices
    """

    def __init__(self, polygons: List[np.ndarray]) -> None:
        self.polygons = polygons

    def project(self, pop: cp.ndarray) -> cp.ndarray:
        """
        Args:
            pop: (P, T, 2) — x, y positions
        Returns:
            (P, T, 2) — positions moved outside all exclusion polygons
        """
        # Transfer to CPU for polygon containment check (see GPU note above)
        xy_np = cp.asnumpy(pop)   # (P, T, 2)
        P, T, _ = xy_np.shape

        for poly in self.polygons:
            for p in range(P):
                for t in range(T):
                    px, py = float(xy_np[p, t, 0]), float(xy_np[p, t, 1])
                    if _point_in_polygon(px, py, poly):
                        nx, ny = _nearest_boundary_point(px, py, poly)
                        xy_np[p, t, 0] = nx
                        xy_np[p, t, 1] = ny

        return cp.asarray(xy_np)
