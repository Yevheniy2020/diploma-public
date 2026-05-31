"""Geometry helpers for the region-based semantic layer.

Spaces are stored as polygons in world (meter) coordinates. The planner
operates on a grid of cells. These helpers bridge the two: cell↔meter
conversion, point-in-polygon, and "find the nearest free grid cell that
lies inside a given polygon", which is what NAVIGATE-to-space needs.
"""
from __future__ import annotations

from math import hypot
from typing import Optional, Sequence

import numpy as np


Vertex = tuple[float, float]
Polygon = Sequence[Vertex]


def meters_to_cell(point_m: Vertex, cell_size: float, grid_h: int) -> tuple[int, int]:
    """World (x east, y north, origin lower-left) → (col, row) with row
    flipped because grid index 0 is the top of the map."""
    col = int(point_m[0] / cell_size)
    row = grid_h - 1 - int(point_m[1] / cell_size)
    return (col, row)


def cell_to_meters(col: int, row: int, cell_size: float, grid_h: int) -> Vertex:
    x = (col + 0.5) * cell_size
    y = ((grid_h - 1 - row) + 0.5) * cell_size
    return (x, y)


def point_in_polygon(point: Vertex, polygon: Polygon) -> bool:
    """Standard ray-cast even-odd test. Vertices on the boundary are
    treated as inside on one side / outside on the other — fine for our
    use because cells near the edge are picked up via the AABB scan."""
    x, y = point
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        # Intersect a ray from (x, y) going to +x with edge (i, j).
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def polygon_centroid(polygon: Polygon) -> Vertex:
    """Average of vertices — not the area-weighted centroid, but good
    enough for placing a name label on a roughly convex space."""
    if not polygon:
        return (0.0, 0.0)
    sx = sum(v[0] for v in polygon)
    sy = sum(v[1] for v in polygon)
    n = len(polygon)
    return (sx / n, sy / n)


def nearest_free_cell_in_polygon(
    inflated_grid: np.ndarray,
    polygon_m: Polygon,
    robot_m: Vertex,
    cell_size: float,
) -> Optional[tuple[int, int]]:
    """Return the (col, row) of the free grid cell whose centre lies
    inside `polygon_m` and is closest (Euclidean) to the robot.

    Returns None when the polygon has no free interior cell — caller
    should treat as `no_path`."""
    h, w = inflated_grid.shape
    if not polygon_m:
        return None

    # AABB of the polygon in cell space, clamped to grid bounds.
    cells = [meters_to_cell(v, cell_size, h) for v in polygon_m]
    min_col = max(0, min(c for c, _ in cells))
    max_col = min(w - 1, max(c for c, _ in cells))
    min_row = max(0, min(r for _, r in cells))
    max_row = min(h - 1, max(r for _, r in cells))

    robot_col, robot_row = meters_to_cell(robot_m, cell_size, h)

    best: Optional[tuple[int, int]] = None
    best_d = float("inf")
    for r in range(min_row, max_row + 1):
        for c in range(min_col, max_col + 1):
            if inflated_grid[r, c] != 0:
                continue
            centre_m = cell_to_meters(c, r, cell_size, h)
            if not point_in_polygon(centre_m, polygon_m):
                continue
            d = hypot(c - robot_col, r - robot_row)
            if d < best_d:
                best_d = d
                best = (c, r)
    return best
