import logging
from collections import deque
from typing import Optional

import numpy as np
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

from services.inflation import inflate

_logger = logging.getLogger(__name__)


def _meters_to_cell(point_m: tuple[float, float], cell_size: float, height: int) -> tuple[int, int]:
    col = int(point_m[0] / cell_size)
    row = height - 1 - int(point_m[1] / cell_size)
    return (col, row)


def _cell_to_meters(col: int, row: int, cell_size: float, height: int) -> tuple[float, float]:
    x = (col + 0.5) * cell_size
    y = ((height - 1 - row) + 0.5) * cell_size
    return (x, y)


def _line_of_sight(grid: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> bool:
    x0, y0 = a
    x1, y1 = b
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    h, w = grid.shape
    x, y = x0, y0
    while True:
        if not (0 <= y < h and 0 <= x < w) or grid[y, x] == 1:
            return False
        if x == x1 and y == y1:
            return True
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy


def _smooth(cells: list[tuple[int, int]], grid: np.ndarray) -> list[tuple[int, int]]:
    if len(cells) <= 2:
        return cells
    smoothed = [cells[0]]
    anchor = 0
    for i in range(1, len(cells) - 1):
        if not _line_of_sight(grid, cells[anchor], cells[i + 1]):
            smoothed.append(cells[i])
            anchor = i
    smoothed.append(cells[-1])
    return smoothed


def _nearest_free(grid: np.ndarray, col: int, row: int) -> Optional[tuple[int, int]]:
    h, w = grid.shape
    if 0 <= row < h and 0 <= col < w and grid[row, col] == 0:
        return (col, row)
    visited: set[tuple[int, int]] = {(col, row)}
    queue: deque[tuple[int, int]] = deque([(col, row)])
    while queue:
        c, r = queue.popleft()
        for dc, dr in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            nc, nr = c + dc, r + dr
            if (nc, nr) in visited or not (0 <= nr < h and 0 <= nc < w):
                continue
            visited.add((nc, nr))
            if grid[nr, nc] == 0:
                return (nc, nr)
            queue.append((nc, nr))
    return None


def plan(
    grid: np.ndarray,
    start_m: tuple[float, float],
    goal_m: tuple[float, float],
    cell_size: float,
) -> list[tuple[float, float]]:
    inflated = inflate(grid)
    h, w = inflated.shape

    start_cell = _meters_to_cell(start_m, cell_size, h)
    goal_cell = _meters_to_cell(goal_m, cell_size, h)

    for cell in (start_cell, goal_cell):
        col, row = cell
        if not (0 <= row < h and 0 <= col < w):
            _logger.warning("plan: cell out of bounds %s in grid %dx%d", cell, w, h)
            return []

    if inflated[start_cell[1], start_cell[0]] == 1:
        snap = _nearest_free(inflated, *start_cell)
        if snap is None:
            return []
        start_cell = snap
    if inflated[goal_cell[1], goal_cell[0]] == 1:
        snap = _nearest_free(inflated, *goal_cell)
        if snap is None:
            return []
        goal_cell = snap

    matrix = (1 - inflated).astype(int).tolist()
    pf_grid = Grid(matrix=matrix)
    start_node = pf_grid.node(start_cell[0], start_cell[1])
    end_node = pf_grid.node(goal_cell[0], goal_cell[1])
    finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
    path, _runs = finder.find_path(start_node, end_node, pf_grid)

    if not path:
        _logger.warning("plan: no path from %s to %s", start_cell, goal_cell)
        return []

    cells = [(node.x, node.y) for node in path]
    cells = _smooth(cells, inflated)
    return [_cell_to_meters(c, r, cell_size, h) for c, r in cells]
