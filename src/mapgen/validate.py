"""Playability validation for generated segments.

Uses BFS to check whether a player can reach the exit from the entry.
Works on numpy grids directly — no .map file needed (generated segments
exist only in memory).

Simpler than pathfind.trace_path() because generated segments don't
use teleporters.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from mapgen.pathfind import _is_passable
from mapgen.schema import Opening


@dataclass
class ValidationResult:
    """Outcome of a playability check."""
    playable: bool           # can player reach exit from entry?
    reachable_pct: float     # % of passable tiles reachable from entry
    path_length: int         # BFS steps from entry to nearest exit tile (-1 if unreachable)
    total_passable: int      # total passable tiles in the grid
    total_reachable: int     # passable tiles reachable from entry


# ── 8-directional BFS ───────────────────────────────────────────────

# Same 8 directions as pathfind.py — player can hook/jump diagonally
_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1),
         (-1, -1), (-1, 1), (1, -1), (1, 1)]


def validate_segment(
    grid: np.ndarray,
    entry: Opening,
    exit_: Opening,
) -> ValidationResult:
    """Check if a player can reach the exit from the entry via BFS.

    Args:
        grid: 2D tile grid (height, width) from build_grid().
        entry: The entry opening specification.
        exit_: The exit opening specification.

    Returns:
        ValidationResult with playability info.
    """
    h, w = grid.shape

    # Collect entry and exit tile positions
    entry_tiles = _opening_tiles(entry, h, w)
    exit_tiles = set(_opening_tiles(exit_, h, w))

    # Count total passable tiles
    total_passable = int(np.sum(np.vectorize(_is_passable)(grid)))

    if not entry_tiles:
        return ValidationResult(
            playable=False, reachable_pct=0.0, path_length=-1,
            total_passable=total_passable, total_reachable=0,
        )

    # BFS from all entry tiles
    visited: dict[tuple[int, int], int] = {}  # (x, y) -> step
    queue: deque[tuple[int, int, int]] = deque()  # (x, y, step)

    for (x, y) in entry_tiles:
        if 0 <= y < h and 0 <= x < w and _is_passable(int(grid[y, x])):
            visited[(x, y)] = 0
            queue.append((x, y, 0))

    exit_step = -1

    while queue:
        x, y, step = queue.popleft()

        # Check if we reached an exit tile
        if (x, y) in exit_tiles and exit_step == -1:
            exit_step = step

        # Expand neighbors
        for dx, dy in _DIRS:
            nx, ny = x + dx, y + dy
            if 0 <= ny < h and 0 <= nx < w and (nx, ny) not in visited:
                if _is_passable(int(grid[ny, nx])):
                    visited[(nx, ny)] = step + 1
                    queue.append((nx, ny, step + 1))

    total_reachable = len(visited)
    reachable_pct = (total_reachable / total_passable * 100) if total_passable > 0 else 0.0

    return ValidationResult(
        playable=exit_step >= 0,
        reachable_pct=round(reachable_pct, 1),
        path_length=exit_step,
        total_passable=total_passable,
        total_reachable=total_reachable,
    )


def _opening_tiles(opening: Opening, h: int, w: int) -> list[tuple[int, int]]:
    """Get the (x, y) tile positions for an opening on the border."""
    tiles = []

    if opening.side == "top":
        for x in range(opening.x, opening.x + opening.width):
            if 0 <= x < w:
                tiles.append((x, 0))

    elif opening.side == "bottom":
        for x in range(opening.x, opening.x + opening.width):
            if 0 <= x < w:
                tiles.append((x, h - 1))

    elif opening.side == "left":
        for y in range(opening.y, opening.y + opening.width):
            if 0 <= y < h:
                tiles.append((0, y))

    elif opening.side == "right":
        for y in range(opening.y, opening.y + opening.width):
            if 0 <= y < h:
                tiles.append((w - 1, y))

    return tiles
