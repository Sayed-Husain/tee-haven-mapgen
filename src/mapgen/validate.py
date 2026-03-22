"""Playability validation for generated segments.

Uses BFS to check whether a player can reach the exit from the entry.
Works on numpy grids directly — no .map file needed (generated segments
exist only in memory).

Simpler than pathfind.trace_path() because generated segments don't
use teleporters.

All BFS logic lives in bfs.py — this module is purely the "validation
question" layer: given a grid and openings, is it playable?
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from mapgen.bfs import (
    opening_tiles,
    bfs_flood,
    bfs_flood_with_steps,
    PASSABLE,
    DIRS_4,
)
from mapgen.schema import Opening


# Re-export opening_tiles for backward compatibility.
# Several modules import _opening_tiles from validate — this alias
# lets them keep working until they're updated to import from bfs.
_opening_tiles = opening_tiles


@dataclass
class ValidationResult:
    """Outcome of a playability check."""
    playable: bool           # can player reach exit from entry?
    reachable_pct: float     # % of passable tiles reachable from entry
    path_length: int         # BFS steps from entry to nearest exit tile (-1 if unreachable)
    total_passable: int      # total passable tiles in the grid
    total_reachable: int     # passable tiles reachable from entry
    island_count: int = 0          # disconnected air regions not reachable from entry
    connected_obstacle_pct: float = 100.0  # % of passable tiles on the main path


def validate_segment(
    grid: np.ndarray,
    entry: Opening,
    exit_: Opening,
) -> ValidationResult:
    """Check if a player can reach the exit from the entry via BFS.

    Also detects "islands" — disconnected passable regions that the
    player can never reach.  This feedback tells the LLM "your blueprint
    is playable but 40% of obstacles are unreachable islands — connect
    them."

    Args:
        grid: 2D tile grid (height, width) from build_grid().
        entry: The entry opening specification.
        exit_: The exit opening specification.

    Returns:
        ValidationResult with playability info + island metrics.
    """
    h, w = grid.shape

    # Collect entry and exit tile positions
    entry_tiles = opening_tiles(entry, h, w)
    exit_tiles = set(opening_tiles(exit_, h, w))

    # Count total passable tiles
    total_passable = int(np.sum(np.isin(grid, list(PASSABLE))))

    if not entry_tiles:
        return ValidationResult(
            playable=False, reachable_pct=0.0, path_length=-1,
            total_passable=total_passable, total_reachable=0,
        )

    # Full BFS with step tracking
    visited, exit_step = bfs_flood_with_steps(
        grid, entry_tiles, exit_tiles, PASSABLE,
    )

    total_reachable = len(visited)
    reachable_pct = (
        (total_reachable / total_passable * 100)
        if total_passable > 0 else 0.0
    )

    # ── Island detection ──
    # Find passable tiles NOT reached by the BFS.  Group them into
    # connected components ("islands").  Each island is a disconnected
    # region the player can never visit — wasted game space.
    reachable_set = set(visited.keys())
    island_count, connected_pct = _count_islands(
        grid, h, w, reachable_set, total_passable,
    )

    return ValidationResult(
        playable=exit_step >= 0,
        reachable_pct=round(reachable_pct, 1),
        path_length=exit_step,
        total_passable=total_passable,
        total_reachable=total_reachable,
        island_count=island_count,
        connected_obstacle_pct=round(connected_pct, 1),
    )


def _count_islands(
    grid: np.ndarray,
    h: int, w: int,
    reachable: set[tuple[int, int]],
    total_passable: int,
) -> tuple[int, float]:
    """Count disconnected passable regions not reachable from entry.

    Uses connected-component analysis: BFS from each unvisited passable
    tile to find its island, then count islands.

    Returns:
        (island_count, connected_obstacle_pct)
        - island_count: number of disconnected regions
        - connected_obstacle_pct: % of passable tiles on the main path
    """
    if total_passable == 0:
        return 0, 100.0

    # Find all passable tiles not in the reachable set
    unreachable_passable: set[tuple[int, int]] = set()
    for y in range(h):
        for x in range(w):
            if grid[y, x] in PASSABLE and (x, y) not in reachable:
                unreachable_passable.add((x, y))

    if not unreachable_passable:
        return 0, 100.0

    # Connected-component BFS on unreachable tiles
    visited: set[tuple[int, int]] = set()
    island_count = 0

    for start in unreachable_passable:
        if start in visited:
            continue

        # New island found — BFS to find all tiles in this component
        island_count += 1
        queue: deque[tuple[int, int]] = deque([start])
        visited.add(start)

        while queue:
            x, y = queue.popleft()
            for dx, dy in DIRS_4:
                nx, ny = x + dx, y + dy
                if (nx, ny) in unreachable_passable and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))

    connected_pct = (len(reachable) / total_passable * 100) if total_passable > 0 else 100.0
    return island_count, connected_pct
