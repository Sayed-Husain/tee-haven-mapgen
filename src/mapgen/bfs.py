"""Shared BFS utilities for map generation and validation.

This is the SINGLE SOURCE OF TRUTH for all navigation-related logic
in the codebase.  Every module that needs to answer "can the player
get from A to B?" imports from here instead of reimplementing BFS.

Why centralise?
  - The diagonal corner-passability check (see `_can_move_diagonal`)
    is critical for correctness.  When it was copy-pasted into 6
    files, 2 of them were missing the check — causing inconsistent
    results between the validator and the LLM feedback visualisation.
  - The passable-tile set must be consistent everywhere.  Having one
    definition prevents bugs where builder.py considers NOHOOK
    impassable while validate.py considers it passable.
  - Future features (dead-end detection, shortcut detection,
    post-processing) all need BFS — they should use the same
    primitives rather than inventing their own.

Functions exported:
  - opening_tiles()       — (x, y) positions for an Opening
  - bfs_flood()           — flood-fill, returns all reachable tiles
  - bfs_reachable()       — quick "can I reach any target?" check
  - bfs_path()            — shortest path with reconstruction
  - bridge_gaps()         — carve channels through solid walls
"""

from __future__ import annotations

from collections import deque

import numpy as np

from mapgen.extract import AIR, SOLID, ENTITY, FREEZE, NOHOOK
from mapgen.schema import Opening


# ── Shared constants ──────────────────────────────────────────────

# 8-directional movement — players can hook/jump diagonally in Gores
DIRS_8 = [(-1, 0), (1, 0), (0, -1), (0, 1),
          (-1, -1), (-1, 1), (1, -1), (1, 1)]

DIRS_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# All tile categories a player can occupy.
# - AIR: empty space
# - ENTITY: spawn markers, checkpoints — treated as air for movement
# - FREEZE: player gets frozen but is still IN this tile (part of
#   the game space in real Gores maps)
# - NOHOOK: solid surface that can't be hooked, but the tee can
#   walk/fall through the space around it
PASSABLE = frozenset({AIR, ENTITY, FREEZE, NOHOOK})

# Subset: tiles the player can occupy WITHOUT getting frozen/killed.
# Used by the freeze-path-clearance logic in builder.py to find
# routes that avoid freeze tiles.
PASSABLE_SAFE = frozenset({AIR, ENTITY, NOHOOK})


# ── Opening helper ────────────────────────────────────────────────

def opening_tiles(opening: Opening, h: int, w: int) -> list[tuple[int, int]]:
    """Get the (x, y) tile positions for an opening on the segment border.

    This was duplicated in validate.py (_opening_tiles) and builder.py
    (_opening_tiles_for_bfs) — now there's one copy.

    Args:
        opening: The entry/exit specification (side, x, y, width).
        h: Grid height.
        w: Grid width.

    Returns:
        List of (x, y) tuples on the segment border.
    """
    tiles: list[tuple[int, int]] = []

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


# ── Diagonal corner-passability ──────────────────────────────────

def _can_move_diagonal(
    grid: np.ndarray,
    x: int, y: int,
    dx: int, dy: int,
    passable: frozenset[int],
) -> bool:
    """Check if a diagonal move from (x,y) by (dx,dy) is valid.

    In DDNet, the tee has a non-zero hitbox (~28 pixels, roughly 2
    tiles).  A diagonal move through a 1-tile corner gap is impossible
    because the hitbox clips the corner:

        ##
        .#    Tee at (0,1) cannot reach (1,2) diagonally
        #.    because both (1,1) and (0,2) are solid.

    Rule: diagonal (dx,dy) from (x,y) is only valid if at least one
    of the two cardinal neighbors is also passable:
      - (x+dx, y)  — horizontal neighbor
      - (x, y+dy)  — vertical neighbor

    For cardinal moves (dx=0 or dy=0), always returns True.
    """
    if dx == 0 or dy == 0:
        return True  # cardinal move, always ok

    h, w = grid.shape
    # Horizontal neighbor
    hx, hy = x + dx, y
    h_ok = 0 <= hx < w and 0 <= hy < h and grid[hy, hx] in passable
    # Vertical neighbor
    vx, vy = x, y + dy
    v_ok = 0 <= vx < w and 0 <= vy < h and grid[vy, vx] in passable
    return h_ok or v_ok


# ── Core BFS functions ───────────────────────────────────────────

def bfs_flood(
    grid: np.ndarray,
    starts: list[tuple[int, int]],
    passable: frozenset[int] = PASSABLE,
    *,
    check_corners: bool = True,
) -> set[tuple[int, int]]:
    """Flood-fill BFS from start positions.  Returns all reachable tiles.

    This is the most general BFS — used when you need the full
    reachable set (e.g., reachability visualisation, island detection,
    gap analysis).

    Args:
        grid: 2D numpy tile grid (h, w).
        starts: Seed positions [(x, y), ...].
        passable: Which tile categories count as passable.
        check_corners: If True, apply diagonal corner-passability
            check.  Set False only for analysis-mode BFS where you
            want an optimistic reachability estimate.

    Returns:
        Set of (x, y) positions reachable from any start.
    """
    h, w = grid.shape
    visited: set[tuple[int, int]] = set()
    queue: deque[tuple[int, int]] = deque()

    for x, y in starts:
        if 0 <= y < h and 0 <= x < w and grid[y, x] in passable:
            visited.add((x, y))
            queue.append((x, y))

    while queue:
        x, y = queue.popleft()
        for dx, dy in DIRS_8:
            nx, ny = x + dx, y + dy
            if 0 <= ny < h and 0 <= nx < w and (nx, ny) not in visited:
                if grid[ny, nx] in passable:
                    if check_corners and not _can_move_diagonal(
                        grid, x, y, dx, dy, passable,
                    ):
                        continue
                    visited.add((nx, ny))
                    queue.append((nx, ny))

    return visited


def bfs_reachable(
    grid: np.ndarray,
    starts: list[tuple[int, int]],
    targets: set[tuple[int, int]],
    passable: frozenset[int] = PASSABLE,
    *,
    check_corners: bool = True,
) -> bool:
    """Quick check: can we reach ANY target from ANY start?

    Short-circuits as soon as a target is found — faster than a full
    flood when you only care about reachability, not the path.

    Args:
        grid: 2D tile grid.
        starts: Seed positions.
        targets: Goal positions.
        passable: Passable tile set.
        check_corners: Apply diagonal corner check.

    Returns:
        True if at least one target is reachable.
    """
    h, w = grid.shape
    visited: set[tuple[int, int]] = set()
    queue: deque[tuple[int, int]] = deque()

    for x, y in starts:
        if 0 <= y < h and 0 <= x < w and grid[y, x] in passable:
            visited.add((x, y))
            queue.append((x, y))

    while queue:
        x, y = queue.popleft()
        if (x, y) in targets:
            return True
        for dx, dy in DIRS_8:
            nx, ny = x + dx, y + dy
            if 0 <= ny < h and 0 <= nx < w and (nx, ny) not in visited:
                if grid[ny, nx] in passable:
                    if check_corners and not _can_move_diagonal(
                        grid, x, y, dx, dy, passable,
                    ):
                        continue
                    visited.add((nx, ny))
                    queue.append((nx, ny))

    return False


def bfs_path(
    grid: np.ndarray,
    starts: list[tuple[int, int]],
    targets: set[tuple[int, int]],
    passable: frozenset[int] = PASSABLE,
    *,
    check_corners: bool = True,
) -> list[tuple[int, int]] | None:
    """BFS that returns the actual shortest path.

    Uses parent-tracking for path reconstruction.  Returns None if
    no target is reachable.

    Args:
        grid: 2D tile grid.
        starts: Seed positions.
        targets: Goal positions.
        passable: Passable tile set.
        check_corners: Apply diagonal corner check.

    Returns:
        List of (x, y) from start to target (inclusive), or None.
    """
    h, w = grid.shape
    parent: dict[tuple[int, int], tuple[int, int] | None] = {}
    queue: deque[tuple[int, int]] = deque()

    for x, y in starts:
        if 0 <= y < h and 0 <= x < w and grid[y, x] in passable:
            parent[(x, y)] = None
            queue.append((x, y))

    target_hit: tuple[int, int] | None = None

    while queue:
        x, y = queue.popleft()
        if (x, y) in targets:
            target_hit = (x, y)
            break
        for dx, dy in DIRS_8:
            nx, ny = x + dx, y + dy
            if 0 <= ny < h and 0 <= nx < w and (nx, ny) not in parent:
                if grid[ny, nx] in passable:
                    if check_corners and not _can_move_diagonal(
                        grid, x, y, dx, dy, passable,
                    ):
                        continue
                    parent[(nx, ny)] = (x, y)
                    queue.append((nx, ny))

    if target_hit is None:
        return None

    # Reconstruct path from target back to start
    path: list[tuple[int, int]] = []
    pos: tuple[int, int] | None = target_hit
    while pos is not None:
        path.append(pos)
        pos = parent[pos]
    path.reverse()
    return path


def bfs_flood_with_steps(
    grid: np.ndarray,
    starts: list[tuple[int, int]],
    targets: set[tuple[int, int]],
    passable: frozenset[int] = PASSABLE,
    *,
    check_corners: bool = True,
) -> tuple[dict[tuple[int, int], int], int]:
    """Flood-fill BFS that tracks step counts (for validate_segment).

    Does a FULL flood (doesn't stop at target), so we get both the
    reachable set and the total passable coverage stats.

    Args:
        grid: 2D tile grid.
        starts: Seed positions.
        targets: Goal positions (checked during traversal).
        passable: Passable tile set.
        check_corners: Apply diagonal corner check.

    Returns:
        Tuple of:
          - visited dict: {(x, y): step_count}
          - exit_step: step count when first target was reached (-1 if never)
    """
    h, w = grid.shape
    visited: dict[tuple[int, int], int] = {}
    queue: deque[tuple[int, int, int]] = deque()

    for x, y in starts:
        if 0 <= y < h and 0 <= x < w and grid[y, x] in passable:
            visited[(x, y)] = 0
            queue.append((x, y, 0))

    exit_step = -1

    while queue:
        x, y, step = queue.popleft()

        if (x, y) in targets and exit_step == -1:
            exit_step = step

        for dx, dy in DIRS_8:
            nx, ny = x + dx, y + dy
            if 0 <= ny < h and 0 <= nx < w and (nx, ny) not in visited:
                if grid[ny, nx] in passable:
                    if check_corners and not _can_move_diagonal(
                        grid, x, y, dx, dy, passable,
                    ):
                        continue
                    visited[(nx, ny)] = step + 1
                    queue.append((nx, ny, step + 1))

    return visited, exit_step


# ── Gap bridging ─────────────────────────────────────────────────

def bridge_gaps(
    grid: np.ndarray,
    starts: list[tuple[int, int]],
    targets: set[tuple[int, int]],
    passable: frozenset[int] = PASSABLE,
    *,
    max_bridges: int = 5,
    channel_width: int = 2,
) -> bool:
    """Carve channels through solid walls to connect disconnected regions.

    This was duplicated in llm.py (_bridge_gap) and assemble.py
    (_validate_full_map).  Now there's one implementation.

    Algorithm:
    1. BFS flood from starts to find the reachable region.
    2. If any target is already reachable, return True immediately.
    3. Find solid tiles on the frontier of the reachable region.
    4. BFS through SOLID tiles from the frontier toward unreachable
       passable tiles (the "other side" of the gap).
    5. Trace the shortest path through solid and carve it to AIR,
       widened to `channel_width` tiles.
    6. Re-flood and check again.  Repeat up to `max_bridges` times.

    Args:
        grid: 2D tile grid — MODIFIED IN PLACE.
        starts: Seed positions (e.g., spawn tiles).
        targets: Goal positions (e.g., finish tiles).
        passable: Passable tile set for the flood.
        max_bridges: Maximum number of bridges to carve.
        channel_width: How wide to make each carved channel.

    Returns:
        True if targets are reachable after bridging (or were already).
    """
    h, w = grid.shape

    for bridge_num in range(max_bridges):
        # Flood from starts
        reachable = bfs_flood(grid, starts, passable)

        # Already connected?
        if any(t in reachable for t in targets):
            return True

        # Find solid tiles adjacent to reachable air (frontier)
        solid_frontier: set[tuple[int, int]] = set()
        for x, y in reachable:
            for dx, dy in DIRS_4:
                nx, ny = x + dx, y + dy
                if 0 <= ny < h and 0 <= nx < w and grid[ny, nx] == SOLID:
                    solid_frontier.add((nx, ny))

        if not solid_frontier:
            break  # no solid to carve through

        # Find unreachable passable tiles (the "other side")
        unreachable: set[tuple[int, int]] = set()
        for y in range(h):
            for x in range(w):
                if (x, y) not in reachable and grid[y, x] in passable:
                    unreachable.add((x, y))

        if not unreachable:
            break  # no disconnected air to bridge to

        # BFS through SOLID from frontier toward unreachable passable
        parent: dict[tuple[int, int], tuple[int, int] | None] = {}
        bfs_queue: deque[tuple[int, int]] = deque()

        for pos in solid_frontier:
            parent[pos] = None
            bfs_queue.append(pos)

        target_solid: tuple[int, int] | None = None

        while bfs_queue:
            bx, by = bfs_queue.popleft()

            # Check adjacency to unreachable passable
            for dx, dy in DIRS_4:
                nx, ny = bx + dx, by + dy
                if (nx, ny) in unreachable:
                    target_solid = (bx, by)
                    break
            if target_solid:
                break

            for dx, dy in DIRS_4:
                nx, ny = bx + dx, by + dy
                if 0 <= ny < h and 0 <= nx < w and (nx, ny) not in parent:
                    if grid[ny, nx] == SOLID:
                        parent[(nx, ny)] = (bx, by)
                        bfs_queue.append((nx, ny))

        if target_solid is None:
            break  # no solid path to the other side

        # Trace path and carve
        path: list[tuple[int, int]] = []
        pos: tuple[int, int] | None = target_solid
        while pos is not None:
            path.append(pos)
            pos = parent[pos]

        carved = 0
        for px, py in path:
            if grid[py, px] == SOLID:
                grid[py, px] = AIR
                carved += 1
            # Widen: carve one adjacent solid tile for channel_width=2
            if channel_width >= 2:
                for dx, dy in DIRS_4:
                    nx, ny = px + dx, py + dy
                    if 0 <= ny < h and 0 <= nx < w and grid[ny, nx] == SOLID:
                        grid[ny, nx] = AIR
                        carved += 1
                        break  # only widen by 1

        print(f"    Bridge {bridge_num + 1}: carved {carved} tiles")

    # Final check
    reachable = bfs_flood(grid, starts, passable)
    return any(t in reachable for t in targets)
