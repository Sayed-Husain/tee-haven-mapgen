"""Post-processing passes for generated segment grids.

Pure functions: grid in -> grid out.  No side effects.  Each function
fixes a specific category of issue that the LLM + builder can't handle
because they require tile-level spatial reasoning.

These run AFTER build_grid() and validation, BEFORE .map export.
The LangGraph post_process node calls them in sequence.

Functions:
  roughen_terrain()     — break up rectangular edges into organic shapes
  fix_edge_bugs()       — add freeze padding to prevent DDNet edge-bug exploits
  remove_freeze_blobs() — delete isolated freeze clusters off the main path
"""

from __future__ import annotations

import random
from collections import deque

import numpy as np

from mapgen.bfs import bfs_flood, bfs_reachable, opening_tiles, PASSABLE, DIRS_4
from mapgen.extract import AIR, SOLID, FREEZE, NOHOOK
from mapgen.schema import Opening


# ── Roughen terrain ──────────────────────────────────────────────

def roughen_terrain(
    grid: np.ndarray,
    entry: Opening,
    exit_: Opening,
    intensity: float = 0.25,
    seed: int | None = None,
) -> np.ndarray:
    """Break up perfect rectangular walls into organic jagged terrain.

    The builder produces clean rectangles because the LLM specifies
    obstacles with exact bounding boxes.  Real Gores maps have rough,
    uneven surfaces.  This pass randomly converts some "surface" solid
    tiles (solid tiles adjacent to air) into air.

    Safety: after roughening, we re-verify BFS connectivity from entry
    to exit.  If roughening breaks the path (unlikely but possible if
    it punches through a 1-tile-thick wall), we undo ALL changes and
    return the original grid.

    Args:
        grid: 2D tile grid (modified copy returned, original untouched).
        entry: Segment entry opening.
        exit_: Segment exit opening.
        intensity: Probability of converting each surface tile (0.0-1.0).
            0.25 = 25% of surface tiles get roughened.
        seed: Random seed for reproducibility (None = random).

    Returns:
        New grid with roughened terrain (or original if safety check fails).
    """
    rng = random.Random(seed)
    h, w = grid.shape
    result = grid.copy()

    # Find "surface" tiles: SOLID tiles with at least one AIR neighbor.
    # These are the visible edges of walls — the candidates for roughening.
    surface_tiles: list[tuple[int, int]] = []
    for y in range(h):
        for x in range(w):
            if result[y, x] != SOLID:
                continue
            for dx, dy in DIRS_4:
                nx, ny = x + dx, y + dy
                if 0 <= ny < h and 0 <= nx < w and result[ny, nx] == AIR:
                    surface_tiles.append((x, y))
                    break  # one air neighbor is enough

    if not surface_tiles:
        return result

    # Randomly convert some surface tiles to AIR
    converted = 0
    for x, y in surface_tiles:
        if rng.random() < intensity:
            result[y, x] = AIR
            converted += 1

    if converted == 0:
        return result

    # Safety check: verify path still exists after roughening
    entry_tiles = opening_tiles(entry, h, w)
    exit_tiles = set(opening_tiles(exit_, h, w))

    if not bfs_reachable(result, entry_tiles, exit_tiles, PASSABLE):
        # Roughening broke connectivity — return original unchanged
        print(f"    Roughen: reverted {converted} changes (broke connectivity)")
        return grid.copy()

    print(f"    Roughen: converted {converted}/{len(surface_tiles)} surface tiles")
    return result


# ── Fix edge bugs ────────────────────────────────────────────────

def fix_edge_bugs(grid: np.ndarray) -> np.ndarray:
    """Add freeze padding to prevent DDNet edge-bug exploits.

    DDNet has a physics quirk: when a player hooks a solid tile that
    has freeze on the opposite side, they can "clip" through the edge
    and skip the freeze entirely.  This is called an "edge bug" and
    skilled players routinely exploit it to bypass freeze corridors.

    The fix (from iMilchshake's gores-mapgen): for every air tile that
    is adjacent to a hookable SOLID tile AND has FREEZE on the opposite
    side, insert a 1-tile FREEZE buffer between the air and the solid.

    Before:
        AIR | SOLID | FREEZE     <- player hooks SOLID, clips into FREEZE

    After:
        AIR | FREEZE | SOLID | FREEZE   <- FREEZE buffer blocks the clip

    We do this by finding all SOLID tiles that are:
      1. Adjacent to AIR on one side (cardinal direction)
      2. Adjacent to FREEZE on the opposite side
    And converting the SOLID to FREEZE.  This "expands" freeze outward
    by 1 tile, blocking the edge-bug exploit.

    We DON'T modify tiles on the grid border (row 0, col 0, etc.)
    because those are the segment boundary walls.

    Args:
        grid: 2D tile grid (modified copy returned).

    Returns:
        New grid with edge-bug padding applied.
    """
    h, w = grid.shape
    result = grid.copy()

    # Opposite direction pairs: if AIR is at (x+dx, y+dy), check
    # FREEZE at (x-dx, y-dy)
    fixes = 0

    for y in range(1, h - 1):  # skip border rows
        for x in range(1, w - 1):  # skip border cols
            if result[y, x] != SOLID:
                continue

            for dx, dy in DIRS_4:
                # Air neighbor on one side
                ax, ay = x + dx, y + dy
                if not (0 <= ay < h and 0 <= ax < w):
                    continue
                if result[ay, ax] != AIR:
                    continue

                # Freeze on the opposite side
                fx, fy = x - dx, y - dy
                if not (0 <= fy < h and 0 <= fx < w):
                    continue
                if result[fy, fx] != FREEZE:
                    continue

                # This solid tile enables an edge-bug: convert to freeze
                result[y, x] = FREEZE
                fixes += 1
                break  # one direction is enough to flag this tile

    if fixes > 0:
        print(f"    Edge-bug fix: padded {fixes} solid tiles with freeze")

    return result


# ── Remove freeze blobs ──────────────────────────────────────────

def remove_freeze_blobs(
    grid: np.ndarray,
    entry: Opening,
    exit_: Opening,
    min_blob_size: int = 1,
) -> np.ndarray:
    """Remove disconnected freeze clusters that serve no gameplay purpose.

    The builder can produce isolated freeze patches — small groups of
    freeze tiles surrounded by solid, completely disconnected from the
    playable path.  These serve no purpose (the player never sees them)
    and waste map space.

    Algorithm:
    1. BFS flood from entry to find all reachable tiles (the "main path")
    2. Find all freeze tiles in the grid
    3. For each freeze tile NOT adjacent to any reachable tile, it's
       "orphaned" — part of a blob the player can never interact with
    4. Connected-component analysis on orphaned freeze tiles
    5. Remove blobs by converting to SOLID (they were hidden in walls anyway)

    We keep freeze tiles that ARE adjacent to reachable air — those are
    the actual gameplay obstacles the player must navigate around.

    Args:
        grid: 2D tile grid (modified copy returned).
        entry: Segment entry opening.
        exit_: Segment exit opening.
        min_blob_size: Minimum blob size to remove (default 1 = remove all).

    Returns:
        New grid with orphaned freeze blobs converted to SOLID.
    """
    h, w = grid.shape
    result = grid.copy()

    # Step 1: Find all tiles reachable from entry
    entry_tiles = opening_tiles(entry, h, w)
    reachable = bfs_flood(result, entry_tiles, PASSABLE)

    # Step 2: Find freeze tiles adjacent to reachable tiles ("active freeze")
    # These are the freeze obstacles the player actually encounters.
    active_freeze: set[tuple[int, int]] = set()
    for x, y in reachable:
        for dx, dy in DIRS_4:
            nx, ny = x + dx, y + dy
            if 0 <= ny < h and 0 <= nx < w and result[ny, nx] == FREEZE:
                active_freeze.add((nx, ny))

    # Step 3: Find ALL freeze tiles, then subtract active ones = orphaned
    all_freeze: set[tuple[int, int]] = set()
    for y in range(h):
        for x in range(w):
            if result[y, x] == FREEZE:
                all_freeze.add((x, y))

    orphaned = all_freeze - active_freeze
    # Also remove freeze tiles that ARE reachable (they're on the path,
    # part of the game space) — we only want truly isolated ones
    orphaned -= reachable

    if not orphaned:
        return result

    # Step 4: Connected-component analysis on orphaned freeze
    # (group them into blobs so we can report sizes)
    visited: set[tuple[int, int]] = set()
    blobs_removed = 0
    tiles_removed = 0

    for start in orphaned:
        if start in visited:
            continue

        # BFS to find this connected component
        component: list[tuple[int, int]] = []
        queue: deque[tuple[int, int]] = deque([start])
        visited.add(start)

        while queue:
            x, y = queue.popleft()
            component.append((x, y))
            for dx, dy in DIRS_4:
                nx, ny = x + dx, y + dy
                if (nx, ny) in orphaned and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))

        # Step 5: Remove blob if large enough
        if len(component) >= min_blob_size:
            for x, y in component:
                result[y, x] = SOLID
            blobs_removed += 1
            tiles_removed += len(component)

    if tiles_removed > 0:
        print(f"    Blob removal: removed {blobs_removed} orphaned freeze "
              f"blobs ({tiles_removed} tiles)")

    return result
