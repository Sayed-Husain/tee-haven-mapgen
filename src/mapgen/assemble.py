"""Multi-segment assembly and .map file export.

Takes generated segment grids, stitches them vertically,
adds spawn/finish entities, and writes a playable .map file
via twmap.

Stitching strategy:
- Segments stacked top-to-bottom (natural Gores flow)
- 2-tile solid border between segments
- Exit of segment N aligns with entry of segment N+1
- Spawn placed at first segment's entry
- Finish placed at last segment's exit
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import twmap

from mapgen.bfs import bfs_flood, bridge_gaps, PASSABLE
from mapgen.extract import AIR, SOLID, DEATH, FREEZE, NOHOOK, ENTITY
from mapgen.schema import Blueprint


# ── Reverse tile mapping: simplified category -> DDNet raw tile ID ──

_TILE_MAP = {
    AIR:    0,    # air
    SOLID:  1,    # hookable solid
    DEATH:  2,    # death
    FREEZE: 9,    # freeze
    NOHOOK: 3,    # nohook (unhookable)
    ENTITY: 0,    # entities are placed separately; default to air
}

# DDNet game-layer tile IDs for race entities
_SPAWN_ID = 192   # spawn point (entity offset + ENTITY_SPAWN)
_START_ID = 33    # TILE_START -- begins the race timer
_FINISH_ID = 34   # TILE_FINISH -- finishes the race


# ── Border between segments ─────────────────────────────────────────

BORDER_HEIGHT = 2  # solid rows between stacked segments


def assemble_map(
    segments: list[tuple[Blueprint, np.ndarray]],
    output_path: str,
) -> Path:
    """Stitch segments into a full .map file.

    Args:
        segments: List of (blueprint, grid) tuples in gameplay order.
        output_path: Where to save the .map file.

    Returns:
        Path to the saved .map file.
    """
    if not segments:
        raise ValueError("Need at least one segment to assemble")

    # ── Step 1: Compute total map dimensions ──
    max_width = max(grid.shape[1] for _, grid in segments)
    total_height = (
        sum(grid.shape[0] for _, grid in segments)
        + BORDER_HEIGHT * (len(segments) - 1)
    )

    # Add padding: 2-tile solid border around entire map
    pad = 2
    map_width = max_width + pad * 2
    map_height = total_height + pad * 2

    print(f"  Map dimensions: {map_width}x{map_height}")

    # ── Step 2: Create full-map grid (all SOLID) ──
    full_grid = np.full((map_height, map_width), SOLID, dtype=np.uint8)

    # Track entity positions (spawn, start, finish)
    entities: list[tuple[int, int, int]] = []  # (x, y, raw_tile_id)

    # ── Step 3: Place each segment and carve connections ──
    y_offset = pad
    prev_exit_range: tuple[int, int] | None = None  # (x0, x1) of previous exit

    for i, (bp, grid) in enumerate(segments):
        seg_h, seg_w = grid.shape

        # Center segment horizontally
        x_offset = pad + (max_width - seg_w) // 2

        # Copy segment grid into full map
        full_grid[y_offset:y_offset + seg_h, x_offset:x_offset + seg_w] = grid

        # ── Carve tunnel through border connecting previous exit -> this entry ──
        if prev_exit_range is not None:
            entry_x0, entry_x1 = _opening_x_range(bp.entry, x_offset)
            # Use the union of both openings so nothing is blocked
            carve_x0 = min(prev_exit_range[0], entry_x0)
            carve_x1 = max(prev_exit_range[1], entry_x1)
            border_y0 = y_offset - BORDER_HEIGHT
            border_y1 = y_offset
            full_grid[border_y0:border_y1, carve_x0:carve_x1] = AIR

        # ── Track this segment's exit for the next border ──
        prev_exit_range = _opening_x_range(bp.exit, x_offset)

        # ── Place entities with safety zones ──
        if i == 0:
            # First segment: spawn + start at entry
            ex, ey = _entry_position(bp, x_offset, y_offset)
            _enforce_entity_safety(full_grid, ex, ey, label="spawn")
            entities.append((ex, ey, _SPAWN_ID))
            entities.append((ex + 1, ey, _START_ID))

        if i == len(segments) - 1:
            # Last segment: finish at exit
            fx, fy = _exit_position(bp, x_offset, y_offset, seg_h)
            _enforce_entity_safety(full_grid, fx, fy, label="finish")
            entities.append((fx, fy, _FINISH_ID))
            entities.append((fx + 1, fy, _FINISH_ID))

        y_offset += seg_h + BORDER_HEIGHT

    # ── Step 4: Full-map playability check ──
    # Each segment was validated individually, but we need to verify
    # the STITCHED map is connected end-to-end. Border tunnels or
    # misaligned exits can leave segments disconnected.
    if entities:
        spawn_pos = [(x, y) for x, y, tid in entities if tid == _SPAWN_ID]
        finish_pos = {(x, y) for x, y, tid in entities if tid == _FINISH_ID}
        if spawn_pos and finish_pos:
            _validate_full_map(full_grid, spawn_pos, finish_pos)

    # ── Step 5: Write .map via twmap ──
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    _write_map(full_grid, entities, str(out))

    print(f"  Map saved: {out}")
    return out


# ── Entity safety zones ────────────────────────────────────────────

def _enforce_entity_safety(
    grid: np.ndarray, x: int, y: int, label: str = "entity",
) -> None:
    """Guarantee an entity position is not inside freeze or death tiles.

    In Gores, the player falls from the entry -- they don't need a
    platform to stand on. We must NOT add solid blocks because that
    would block the downward path into the segment.

    All we do here is:
    - Ensure the entity tile itself is AIR (so the entity works)
    - Convert any freeze/death tiles in a small zone around it to
      AIR so the player has a few frames to orient before navigating

    Previously this was two near-identical functions (_enforce_spawn_safety
    and _enforce_finish_safety). Now there's one.

    Modifies grid in-place.
    """
    h, w = grid.shape
    radius = 2  # 2 tiles left/right of entity center

    # Ensure entity + adjacent tile positions are AIR
    for dx in range(0, 2):  # entity at (x, y), second at (x+1, y)
        ex = x + dx
        if 0 <= y < h and 0 <= ex < w:
            if grid[y, ex] not in (AIR,):
                grid[y, ex] = AIR

    # Clear freeze/death in a small buffer (5 wide x 3 tall)
    for dy in range(-1, 2):
        for dx in range(-radius, radius + 1):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                if grid[ny, nx] in (FREEZE, DEATH):
                    grid[ny, nx] = AIR


# ── Full-map playability check ─────────────────────────────────────

def _validate_full_map(
    grid: np.ndarray,
    spawn_pos: list[tuple[int, int]],
    finish_pos: set[tuple[int, int]],
) -> None:
    """BFS from spawn to finish on the stitched map. Bridge gaps if needed.

    This is the final safety net -- each segment passed validation
    individually, but the assembled map might have disconnections at
    segment borders. If BFS fails, we carve 2-tile-wide channels
    through solid walls to connect the regions.

    Uses bfs.bridge_gaps() -- the shared implementation that was
    previously duplicated here and in llm.py.

    Modifies grid in-place.
    """
    # Quick check first
    reachable = bfs_flood(grid, spawn_pos, PASSABLE)

    if any(pos in reachable for pos in finish_pos):
        print("  Full-map check: spawn -> finish CONNECTED")
        return

    # Not connected -- bridge
    print("  Full-map check: spawn -> finish DISCONNECTED! Bridging...")
    connected = bridge_gaps(grid, spawn_pos, finish_pos, PASSABLE)

    if connected:
        print("  Full-map check: spawn -> finish CONNECTED")
    else:
        print("  WARNING: Could not connect spawn -> finish after bridging!")


# ── Position helpers ────────────────────────────────────────────────

def _opening_x_range(opening, x_off: int) -> tuple[int, int]:
    """Get the global x-range (x0, x1) of an entry/exit opening."""
    return (x_off + opening.x, x_off + opening.x + opening.width)


def _entry_position(bp: Blueprint, x_off: int, y_off: int) -> tuple[int, int]:
    """Get the world position of the entry opening center."""
    if bp.entry.side == "top":
        return (x_off + bp.entry.x + bp.entry.width // 2, y_off + 1)
    elif bp.entry.side == "left":
        return (x_off + 1, y_off + bp.entry.y + bp.entry.width // 2)
    elif bp.entry.side == "right":
        return (x_off + bp.width - 2, y_off + bp.entry.y + bp.entry.width // 2)
    else:  # bottom
        return (x_off + bp.entry.x + bp.entry.width // 2, y_off + bp.height - 2)


def _exit_position(
    bp: Blueprint, x_off: int, y_off: int, seg_h: int,
) -> tuple[int, int]:
    """Get the world position of the exit opening center."""
    if bp.exit.side == "bottom":
        return (x_off + bp.exit.x + bp.exit.width // 2, y_off + seg_h - 2)
    elif bp.exit.side == "top":
        return (x_off + bp.exit.x + bp.exit.width // 2, y_off + 1)
    elif bp.exit.side == "left":
        return (x_off + 1, y_off + bp.exit.y + bp.exit.width // 2)
    else:  # right
        return (x_off + bp.width - 2, y_off + bp.exit.y + bp.exit.width // 2)


# ── twmap .map writer ──────────────────────────────────────────────

def _write_map(
    grid: np.ndarray,
    entities: list[tuple[int, int, int]],
    output_path: str,
) -> None:
    """Write a numpy grid + entities to a DDNet .map file via twmap."""
    h, w = grid.shape

    m = twmap.Map.empty("DDNet06")

    # Create physics group with game layer
    physics_group = m.groups.new_physics()
    game_layer = physics_group.layers.new_game(width=w, height=h)

    # Convert simplified categories -> DDNet tile IDs
    tiles = game_layer.tiles  # shape: (h, w, 2) -> [tile_id, flags]

    for y in range(h):
        for x in range(w):
            cat = int(grid[y, x])
            tiles[y, x, 0] = _TILE_MAP.get(cat, 0)
            tiles[y, x, 1] = 0  # no flags

    # Place entities (overwrite tile IDs at entity positions)
    for ex, ey, tile_id in entities:
        if 0 <= ey < h and 0 <= ex < w:
            current = tiles[ey, ex, 0]
            if current not in (0, tile_id):
                print(f"  WARNING: Entity {tile_id} at ({ex},{ey}) "
                      f"overwrites tile {current} - forcing to entity")
            tiles[ey, ex, 0] = tile_id
            tiles[ey, ex, 1] = 0

    game_layer.tiles = tiles

    m.save(output_path)
