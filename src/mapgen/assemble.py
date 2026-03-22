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

Public API:
    stitch_segments() → (full_grid, entities)  — pure stitching, no I/O
    write_map()       → .map file              — twmap export
    assemble_map()    → Path                   — convenience wrapper (stitch + write)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import twmap

from mapgen.bfs import bfs_flood, bridge_gaps, PASSABLE
from mapgen.extract import AIR, SOLID, DEATH, FREEZE, NOHOOK, ENTITY
from mapgen.schema import Blueprint


# ── Reverse tile mapping: simplified category -> DDNet raw tile ID ──

TILE_MAP = {
    AIR:    0,    # air
    SOLID:  1,    # hookable solid
    DEATH:  2,    # death
    FREEZE: 9,    # freeze
    NOHOOK: 3,    # nohook (unhookable)
    ENTITY: 0,    # entities are placed separately; default to air
}

# DDNet game-layer tile IDs for race entities
SPAWN_ID = 192   # spawn point (entity offset + ENTITY_SPAWN)
START_ID = 33    # TILE_START -- begins the race timer
FINISH_ID = 34   # TILE_FINISH -- finishes the race

# Border between segments
BORDER_HEIGHT = 2  # solid rows between stacked segments


# ── Stitch segments (pure logic, no I/O) ─────────────────────────

def stitch_segments(
    segments: list[tuple[Blueprint, np.ndarray]],
) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """Stitch segment grids into a single full-map grid.

    Returns:
        (full_grid, entities) where entities is a list of
        (x, y, tile_id) tuples for spawn/start/finish placement.
    """
    if not segments:
        raise ValueError("Need at least one segment to assemble")

    # Compute total map dimensions
    max_width = max(grid.shape[1] for _, grid in segments)
    total_height = (
        sum(grid.shape[0] for _, grid in segments)
        + BORDER_HEIGHT * (len(segments) - 1)
    )

    pad = 2
    map_width = max_width + pad * 2
    map_height = total_height + pad * 2

    print(f"  Map dimensions: {map_width}x{map_height}")

    # Create full-map grid (all SOLID)
    full_grid = np.full((map_height, map_width), SOLID, dtype=np.uint8)
    entities: list[tuple[int, int, int]] = []

    # Place each segment and carve connections
    y_offset = pad
    prev_exit_range: tuple[int, int] | None = None

    for i, (bp, grid) in enumerate(segments):
        seg_h, seg_w = grid.shape
        x_offset = pad + (max_width - seg_w) // 2

        # Copy segment grid into full map
        full_grid[y_offset:y_offset + seg_h, x_offset:x_offset + seg_w] = grid

        # Carve tunnel through border connecting previous exit → this entry
        if prev_exit_range is not None:
            entry_x0, entry_x1 = _opening_x_range(bp.entry, x_offset)
            carve_x0 = min(prev_exit_range[0], entry_x0)
            carve_x1 = max(prev_exit_range[1], entry_x1)
            border_y0 = y_offset - BORDER_HEIGHT
            border_y1 = y_offset
            full_grid[border_y0:border_y1, carve_x0:carve_x1] = AIR

        prev_exit_range = _opening_x_range(bp.exit, x_offset)

        # Place entities
        if i == 0:
            ex, ey = _entry_position(bp, x_offset, y_offset)
            _enforce_entity_safety(full_grid, ex, ey, label="spawn")
            entities.append((ex, ey, SPAWN_ID))
            entities.append((ex + 1, ey, START_ID))

        if i == len(segments) - 1:
            fx, fy = _exit_position(bp, x_offset, y_offset, seg_h)
            _enforce_entity_safety(full_grid, fx, fy, label="finish")
            entities.append((fx, fy, FINISH_ID))
            entities.append((fx + 1, fy, FINISH_ID))

        y_offset += seg_h + BORDER_HEIGHT

    # Full-map playability check
    if entities:
        spawn_pos = [(x, y) for x, y, tid in entities if tid == SPAWN_ID]
        finish_pos = {(x, y) for x, y, tid in entities if tid == FINISH_ID}
        if spawn_pos and finish_pos:
            _validate_full_map(full_grid, spawn_pos, finish_pos)

    return full_grid, entities


# ── Write .map file ──────────────────────────────────────────────

def write_map(
    grid: np.ndarray,
    entities: list[tuple[int, int, int]],
    output_path: str,
    visual_grid: Optional[np.ndarray] = None,
    visual_flags: Optional[np.ndarray] = None,
    tileset_path: Optional[str] = None,
) -> Path:
    """Write a numpy grid + entities to a DDNet .map file via twmap.

    Args:
        grid: game layer tile grid (using simplified categories)
        entities: list of (x, y, tile_id) for spawn/start/finish
        output_path: where to save the .map file
        visual_grid: optional visual tile indices (same dimensions as grid)
        visual_flags: optional visual tile flags (XFLIP/YFLIP/ROTATE)
        tileset_path: path to tileset .png image (required if visual_grid is set)

    Returns:
        Path to saved .map file.
    """
    h, w = grid.shape
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    m = twmap.Map.empty("DDNet06")

    # Create physics group with game layer
    physics_group = m.groups.new_physics()
    game_layer = physics_group.layers.new_game(width=w, height=h)

    # Convert simplified categories → DDNet tile IDs
    tiles = game_layer.tiles
    for y in range(h):
        for x in range(w):
            cat = int(grid[y, x])
            tiles[y, x, 0] = TILE_MAP.get(cat, 0)
            tiles[y, x, 1] = 0

    # Place entities
    for ex, ey, tile_id in entities:
        if 0 <= ey < h and 0 <= ex < w:
            current = tiles[ey, ex, 0]
            if current not in (0, tile_id):
                print(f"  WARNING: Entity {tile_id} at ({ex},{ey}) "
                      f"overwrites tile {current}")
            tiles[ey, ex, 0] = tile_id
            tiles[ey, ex, 1] = 0

    game_layer.tiles = tiles

    # Add visual layer if provided
    if visual_grid is not None and tileset_path is not None:
        _add_visual_layer(m, visual_grid, visual_flags, tileset_path, w, h)

    m.save(str(out))
    print(f"  Map saved: {out}")
    return out


def _add_visual_layer(
    m: twmap.Map,
    visual_grid: np.ndarray,
    visual_flags: Optional[np.ndarray],
    tileset_path: str,
    w: int,
    h: int,
) -> None:
    """Add a visual tile layer to the map using the given tileset."""
    # Load tileset image
    tileset_img = twmap.Image.from_file(tileset_path)
    img_idx = m.images.append(tileset_img)

    # Create a design group for visual tiles
    design_group = m.groups.new_design()
    tile_layer = design_group.layers.new_tiles(width=w, height=h)
    tile_layer.image = img_idx

    tiles = tile_layer.tiles
    for y in range(h):
        for x in range(w):
            tiles[y, x, 0] = int(visual_grid[y, x])
            if visual_flags is not None:
                tiles[y, x, 1] = int(visual_flags[y, x])
            else:
                tiles[y, x, 1] = 0

    tile_layer.tiles = tiles


# ── Convenience wrapper ──────────────────────────────────────────

def assemble_map(
    segments: list[tuple[Blueprint, np.ndarray]],
    output_path: str,
) -> Path:
    """Stitch segments and write .map file. Convenience wrapper.

    This is the original API — kept for backward compatibility.
    For the LangGraph pipeline, use stitch_segments() + write_map()
    separately so the automap node can run in between.
    """
    full_grid, entities = stitch_segments(segments)
    return write_map(full_grid, entities, output_path)


# ── Entity safety zones ──────────────────────────────────────────

def _enforce_entity_safety(
    grid: np.ndarray, x: int, y: int, label: str = "entity",
) -> None:
    """Guarantee an entity position is not inside freeze or death tiles."""
    h, w = grid.shape
    radius = 2

    for dx in range(0, 2):
        ex = x + dx
        if 0 <= y < h and 0 <= ex < w:
            if grid[y, ex] not in (AIR,):
                grid[y, ex] = AIR

    for dy in range(-1, 2):
        for dx in range(-radius, radius + 1):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                if grid[ny, nx] in (FREEZE, DEATH):
                    grid[ny, nx] = AIR


# ── Full-map playability check ───────────────────────────────────

def _validate_full_map(
    grid: np.ndarray,
    spawn_pos: list[tuple[int, int]],
    finish_pos: set[tuple[int, int]],
) -> None:
    """BFS from spawn to finish. Bridge gaps if needed."""
    reachable = bfs_flood(grid, spawn_pos, PASSABLE)

    if any(pos in reachable for pos in finish_pos):
        print("  Full-map check: spawn -> finish CONNECTED")
        return

    print("  Full-map check: spawn -> finish DISCONNECTED! Bridging...")
    connected = bridge_gaps(grid, spawn_pos, finish_pos, PASSABLE)

    if connected:
        print("  Full-map check: spawn -> finish CONNECTED")
    else:
        print("  WARNING: Could not connect spawn -> finish after bridging!")


# ── Position helpers ─────────────────────────────────────────────

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
    else:
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
    else:
        return (x_off + bp.width - 2, y_off + bp.exit.y + bp.exit.width // 2)
