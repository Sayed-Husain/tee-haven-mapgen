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
from mapgen.schema import Blueprint, Opening, CheckpointSpec


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

        # Place entities — scan actual grid for safe AIR positions
        if i == 0:
            # Spawn: find AIR tiles in upper portion of walker-generated lobby
            exit_left = x_offset + bp.exit.x
            exit_right = exit_left + bp.exit.width
            spawn_placed = 0

            for sy in range(y_offset + 3, y_offset + 7):
                for sx in range(x_offset + 4, x_offset + seg_w - 4, 2):
                    if not (0 <= sy < map_height and 0 <= sx < map_width):
                        continue
                    if full_grid[sy, sx] != AIR:
                        continue
                    # Skip tiles above exit corridor
                    if exit_left <= sx <= exit_right:
                        continue
                    entities.append((sx, sy, SPAWN_ID))
                    spawn_placed += 1
                if spawn_placed >= 8:
                    break

            # Start line: find first air row in exit corridor area
            exit_cx = (exit_left + exit_right) // 2
            for start_y in range(y_offset + seg_h - 6, y_offset + seg_h - 1):
                if 0 <= start_y < map_height and full_grid[start_y, exit_cx] == AIR:
                    for sx in range(exit_left, exit_right):
                        if 0 <= sx < map_width and full_grid[start_y, sx] == AIR:
                            entities.append((sx, start_y, START_ID))
                    break

        if i == len(segments) - 1:
            # Finish: find wide air row in lower portion of walker-generated lobby
            for fy in range(y_offset + seg_h - 8, y_offset + seg_h - 3):
                row_tiles = []
                for fx in range(x_offset + 4, x_offset + seg_w - 4):
                    if 0 <= fy < map_height and 0 <= fx < map_width:
                        if full_grid[fy, fx] == AIR:
                            row_tiles.append(fx)
                if len(row_tiles) >= 5:
                    for fx in row_tiles:
                        entities.append((fx, fy, FINISH_ID))
                    break

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
    visual_layers: Optional[list] = None,
    background: Optional[dict] = None,
) -> Path:
    """Write a numpy grid + entities to a DDNet .map file via twmap.

    Args:
        grid: game layer tile grid (using simplified categories)
        entities: list of (x, y, tile_id) for spawn/start/finish
        output_path: where to save the .map file
        visual_layers: list of VisualLayer objects (from automap.py)
        background: dict with 'top_color' and 'bottom_color' as (R,G,B,A)
                    tuples for a vertical gradient background

    Returns:
        Path to saved .map file.
    """
    h, w = grid.shape
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    m = twmap.Map.empty("DDNet06")

    # Add background gradient quad (renders behind everything)
    if background:
        _add_background(m, w, h, background)

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

    # Add visual layers
    if visual_layers:
        for vlayer in visual_layers:
            _add_visual_layer(m, vlayer.indices, vlayer.flags,
                              vlayer.tileset_path, w, h,
                              color=vlayer.color)

    # Add start/finish marker quads (rendered on top of everything)
    _add_markers(m, entities)

    # Add direction arrows at segment junctions
    if visual_layers:
        # Calculate segment junction y-positions from grid height
        _add_direction_arrows(m, grid, entities)

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
    color: tuple[int, int, int, int] = (255, 255, 255, 255),
) -> None:
    """Add a visual tile layer to the map using the given tileset."""
    img = m.images.new_from_file(tileset_path)
    img_idx = len(m.images) - 1

    design_group = m.groups.new()
    tile_layer = design_group.layers.new_tiles(width=w, height=h)
    tile_layer.image = img_idx
    tile_layer.color = color  # RGBA tint for shape-only tilesets

    tiles = tile_layer.tiles
    for y in range(h):
        for x in range(w):
            tiles[y, x, 0] = int(visual_grid[y, x])
            if visual_flags is not None:
                tiles[y, x, 1] = int(visual_flags[y, x])
            else:
                tiles[y, x, 1] = 0

    tile_layer.tiles = tiles


def _add_markers(
    m: twmap.Map,
    entities: list[tuple[int, int, int]],
) -> None:
    """Add colored marker quads at spawn and finish positions.

    Creates a foreground group with semi-transparent colored quads:
    - Green glow at spawn/start area
    - Red glow at finish area
    """
    spawn_pos = [(x, y) for x, y, tid in entities if tid == SPAWN_ID]
    start_pos = [(x, y) for x, y, tid in entities if tid == START_ID]
    finish_pos = [(x, y) for x, y, tid in entities if tid == FINISH_ID]

    if not spawn_pos and not finish_pos:
        return

    marker_group = m.groups.new()
    marker_layer = marker_group.layers.new_quads()

    # Start marker — green overlay at start line (not spawn)
    if start_pos:
        min_x = min(x for x, y in start_pos)
        max_x = max(x for x, y in start_pos) + 1
        sy = start_pos[0][1]
        cx = (min_x + max_x) / 2
        tile_w = max_x - min_x

        q = marker_layer.quads.new(cx, sy + 0.5, tile_w, 1)
        q.colors = [
            (50, 255, 50, 100),
            (50, 255, 50, 100),
            (50, 255, 50, 100),
            (50, 255, 50, 100),
        ]

    # Finish marker — red overlay exactly matching finish tiles
    if finish_pos:
        min_x = min(x for x, y in finish_pos)
        max_x = max(x for x, y in finish_pos) + 1
        fy = finish_pos[0][1]
        cx = (min_x + max_x) / 2
        tile_w = max_x - min_x

        q = marker_layer.quads.new(cx, fy + 0.5, tile_w, 1)
        q.colors = [
            (255, 50, 50, 100),
            (255, 50, 50, 100),
            (255, 50, 50, 100),
            (255, 50, 50, 100),
        ]


def _add_direction_arrows(
    m: twmap.Map,
    grid: np.ndarray,
    entities: list[tuple[int, int, int]],
) -> None:
    """Add downward-pointing arrow quads to guide the player.

    Places semi-transparent arrow markers at regular intervals along
    the air path. Arrows are small triangular quads pointing downward
    (since maps flow top-to-bottom).
    """
    h, w = grid.shape
    arrow_group = m.groups.new()
    arrow_layer = arrow_group.layers.new_quads()

    # Find the center of air passages at regular y intervals
    spawn_y = min((y for _, y, tid in entities if tid == SPAWN_ID), default=5)
    finish_y = max((y for _, y, tid in entities if tid == FINISH_ID), default=h - 5)

    # Place arrows every 40 tiles vertically
    arrow_spacing = 40
    arrow_count = 0

    for ay in range(spawn_y + 20, finish_y - 10, arrow_spacing):
        # Find center of air at this y level
        air_xs = [x for x in range(w) if grid[ay, x] == AIR]
        if not air_xs:
            continue

        cx = (min(air_xs) + max(air_xs)) / 2

        # Create a downward arrow using a small quad
        # Arrow body
        q = arrow_layer.quads.new(cx, ay, 2, 3)
        q.colors = [
            (255, 255, 255, 60),  # top - wider
            (255, 255, 255, 60),
            (255, 255, 255, 30),  # bottom - faded
            (255, 255, 255, 30),
        ]
        arrow_count += 1

    if arrow_count > 0:
        print(f"  Added {arrow_count} direction arrows")


def _add_background(
    m: twmap.Map,
    map_w: int,
    map_h: int,
    bg: dict,
) -> None:
    """Add a background gradient quad behind the map.

    Creates a group with a single colorless quad layer. The quad covers
    the entire map area with a vertical gradient (top_color → bottom_color).
    """
    top = tuple(bg.get("top_color", (80, 120, 180, 255)))
    bottom = tuple(bg.get("bottom_color", (20, 30, 50, 255)))

    bg_group = m.groups.new()
    bg_layer = bg_group.layers.new_quads()

    # Quad covers map area with generous margin
    cx = map_w // 2
    cy = map_h // 2
    qw = map_w + 40   # extra margin on sides
    qh = map_h + 40   # extra margin top/bottom

    q = bg_layer.quads.new(cx, cy, qw, qh)
    q.colors = [
        top,       # top-left
        top,       # top-right
        bottom,    # bottom-left
        bottom,    # bottom-right
    ]


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


# ── Spawn / Finish dedicated segments ────────────────────────────

SPAWN_SEG_HEIGHT = 25  # room for organic carving
FINISH_SEG_HEIGHT = 20


def build_spawn_segment(
    width: int, exit_x: int, exit_width: int = 5, seed: int | None = None,
) -> tuple[Blueprint, np.ndarray]:
    """Build an organic spawn lobby using the walker engine.

    Instead of a static rectangle, runs the walker with wide-open
    params to carve a unique, organic-shaped lobby room. Then
    validates that spawn positions are safe and exit connects.

    Args:
        width: segment width (matches other segments)
        exit_x: x-position of exit opening center
        exit_width: width of exit opening
        seed: random seed for walker (None = random)
    """
    from .walker import generate_segment_grid, WalkerConfig

    h = SPAWN_SEG_HEIGHT

    # Walker config for lobby — moderate size, organic shape
    lobby_config = WalkerConfig(
        inner_size=6,           # moderate open space
        outer_margin=2,         # will be converted to solid
        inner_circularity=0.4,
        outer_circularity=0.3,
        momentum_prob=0.4,
        shift_weights=(0.25, 0.25, 0.25, 0.25),
        size_mutate_prob=0.05,
        size_range=(5, 8),
        circ_mutate_prob=0.02,
        waypoint_reached_dist=6,
        fade_steps=0,
        fade_min_size=5,
    )

    grid, _ = generate_segment_grid(
        width=width,
        height=h,
        config=lobby_config,
        n_waypoints=3,
        seed=seed,
        entry_x=width // 2,
        exit_x=exit_x,
    )

    # LOBBY SAFETY: convert ALL freeze to solid (no danger in spawn area)
    grid[grid == FREEZE] = SOLID

    # Add solid floor platforms for players to stand on
    # Find the lowest air row and place a platform below it
    for y in range(h - 4, 4, -1):
        air_count = int(np.sum(grid[y, :] == AIR))
        if air_count > 10:
            # Place solid platform 2 rows below this air row
            plat_y = min(y + 2, h - 3)
            air_xs = [x for x in range(width) if grid[y, x] == AIR]
            if air_xs:
                for x in range(min(air_xs), max(air_xs) + 1):
                    grid[plat_y, x] = SOLID
            break

    # Carve entry/exit openings
    from .builder import _carve_opening
    spawn_w = min(width - 6, 20)
    spawn_x0 = (width - spawn_w) // 2

    entry = Opening(side="top", x=spawn_x0, y=0, width=spawn_w)
    exit_w = max(exit_width, 7)
    exit_x0 = max(1, exit_x - exit_w // 2)
    exit_ = Opening(side="bottom", x=exit_x0, y=0, width=exit_w)

    _carve_opening(grid, entry)
    _carve_opening(grid, exit_)

    cp = CheckpointSpec(x=width // 2 - 3, y=h // 2, width=6)
    bp = Blueprint(width=width, height=h, difficulty="easy",
                   entry=entry, exit=exit_, checkpoint=cp, obstacles=[])

    return bp, grid


def build_finish_segment(
    width: int, entry_x: int, entry_width: int = 5, seed: int | None = None,
) -> tuple[Blueprint, np.ndarray]:
    """Build an organic finish area using the walker engine.

    Same approach as spawn lobby — walker with wide-open params
    for a unique, organic-shaped landing room.
    """
    from .walker import generate_segment_grid, WalkerConfig

    h = FINISH_SEG_HEIGHT

    lobby_config = WalkerConfig(
        inner_size=6,
        outer_margin=2,
        inner_circularity=0.4,
        outer_circularity=0.3,
        momentum_prob=0.4,
        shift_weights=(0.25, 0.25, 0.25, 0.25),
        size_mutate_prob=0.05,
        size_range=(5, 8),
        circ_mutate_prob=0.02,
        waypoint_reached_dist=6,
        fade_steps=0,
        fade_min_size=5,
    )

    grid, _ = generate_segment_grid(
        width=width,
        height=h,
        config=lobby_config,
        n_waypoints=3,
        seed=seed,
        entry_x=entry_x,
        exit_x=width // 2,
    )

    # LOBBY SAFETY: convert ALL freeze to solid (safe landing area)
    grid[grid == FREEZE] = SOLID

    # Add solid floor platform for landing
    for y in range(h - 4, 4, -1):
        air_count = int(np.sum(grid[y, :] == AIR))
        if air_count > 10:
            plat_y = min(y + 2, h - 3)
            air_xs = [x for x in range(width) if grid[y, x] == AIR]
            if air_xs:
                for x in range(min(air_xs), max(air_xs) + 1):
                    grid[plat_y, x] = SOLID
            break

    # Carve entry/exit openings
    from .builder import _carve_opening
    entry_w = max(entry_width, 7)
    entry_x0 = max(1, entry_x - entry_w // 2)
    entry = Opening(side="top", x=entry_x0, y=0, width=entry_w)

    finish_w = min(width - 6, 20)
    finish_x0 = (width - finish_w) // 2
    exit_ = Opening(side="bottom", x=finish_x0, y=0, width=finish_w)

    _carve_opening(grid, entry)
    _carve_opening(grid, exit_)

    cp = CheckpointSpec(x=width // 2 - 3, y=h // 2, width=6)
    bp = Blueprint(width=width, height=h, difficulty="easy",
                   entry=entry, exit=exit_, checkpoint=cp, obstacles=[])

    return bp, grid


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
