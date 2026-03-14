"""Blueprint-to-grid converter.

Takes a validated Blueprint and produces a numpy tile grid.

Strategy: start from an all-SOLID grid and carve openings.
- Everything not explicitly carved is wall (safe by default)
- Entry/exit carving guarantees those are the only border openings
- Each obstacle pattern carves its own playable space
"""

from __future__ import annotations

import numpy as np

from mapgen.extract import AIR, SOLID, DEATH, FREEZE, NOHOOK, ENTITY
from mapgen.schema import Blueprint, Opening, Obstacle


def build_grid(bp: Blueprint) -> np.ndarray:
    """Convert a Blueprint into a 2D numpy tile grid.

    Pipeline:
      1. Fill everything with SOLID
      2. Carve entry and exit openings on the border
      3. Place the checkpoint platform (SOLID row + AIR above)
      4. Place each obstacle via its pattern function
      5. Return the grid

    Returns:
        numpy.ndarray of shape (height, width) with dtype uint8.
    """
    grid = np.full((bp.height, bp.width), SOLID, dtype=np.uint8)

    # Carve entry and exit
    _carve_opening(grid, bp.entry)
    _carve_opening(grid, bp.exit)

    # Place checkpoint — solid platform row with air above
    _place_checkpoint(grid, bp.checkpoint.x, bp.checkpoint.y, bp.checkpoint.width)

    # Place each obstacle
    for obs in bp.obstacles:
        _place_obstacle(grid, obs)

    return grid


# ── Opening carving ─────────────────────────────────────────────────

def _carve_opening(grid: np.ndarray, opening: Opening) -> None:
    """Carve an entry/exit opening on the segment border.

    Replaces SOLID tiles with AIR at the border edge and a few
    tiles deep inward so the player can actually enter/exit.
    """
    h, w = grid.shape
    depth = 3  # how many tiles deep to carve inward

    if opening.side == "top":
        x0 = max(0, opening.x)
        x1 = min(w, opening.x + opening.width)
        y1 = min(h, depth)
        grid[0:y1, x0:x1] = AIR

    elif opening.side == "bottom":
        x0 = max(0, opening.x)
        x1 = min(w, opening.x + opening.width)
        y0 = max(0, h - depth)
        grid[y0:h, x0:x1] = AIR

    elif opening.side == "left":
        y0 = max(0, opening.y)
        y1 = min(h, opening.y + opening.width)
        x1 = min(w, depth)
        grid[y0:y1, 0:x1] = AIR

    elif opening.side == "right":
        y0 = max(0, opening.y)
        y1 = min(h, opening.y + opening.width)
        x0 = max(0, w - depth)
        grid[y0:y1, x0:w] = AIR


# ── Checkpoint ──────────────────────────────────────────────────────

def _place_checkpoint(grid: np.ndarray, x: int, y: int, width: int) -> None:
    """Place a checkpoint platform: SOLID row at y, AIR for 2 rows above.

    The player needs at least 2 tiles of air above the platform to
    stand (1 tile for the tee body, 1 for headroom).
    """
    h, w = grid.shape
    x0 = max(0, x)
    x1 = min(w, x + width)

    # Platform surface
    if 0 <= y < h:
        grid[y, x0:x1] = SOLID

    # Air above (2 rows for player + headroom)
    for dy in range(1, 3):
        row = y - dy
        if 0 <= row < h:
            grid[row, x0:x1] = AIR


# ── Obstacle dispatcher ────────────────────────────────────────────

def _place_obstacle(grid: np.ndarray, obs: Obstacle) -> None:
    """Route an obstacle to its specific pattern function."""
    dispatch = {
        "platform": _place_platform,
        "freeze_corridor": _place_freeze_corridor,
        "death_zone": _place_death_zone,
        "hook_point": _place_hook_point,
        "wall_gap": _place_wall_gap,
        "nohook_wall": _place_nohook_wall,
        "narrow_passage": _place_narrow_passage,
    }
    fn = dispatch.get(obs.type)
    if fn:
        fn(grid, obs)


# ── Pattern functions ───────────────────────────────────────────────
# Each takes the full grid + the Obstacle, modifies grid in-place.

def _place_platform(grid: np.ndarray, obs: Obstacle) -> None:
    """Flat SOLID surface with AIR above.

    Like a checkpoint but placed as an obstacle (shorter rest point,
    stepping stone).  2 rows of air above the platform.
    """
    h, w = grid.shape
    x0, x1 = max(0, obs.x), min(w, obs.x + obs.width)
    y = obs.y

    # Platform row
    if 0 <= y < h:
        grid[y, x0:x1] = SOLID

    # Air above (2 rows)
    for dy in range(1, 3):
        row = y - dy
        if 0 <= row < h:
            grid[row, x0:x1] = AIR


def _place_freeze_corridor(grid: np.ndarray, obs: Obstacle) -> None:
    """Air corridor bordered by FREEZE tiles.

    direction (from params):
      - "horizontal" (default): carves left→right
      - "vertical": carves top→bottom

    Layout (horizontal, height=4):
      FREEZE FREEZE FREEZE ...
      AIR    AIR    AIR    ...
      AIR    AIR    AIR    ...
      FREEZE FREEZE FREEZE ...

    Minimum height=3 (horizontal) or width=3 (vertical) enforced
    so there's always at least 1 row/col of air interior.
    """
    h, w = grid.shape
    direction = obs.params.get("direction", "horizontal")

    # Enforce minimum dimensions so the corridor has interior air
    eff_width = max(obs.width, 3)
    eff_height = max(obs.height, 3)
    obs = Obstacle(obs.type, obs.x, obs.y, eff_width, eff_height, obs.params)

    x0, x1 = max(0, obs.x), min(w, obs.x + obs.width)
    y0, y1 = max(0, obs.y), min(h, obs.y + obs.height)

    if direction == "horizontal":
        # Top and bottom rows = FREEZE, interior = AIR
        for y in range(y0, y1):
            for x in range(x0, x1):
                if y == y0 or y == y1 - 1:
                    grid[y, x] = FREEZE
                else:
                    grid[y, x] = AIR
    else:  # vertical
        # Left and right columns = FREEZE, interior = AIR
        for y in range(y0, y1):
            for x in range(x0, x1):
                if x == x0 or x == x1 - 1:
                    grid[y, x] = FREEZE
                else:
                    grid[y, x] = AIR


def _place_death_zone(grid: np.ndarray, obs: Obstacle) -> None:
    """Rectangular area of DEATH tiles.

    Player must jump over or find a way around.  Surrounded by
    a 1-tile air border so the player can see and avoid it.
    """
    h, w = grid.shape
    x0, x1 = max(0, obs.x), min(w, obs.x + obs.width)
    y0, y1 = max(0, obs.y), min(h, obs.y + obs.height)

    # Air border around the death zone (1 tile)
    ax0, ax1 = max(0, x0 - 1), min(w, x1 + 1)
    ay0, ay1 = max(0, y0 - 1), min(h, y1 + 1)
    grid[ay0:ay1, ax0:ax1] = AIR

    # Death tiles in the center
    grid[y0:y1, x0:x1] = DEATH


def _place_hook_point(grid: np.ndarray, obs: Obstacle) -> None:
    """1-2 SOLID tiles in open air for hooking.

    Carves a region of AIR and places a small solid block
    the player must hook onto to traverse.
    """
    h, w = grid.shape

    # Carve air region (obstacle bounding box)
    x0, x1 = max(0, obs.x), min(w, obs.x + obs.width)
    y0, y1 = max(0, obs.y), min(h, obs.y + obs.height)
    grid[y0:y1, x0:x1] = AIR

    # Place 1-2 solid hook tiles in the center
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    if 0 <= cy < h and 0 <= cx < w:
        grid[cy, cx] = SOLID
    # Second tile to the right if space allows
    if 0 <= cy < h and 0 <= cx + 1 < w and cx + 1 < x1:
        grid[cy, cx + 1] = SOLID


def _place_wall_gap(grid: np.ndarray, obs: Obstacle) -> None:
    """Solid wall with an AIR gap the player must pass through.

    params:
      - gap_size (int): height of the gap in tiles (default: 3)
      - gap_y (int): offset from top of wall where gap starts (default: auto-center)
    """
    h, w = grid.shape
    x0, x1 = max(0, obs.x), min(w, obs.x + obs.width)
    y0, y1 = max(0, obs.y), min(h, obs.y + obs.height)

    gap_size = obs.params.get("gap_size", 3)
    gap_y = obs.params.get("gap_y", None)

    wall_h = y1 - y0
    if gap_y is None:
        # Center the gap
        gap_y = max(0, (wall_h - gap_size) // 2)

    # Fill wall with SOLID
    grid[y0:y1, x0:x1] = SOLID

    # Carve the gap
    gy0 = y0 + gap_y
    gy1 = min(y1, gy0 + gap_size)
    grid[gy0:gy1, x0:x1] = AIR

    # Carve air on both sides of the wall so player can approach
    air_depth = 2
    # Left side
    lx0 = max(0, x0 - air_depth)
    grid[gy0:gy1, lx0:x0] = AIR
    # Right side
    rx1 = min(w, x1 + air_depth)
    grid[gy0:gy1, x1:rx1] = AIR


def _place_nohook_wall(grid: np.ndarray, obs: Obstacle) -> None:
    """NOHOOK rectangle — player's hook doesn't attach.

    Forces creative movement: player must use momentum or
    other surfaces to get past.
    """
    h, w = grid.shape
    x0, x1 = max(0, obs.x), min(w, obs.x + obs.width)
    y0, y1 = max(0, obs.y), min(h, obs.y + obs.height)
    grid[y0:y1, x0:x1] = NOHOOK


def _place_narrow_passage(grid: np.ndarray, obs: Obstacle) -> None:
    """Tight AIR corridor between walls.

    params:
      - direction: "horizontal" or "vertical" (default: "horizontal")
      - gap_width: width of the passage in tiles (default: 2)

    Carves a narrow air path through the bounding box,
    leaving walls on both sides.
    """
    h, w = grid.shape
    direction = obs.params.get("direction", "horizontal")
    gap_width = obs.params.get("gap_width", 2)

    x0, x1 = max(0, obs.x), min(w, obs.x + obs.width)
    y0, y1 = max(0, obs.y), min(h, obs.y + obs.height)

    # Ensure the area is solid first
    grid[y0:y1, x0:x1] = SOLID

    if direction == "horizontal":
        # Carve horizontal air strip in the center
        cy = (y0 + y1) // 2
        gy0 = max(y0, cy - gap_width // 2)
        gy1 = min(y1, gy0 + gap_width)
        grid[gy0:gy1, x0:x1] = AIR
    else:  # vertical
        # Carve vertical air strip in the center
        cx = (x0 + x1) // 2
        gx0 = max(x0, cx - gap_width // 2)
        gx1 = min(x1, gx0 + gap_width)
        grid[y0:y1, gx0:gx1] = AIR
