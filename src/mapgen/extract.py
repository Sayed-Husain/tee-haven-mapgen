"""Extract game-layer tiles from a .map file and convert to a simplified grid."""

import numpy as np
import twmap

# ── DDNet tile ID → simplified category ──────────────────────────
# Full list: https://ddnet.org/explain (tile index reference)
#
#   0 = Air              →  AIR
#   1 = Solid (hookable) →  SOLID
#   2 = Death            →  DEATH
#   3 = Nohook           →  NOHOOK
#   9 = Freeze           →  FREEZE
#  11 = Unfreeze         →  (treat as air — gameplay marker, not obstacle)
#  12 = Deep freeze      →  FREEZE
#  13 = Deep unfreeze    →  (treat as air)
#  21 = Finish           →  ENTITY
#  22 = Start            →  ENTITY
#  33 = Tele-in          →  ENTITY
#  34 = Tele-out         →  ENTITY
#  60 = Through          →  AIR  (pass-through tile)
# 192+ = Entities        →  ENTITY (spawns, weapons, etc.)

AIR    = 0
SOLID  = 1
DEATH  = 2
FREEZE = 3
NOHOOK = 4
ENTITY = 5

CATEGORY_NAMES = {
    AIR:    "air",
    SOLID:  "solid",
    DEATH:  "death",
    FREEZE: "freeze",
    NOHOOK: "nohook",
    ENTITY: "entity",
}

# ASCII characters for rendering each category
ASCII_CHARS = {
    AIR:    ".",
    SOLID:  "#",
    DEATH:  "X",
    FREEZE: "~",
    NOHOOK: "%",
    ENTITY: "!",
}

# Mapping from DDNet tile ID → our simplified category
_TILE_MAP: dict[int, int] = {
    0:  AIR,
    1:  SOLID,
    2:  DEATH,
    3:  NOHOOK,
    9:  FREEZE,
    11: AIR,       # unfreeze → air (gameplay marker)
    12: FREEZE,    # deep freeze
    13: AIR,       # deep unfreeze
    60: AIR,       # through
}


def _classify_tile(tile_id: int) -> int:
    """Map a DDNet tile ID to our simplified category."""
    if tile_id in _TILE_MAP:
        return _TILE_MAP[tile_id]
    if tile_id >= 192:
        return ENTITY
    # Anything else we haven't mapped (tele, switch markers, etc.)
    return ENTITY


def load_game_layer(path: str) -> np.ndarray:
    """Load a .map file and return the game layer as a 2D numpy grid.

    Each cell contains one of our simplified category constants
    (AIR, SOLID, DEATH, FREEZE, NOHOOK, ENTITY).

    Returns:
        np.ndarray of shape (height, width) with dtype uint8.
    """
    m = twmap.Map(path)
    gl = m.game_layer()
    tiles = gl.tiles  # shape: (height, width, 2) → [id, flags]

    # Extract just the tile IDs (ignore flags for now)
    ids = tiles[:, :, 0]

    # Vectorized classification using a lookup table
    # Build a 256-entry LUT (all possible uint8 values)
    lut = np.array([_classify_tile(i) for i in range(256)], dtype=np.uint8)
    grid = lut[ids]

    return grid


def grid_to_ascii(grid: np.ndarray) -> str:
    """Render a 2D grid as ASCII art.

    Each cell is mapped to a single character using ASCII_CHARS.
    """
    rows = []
    for y in range(grid.shape[0]):
        row = "".join(ASCII_CHARS[cell] for cell in grid[y])
        rows.append(row)
    return "\n".join(rows)


def find_start_finish(path: str) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
    """Find the start and finish tile positions from raw map data.

    DDNet tile IDs: 22 = Start, 21 = Finish.

    Returns:
        (start_pos, finish_pos) where each is (x, y) or None if not found.
    """
    m = twmap.Map(path)
    gl = m.game_layer()
    tiles = gl.tiles  # (height, width, 2) → [id, flags]
    ids = tiles[:, :, 0]

    start = None
    finish = None

    # Find first occurrence of each
    start_ys, start_xs = np.where(ids == 22)
    if len(start_xs) > 0:
        start = (int(start_xs[0]), int(start_ys[0]))

    finish_ys, finish_xs = np.where(ids == 21)
    if len(finish_xs) > 0:
        finish = (int(finish_xs[0]), int(finish_ys[0]))

    return start, finish


def print_grid_stats(grid: np.ndarray) -> None:
    """Print a summary of tile type counts."""
    total = grid.size
    for cat in sorted(CATEGORY_NAMES.keys()):
        count = int(np.sum(grid == cat))
        pct = count / total * 100
        char = ASCII_CHARS[cat]
        name = CATEGORY_NAMES[cat]
        print(f"  {char} {name:8s}: {count:6d} ({pct:5.1f}%)")
    print(f"  {'total':10s}: {total:6d}")
