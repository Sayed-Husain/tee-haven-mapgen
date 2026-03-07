"""Detect horizontal floors in a Gores map.

Gores maps are structured as winding paths: the player goes right on one
"floor", hits a wall, drops down (or up) to the next floor, and continues.
The map reuses horizontal space across multiple vertical layers, separated
by thick bands of solid/freeze tiles.

This module detects those floor boundaries so we can segment each floor
independently — which is more accurate than slicing the entire map by
columns.
"""

from dataclasses import dataclass

import numpy as np

from mapgen.extract import AIR


@dataclass
class Floor:
    """A horizontal band of playable space in the map."""
    index: int
    y_start: int      # first row of the floor (inclusive)
    y_end: int         # last row of the floor (exclusive)
    grid: np.ndarray   # the sub-grid for this floor (height x full_width)

    @property
    def height(self) -> int:
        return self.y_end - self.y_start


def detect_floors(
    grid: np.ndarray,
    wall_threshold: float = 0.90,
    min_floor_height: int = 5,
) -> list[Floor]:
    """Detect horizontal floors separated by solid divider bands.

    Algorithm:
    1. For each row, compute what fraction of tiles are NOT air.
    2. Rows where that fraction >= wall_threshold are "divider rows"
       (thick walls, ceilings, floors that span the full width).
    3. Groups of consecutive non-divider rows form "floors" — the
       playable corridors where the actual gameplay happens.
    4. Floors shorter than min_floor_height are discarded (noise).

    Args:
        grid: 2D numpy array (height x width) from extract.load_game_layer().
        wall_threshold: fraction of non-air tiles for a row to be a divider.
        min_floor_height: minimum rows for a floor to be kept.

    Returns:
        List of Floor objects, ordered top-to-bottom.
    """
    height, width = grid.shape

    # Fraction of air in each row
    air_fraction = np.mean(grid == AIR, axis=1)

    # A divider row is one where very little is air (almost all wall)
    is_divider = air_fraction < (1.0 - wall_threshold)

    # Walk through rows, grouping consecutive non-divider rows into floors
    floors: list[Floor] = []
    in_floor = False
    floor_start = 0

    for y in range(height):
        if not is_divider[y] and not in_floor:
            # Entering a new floor
            in_floor = True
            floor_start = y
        elif is_divider[y] and in_floor:
            # Leaving a floor
            in_floor = False
            if y - floor_start >= min_floor_height:
                floors.append(Floor(
                    index=len(floors),
                    y_start=floor_start,
                    y_end=y,
                    grid=grid[floor_start:y, :],
                ))

    # Handle case where map doesn't end with a divider
    if in_floor and height - floor_start >= min_floor_height:
        floors.append(Floor(
            index=len(floors),
            y_start=floor_start,
            y_end=height,
            grid=grid[floor_start:height, :],
        ))

    return floors
