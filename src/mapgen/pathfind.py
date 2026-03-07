"""Trace the player path through a Gores map and detect checkpoints.

Instead of asking an LLM to segment the map, we imitate the player:
1. Start from spawn points (or tele-out if spawns are in a lobby).
2. BFS through passable tiles (air + entity) to find the reachable area.
3. When we hit tele-in tiles, jump to tele-out tiles (teleporter links).
4. Along the path, detect checkpoints — flat platforms where a player
   can stand (solid/freeze with air above, at least 3 tiles wide).
5. Record checkpoints in the order BFS discovers them.
6. Segments are the regions between consecutive checkpoints.

This is fully algorithmic — no LLM needed, deterministic, fast.
"""

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from mapgen.extract import AIR, SOLID, FREEZE, NOHOOK, ENTITY


# ── Data types ───────────────────────────────────────────────────────

@dataclass
class Checkpoint:
    """A flat platform where the player can stand."""
    tiles: list[tuple[int, int]]   # list of (x, y) positions of the platform tiles
    y: int                         # row of the platform (the solid/freeze row)
    x_start: int                   # leftmost column
    x_end: int                     # rightmost column (inclusive)
    bfs_order: int = 0             # when BFS first reached this checkpoint

    @property
    def width(self) -> int:
        return self.x_end - self.x_start + 1

    @property
    def center_x(self) -> int:
        return (self.x_start + self.x_end) // 2


@dataclass
class Segment:
    """A rectangular region between two consecutive checkpoints."""
    index: int
    x_start: int
    x_end: int
    y_start: int
    y_end: int
    grid: np.ndarray
    checkpoint: Checkpoint | None = None  # the checkpoint at the START of this segment
    source_map: str = ""


# ── Passability ──────────────────────────────────────────────────────

def _is_passable(tile: int) -> bool:
    """Can a player be in this tile?

    In Gores, players traverse air, entity markers, freeze tiles
    (they get frozen temporarily but it's part of the path), and
    nohook tiles (solid but part of the movement space).
    """
    return tile in (AIR, ENTITY, FREEZE, NOHOOK)


def _is_platform(tile: int) -> bool:
    """Can a player stand ON this tile? (solid, freeze, nohook)"""
    return tile in (SOLID, FREEZE, NOHOOK)


def _is_standable(tile: int) -> bool:
    """Is this tile clear air where a player's body can be?

    For checkpoint detection: the tile ABOVE a platform must be clear
    air/entity — not freeze or nohook (which are obstacles even if
    BFS-passable).
    """
    return tile == AIR or tile == ENTITY


# ── Checkpoint detection ─────────────────────────────────────────────

def detect_checkpoints(grid: np.ndarray, min_width: int = 3) -> list[Checkpoint]:
    """Find all checkpoint platforms in the grid.

    A checkpoint is a horizontal run of solid/freeze/nohook tiles that have
    clear air directly above them — places a player can safely stand and
    reset their double jump.

    The tile above must be AIR or ENTITY (not freeze/nohook), because
    a player standing in freeze isn't at a real checkpoint.

    Args:
        grid: 2D numpy array (height x width).
        min_width: minimum platform width in tiles.

    Returns:
        List of Checkpoint objects (unordered).
    """
    height, width = grid.shape
    checkpoints = []

    for y in range(1, height):  # skip row 0 (nothing above it)
        # Find tiles where: current tile is platform AND tile above is standable air
        run_start = None

        for x in range(width):
            is_cp_tile = _is_platform(grid[y, x]) and _is_standable(grid[y - 1, x])

            if is_cp_tile:
                if run_start is None:
                    run_start = x
            else:
                if run_start is not None and (x - run_start) >= min_width:
                    tiles = [(xi, y) for xi in range(run_start, x)]
                    checkpoints.append(Checkpoint(
                        tiles=tiles,
                        y=y,
                        x_start=run_start,
                        x_end=x - 1,
                    ))
                run_start = None

        # Handle run that extends to the edge
        if run_start is not None and (width - run_start) >= min_width:
            tiles = [(xi, y) for xi in range(run_start, width)]
            checkpoints.append(Checkpoint(
                tiles=tiles,
                y=y,
                x_start=run_start,
                x_end=width - 1,
            ))

    return checkpoints


# ── BFS path tracing ─────────────────────────────────────────────────

def _find_tiles_by_raw_id(path: str, tile_id: int) -> list[tuple[int, int]]:
    """Find all positions of a specific raw DDNet tile ID.

    Returns list of (x, y) tuples.
    """
    import twmap
    m = twmap.Map(path)
    gl = m.game_layer()
    ids = gl.tiles[:, :, 0]
    ys, xs = np.where(ids == tile_id)
    return list(zip(xs.astype(int), ys.astype(int)))


def _read_tele_channels(map_path: str) -> tuple[dict[tuple[int, int], int], dict[int, list[tuple[int, int]]]]:
    """Read tele layer for teleporter channel pairings.

    Returns:
        (tele_in_channels, tele_out_by_channel) where:
        - tele_in_channels: {(x, y): channel_id} for tele-in tiles
        - tele_out_by_channel: {channel_id: [(x, y), ...]} for tele-out tiles
    """
    import twmap
    m = twmap.Map(map_path)
    gl = m.game_layer()
    game_ids = gl.tiles[:, :, 0]

    tl = m.tele_layer()
    if tl is None:
        # Map has no tele layer → no teleporters
        return {}, {}
    tele_tiles = tl.tiles  # (h, w, 2) → [channel, tele_id]

    # Map tele-in positions to their channel
    tele_in_channels: dict[tuple[int, int], int] = {}
    ys, xs = np.where(game_ids == 33)  # tele-in
    for x, y in zip(xs.astype(int), ys.astype(int)):
        ch = int(tele_tiles[y, x, 0])
        if ch > 0:
            tele_in_channels[(x, y)] = ch

    # Group tele-out positions by channel
    tele_out_by_channel: dict[int, list[tuple[int, int]]] = {}
    ys, xs = np.where(game_ids == 34)  # tele-out
    for x, y in zip(xs.astype(int), ys.astype(int)):
        ch = int(tele_tiles[y, x, 0])
        if ch > 0:
            tele_out_by_channel.setdefault(ch, []).append((int(x), int(y)))

    return tele_in_channels, tele_out_by_channel


def trace_path(
    grid: np.ndarray,
    map_path: str,
) -> tuple[dict[tuple[int, int], int], list[tuple[int, int]]]:
    """BFS from spawn through passable tiles with teleporter support.

    Uses the tele layer to correctly pair tele-in → tele-out by channel.
    Supports multiple teleporter events (maps can have several tele links).

    Args:
        grid: classified 2D grid.
        map_path: path to .map file (for raw tile IDs + tele layer).

    Returns:
        (visited, path_order) where:
        - visited: dict mapping (x, y) → BFS step number
        - path_order: list of (x, y) in BFS visit order
    """
    height, width = grid.shape

    # Find spawn positions (raw tile id 192 = spawn)
    spawns = _find_tiles_by_raw_id(map_path, 192)

    # Read tele layer for channel-aware teleporting
    tele_in_channels, tele_out_by_channel = _read_tele_channels(map_path)
    # Track which channels have already been activated
    activated_channels: set[int] = set()

    # BFS starting from spawns
    visited: dict[tuple[int, int], int] = {}
    path_order: list[tuple[int, int]] = []
    queue: deque[tuple[int, int]] = deque()

    for pos in spawns:
        if pos not in visited:
            visited[pos] = 0
            queue.append(pos)
            path_order.append(pos)

    # 8-directional neighbors (player can move in all directions via hook/jump)
    directions = [
        (-1, -1), (0, -1), (1, -1),
        (-1,  0),          (1,  0),
        (-1,  1), (0,  1), (1,  1),
    ]

    while queue:
        x, y = queue.popleft()
        step = visited[(x, y)]

        # Check if we're on a tele-in tile → teleport via channel
        if (x, y) in tele_in_channels:
            ch = tele_in_channels[(x, y)]
            if ch not in activated_channels:
                activated_channels.add(ch)
                for tx, ty in tele_out_by_channel.get(ch, []):
                    if (tx, ty) not in visited and 0 <= tx < width and 0 <= ty < height:
                        visited[(tx, ty)] = step + 1
                        queue.append((tx, ty))
                        path_order.append((tx, ty))

        # Expand to neighbors
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in visited:
                if _is_passable(grid[ny, nx]):
                    visited[(nx, ny)] = step + 1
                    queue.append((nx, ny))
                    path_order.append((nx, ny))

    return visited, path_order


# ── Checkpoint ordering ──────────────────────────────────────────────

def order_checkpoints(
    checkpoints: list[Checkpoint],
    visited: dict[tuple[int, int], int],
) -> list[Checkpoint]:
    """Order checkpoints by when BFS first reaches them.

    For each checkpoint, the BFS step is the minimum step number of
    any AIR tile directly above the platform (where the player stands).

    Checkpoints that BFS never reaches are discarded.
    """
    ordered = []

    for cp in checkpoints:
        # The player stands on the air tile ABOVE the platform
        standing_tiles = [(x, cp.y - 1) for x, _ in cp.tiles]
        steps = [visited[pos] for pos in standing_tiles if pos in visited]

        if steps:
            cp.bfs_order = min(steps)  # keep the actual BFS step number
            ordered.append(cp)

    ordered.sort(key=lambda c: c.bfs_order)
    return ordered


# ── Segment construction ─────────────────────────────────────────────

def _merge_nearby_checkpoints(
    checkpoints: list[Checkpoint],
    min_distance: int = 8,
) -> list[Checkpoint]:
    """Merge checkpoints that are very close together.

    When two consecutive checkpoints (in BFS order) are within
    min_distance tiles of each other (Euclidean), keep only the first.
    This prevents over-segmentation from adjacent platforms.
    """
    if not checkpoints:
        return []

    merged = [checkpoints[0]]
    for cp in checkpoints[1:]:
        prev = merged[-1]
        dx = abs(cp.center_x - prev.center_x)
        dy = abs(cp.y - prev.y)
        dist = (dx**2 + dy**2) ** 0.5

        if dist >= min_distance:
            merged.append(cp)

    return merged


def build_segments(
    grid: np.ndarray,
    checkpoints: list[Checkpoint],
    visited: dict[tuple[int, int], int],
    source_name: str = "",
) -> list[Segment]:
    """Build segments using Voronoi-like checkpoint assignment.

    For each BFS-reachable tile, we assign it to the checkpoint that was
    most recently discovered before it.  In other words: "which checkpoint
    was the player at most recently when they reached this tile?"

    This naturally follows the winding path — tiles in a corridor between
    checkpoint A and checkpoint B all get assigned to checkpoint A.
    """
    height, width = grid.shape

    if not checkpoints:
        return [Segment(
            index=0,
            x_start=0, x_end=width,
            y_start=0, y_end=height,
            grid=grid,
            source_map=source_name,
        )]

    # Sorted checkpoint BFS steps for bisect lookup
    import bisect
    cp_steps = [cp.bfs_order for cp in checkpoints]

    # Assign each tile to its owning checkpoint
    # tile_owners[i] = list of (x, y) positions owned by checkpoint i
    tile_owners: dict[int, list[tuple[int, int]]] = {i: [] for i in range(len(checkpoints))}

    for (x, y), step in visited.items():
        # Find the checkpoint with the largest bfs_order ≤ step
        idx = bisect.bisect_right(cp_steps, step) - 1
        if idx < 0:
            idx = 0  # before first checkpoint → assign to first
        tile_owners[idx].append((x, y))

    # Build segments from owned tile regions
    segments = []
    for i, cp in enumerate(checkpoints):
        tiles = tile_owners[i]
        if not tiles:
            continue

        xs = [t[0] for t in tiles]
        ys = [t[1] for t in tiles]

        # Bounding box with small padding
        x_min = max(0, min(xs) - 1)
        x_max = min(width, max(xs) + 2)
        y_min = max(0, min(ys) - 1)
        y_max = min(height, max(ys) + 2)

        segments.append(Segment(
            index=len(segments),
            x_start=x_min,
            x_end=x_max,
            y_start=y_min,
            y_end=y_max,
            grid=grid[y_min:y_max, x_min:x_max],
            checkpoint=cp,
            source_map=source_name,
        ))

    return segments


# ── Main pipeline ────────────────────────────────────────────────────

def segment_map(
    grid: np.ndarray,
    map_path: str,
    source_name: str = "",
    min_checkpoint_width: int = 3,
    min_checkpoint_distance: int = 50,
) -> list[Segment]:
    """Full algorithmic segmentation pipeline.

    1. Detect all checkpoint platforms.
    2. BFS from spawn through passable tiles (with teleporter handling).
    3. Order checkpoints by BFS discovery.
    4. Merge nearby checkpoints.
    5. Build segments between consecutive checkpoints.

    Args:
        grid: 2D classified grid from extract.load_game_layer().
        map_path: path to .map file (needed for raw tile IDs).
        source_name: map name for metadata.
        min_checkpoint_width: minimum platform width (tiles).
        min_checkpoint_distance: minimum distance between checkpoints.

    Returns:
        List of Segment objects in gameplay order.
    """
    # Step 1: find all checkpoint platforms
    all_checkpoints = detect_checkpoints(grid, min_width=min_checkpoint_width)
    print(f"  Found {len(all_checkpoints)} potential checkpoint platforms")

    # Step 2: BFS from spawn
    visited, path_order = trace_path(grid, map_path)
    print(f"  BFS explored {len(visited)} tiles")

    # Step 3: order checkpoints by BFS discovery
    ordered = order_checkpoints(all_checkpoints, visited)
    print(f"  {len(ordered)} checkpoints reachable by player")

    # Step 4: merge nearby checkpoints to avoid over-segmentation
    merged = _merge_nearby_checkpoints(ordered, min_distance=min_checkpoint_distance)
    print(f"  {len(merged)} checkpoints after merging (min distance={min_checkpoint_distance})")

    # Step 5: build segments
    segments = build_segments(grid, merged, visited, source_name)
    print(f"  {len(segments)} segments created")

    return segments
