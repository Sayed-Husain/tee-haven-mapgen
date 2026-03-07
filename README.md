# tee-haven-mapgen

AI-assisted Teeworlds Gores map generation tool. Segments Gores maps into gameplay sections using algorithmic path tracing.

## How It Works

The segmentation pipeline imitates a player walking through a Gores map from spawn to finish. As they travel, they pass flat platforms (checkpoints) where they can safely stand. Those checkpoints divide the map into gameplay sections.

### Step 1: Detect Checkpoint Platforms

Scans every row of the grid looking for horizontal runs of tiles where:

- The tile itself is **standable-on** (solid, freeze, or nohook)
- The tile directly **above** is clear air (not freeze — standing in freeze isn't a real checkpoint)

A run must be at least 3 tiles wide. Each qualifying run becomes a checkpoint.

### Step 2: BFS From Spawn

Simulates player reachability using breadth-first search:

1. Starts from spawn tiles (DDNet tile ID 192)
2. Reads the **tele layer** to pair tele-in/tele-out tiles by channel number
3. Floods outward with **8-directional movement** (player can hook/jump in all directions)
4. Passable tiles: air, entity markers, freeze, and nohook
5. When BFS reaches a **tele-in** tile, it teleports to all **tele-out** tiles on the same channel
6. Each visited tile gets a **step number** — distance from spawn in BFS steps

Making freeze and nohook passable is critical — in Gores, players traverse freeze tiles (they get frozen temporarily but it's part of the path). This gives 97%+ map coverage.

### Step 3: Order Checkpoints by BFS Discovery

For each checkpoint platform, checks when BFS first reached the air tile above it. This gives checkpoints in **gameplay order** — the order a player encounters them walking through the map. Unreachable checkpoints are discarded.

### Step 4: Merge Nearby Checkpoints

Adjacent platforms in the same room would create tiny useless segments. Consecutive checkpoints within 50 tiles (Euclidean distance) are merged, keeping only the first.

### Step 5: Build Segments (Voronoi Assignment)

For every BFS-visited tile, asks: **"which checkpoint was the player at most recently when they reached this tile?"**

Uses binary search to find the checkpoint with the largest `bfs_order` <= the tile's step number. That checkpoint "owns" the tile.

This naturally follows the winding path — tiles in a corridor between checkpoint A and checkpoint B all belong to checkpoint A. Each checkpoint's owned tiles get a bounding box, which becomes the segment rectangle.

### Why It Works for Winding Maps

BFS step numbers encode **path distance from spawn**, not spatial position. Two tiles in the same column but at different heights have very different step numbers because the player reaches them at different points in the journey. The Voronoi assignment naturally separates them into different segments — solving the problem of column-based slicing that can't distinguish vertically stacked corridors.

```
Spawn -> BFS flood (freeze-passable, teleporter-aware)
  -> every tile gets a step number
  -> checkpoints ordered by when player reaches them
  -> nearby checkpoints merged
  -> each tile assigned to its most recent checkpoint
  -> bounding boxes -> segments
```

## Setup

Requires Python 3.11+.

```bash
python -m venv .venv
.venv/Scripts/activate  # Windows
pip install -e .
```

## Usage

### Segment a map

```bash
mapgen segment maps/samples/Simpler.map
```

Outputs a text file with ASCII grids per segment and a PNG visualization to `maps/output/`.

Options:
- `-o FILE` — custom output path
- `-d N` — min distance between checkpoints (default: 50, lower = more segments)

### Extract game layer as ASCII

```bash
mapgen extract maps/samples/Simpler.map
```

### Detect horizontal floors

```bash
mapgen floors maps/samples/Simpler.map
```

## Tile Legend

| Char | Tile    | Description                                    |
|------|---------|------------------------------------------------|
| `.`  | Air     | Empty space                                    |
| `#`  | Solid   | Walls and platforms                            |
| `X`  | Death   | Instant kill                                   |
| `~`  | Freeze  | Freezes player temporarily (still traversable) |
| `%`  | Nohook  | Solid surface, hook doesn't attach             |
| `!`  | Entity  | Spawns, teleporters, start/finish lines        |

## Project Structure

```
src/mapgen/
  extract.py    - Load .map files, classify tiles, render ASCII
  pathfind.py   - BFS path tracing, checkpoint detection, Voronoi segmentation
  floors.py     - Horizontal floor detection (solid band dividers)
  segment.py    - Hybrid LLM approach (experimental, not default)
  visualize.py  - PNG renderer with colored segment overlays
  cli.py        - CLI entry point
```
