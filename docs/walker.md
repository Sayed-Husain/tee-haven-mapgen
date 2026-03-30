# Two-Kernel Walker

The walker is the core generation algorithm. It carves playable passages through a fully solid grid by stepping between waypoints, applying two kernels per step to produce air corridors surrounded by freeze borders.

## Why Two Kernels?

In Gores maps, the tile structure follows a strict layering: SOLID (hookable terrain) -> FREEZE (dangerous barrier) -> AIR (playable passage). The freeze layer between solid and air is what creates gameplay challenge -- touching freeze tiles freezes the player.

The two-kernel approach produces this layering by construction:

1. **Outer kernel** (larger): stamps FREEZE wherever the grid is SOLID. Never overwrites AIR.
2. **Inner kernel** (smaller, inside the outer): stamps AIR unconditionally.

Because the outer kernel runs first and is larger than the inner, every air tile is guaranteed to have freeze around it. The freeze border emerges naturally from the size difference between the kernels.

```
Before (all solid):     After outer kernel:     After inner kernel:
################        ################        ################
################        ###~~~~~~~~~~###        ###~~~~~~~~~~###
################        ###~~~~~~~~~~###        ###~........~###
################        ###~~~~~~~~~~###        ###~........~###
################        ###~~~~~~~~~~###        ###~........~###
################        ###~~~~~~~~~~###        ###~~~~~~~~~~###
################        ################        ################

# = SOLID    ~ = FREEZE    . = AIR
```

The inner kernel is 4 tiles wide, the outer adds 2 tiles of margin on each side = 8 tiles total carved area, with 2 tiles of freeze on every side of the air passage.

## Kernel Implementation

```python
@dataclass
class Kernel:
    size: int                  # diameter in tiles
    circularity: float = 0.0  # 0 = square, 1 = circle
```

The kernel determines which tiles to carve at each walker position. The `circularity` parameter interpolates between a square footprint (Chebyshev distance) and a circular footprint (Euclidean distance):

- `circularity = 0`: all tiles within `size/2` Chebyshev distance (square)
- `circularity = 1`: all tiles within `size/2` Euclidean distance (circle)
- Intermediate values: a tile is included if `max(chebyshev, lerp(chebyshev, euclidean, circularity)) <= radius`

This gives smooth control over passage shape. Square kernels produce blocky corridors; circular kernels produce rounded ones.

## Walker Algorithm

### WalkerConfig

```python
@dataclass
class WalkerConfig:
    inner_size: int = 4              # air passage width
    outer_margin: int = 2            # freeze border thickness
    inner_circularity: float = 0.3   # passage shape
    outer_circularity: float = 0.0   # border shape
    momentum_prob: float = 0.6       # direction persistence
    shift_weights: tuple = (0.5, 0.225, 0.2, 0.075)
    size_jitter: float = 0.0         # per-step kernel variation
    circularity_jitter: float = 0.0  # per-step shape variation
    waypoint_jitter: float = 0.0     # waypoint position randomness
```

**`inner_size`**: Passage width in tiles. 3 = claustrophobic (barely fits a tee), 4-5 = standard, 6-8 = open areas. Must be >= 3 for the tee to fit (2 tiles body + 1 tile clearance).

**`outer_margin`**: Extra tiles beyond inner for the freeze border. 1 = thin border, 2 = standard, 3+ = thick freeze gauntlet.

**`momentum_prob`**: Probability of continuing in the same direction. 0.0 = random walk (very winding), 0.5 = moderate curves, 0.9 = nearly straight lines. Controls how "winding" the corridor feels.

**`shift_weights`**: Direction bias relative to the current waypoint target. Four floats for [toward_goal, lateral_1, lateral_2, away_from_goal]. Default heavily favors moving toward the target (0.5) with some lateral movement (0.225 + 0.2) and minimal backtracking (0.075).

### Step Function

Each walker step:

1. **Check waypoint**: if within threshold distance of current target waypoint, advance to next
2. **Mutate kernel**: apply `size_jitter` and `circularity_jitter` for per-step variation
3. **Choose direction**: with probability `momentum_prob`, keep current direction. Otherwise, sample from `shift_weights` relative to target direction
4. **Move**: advance walker position by 1 tile in chosen direction (stay within grid bounds)
5. **Carve**: apply outer kernel (FREEZE on SOLID), then inner kernel (AIR)

### Waypoint System

`generate_segment_grid()` places waypoints along the flow axis:

1. Entry point at top-center (for top-to-bottom flow)
2. N intermediate waypoints distributed vertically with horizontal jitter
3. Exit point at bottom-center

The walker visits each waypoint in order. Waypoint jitter controls how much the path deviates from a straight vertical line -- more jitter means more horizontal wandering.

## Challenge Type Calibration

Different WalkerConfig parameters produce different challenge types:

| Challenge Type | inner_size | outer_margin | momentum | Character |
|---------------|-----------|-------------|----------|-----------|
| Winding freeze corridor | 4 | 2 | 0.5 | Curvy paths through thick freeze |
| Tight solid corridor | 3 | 2 | 0.7 | Narrow, fairly straight |
| Open air zigzag | 6 | 2 | 0.3 | Wide, very winding |
| High air traverse | 8 | 2 | 0.6 | Very open, moderate curves |
| Freeze zigzag | 5 | 3 | 0.4 | Medium width, thick freeze |

These parameters were derived from analyzing real map segment statistics:
- Average passage width in cluster -> `inner_size`
- Freeze-to-air ratio in cluster -> `outer_margin`
- Path directness (BFS distance vs Euclidean) -> `momentum_prob`

## Post-Processing

The walker output needs four post-processing steps:

### 1. Passage Widening
BFS scan finds air tiles with fewer than 3 tiles of clearance in any direction. Widens to 3 by converting adjacent freeze/solid to air. The tee needs 2 tiles of headroom to fit; 3 tiles provides minimal clearance.

### 2. Edge Bug Fixing
DDNet has a physics exploit called "edge bugging" where players can pass through freeze at solid-air corners. The fix: for every air tile adjacent to hookable SOLID with FREEZE on the opposite side, insert a 1-tile FREEZE buffer between air and solid.

### 3. Freeze Blob Removal
Connected component analysis on freeze tiles. Disconnected freeze clusters (not connected to the main structure) are removed -- they serve no gameplay purpose and create visual noise.

### 4. Freeze Border Enforcement
Final pass ensuring the SOLID -> FREEZE -> AIR layering is maintained everywhere. Any solid tile directly touching an air tile gets converted to freeze. This catches edge cases from passage widening which can remove freeze borders.

## Performance

Segment generation is fast: 50-200ms per segment depending on size. A 5-segment map with lobbies generates in under 3 seconds total, including post-processing, validation, assembly, and automapping.

The walker is deterministic given the seed and config. Same inputs = same output, enabling reproducible map generation and snapshot testing.
