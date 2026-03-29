# Architecture

The system has three pipelines: analysis (one-time), calibration (bridging), and generation (per map).

## Data Flow

```
Real Gores Maps (892)
    |
    v
[extract.py] -> tile grids (6 categories)
    |
    v
[pathfind.py] -> BFS step numbers + checkpoints + Voronoi segments
    |
    v
[analyze.py] -> per-segment statistics (tile ratios, openness, flow)
    |
    v
[cluster.py] -> PatternLibrary (20 challenge types)
    |
    v
[config_mapping.py] -> cluster label -> WalkerConfig
    |
    v
[graph.py]  LangGraph state machine
    |
    +---> [llm.py] GPT-4o selects challenge sequence + theme
    |         |
    |         v
    +---> [walker.py] carves each segment (two-kernel)
    |         |
    |         v
    +---> [postprocess.py] widen, edge bugs, freeze borders
    |         |
    |         v
    +---> [validate.py] BFS connectivity check
    |         |
    |         v (fail? retry with new seed)
    |         |
    +---> [assemble.py] stitch segments + lobbies + arrows + background
    |         |
    |         v
    +---> [automap.py] DDNet rules -> visual tile layers
    |         |
    |         v
    +---> [export] write .map file
```

## Analysis Pipeline

Runs once over a corpus of real Gores maps. Purpose: extract structural data that the generator can learn from.

### extract.py
Loads `.map` files via the twmap library. Classifies every tile into 6 categories:

| ID | Name | Char | DDNet Tile IDs |
|----|------|------|----------------|
| 0 | AIR | `.` | 0 |
| 1 | SOLID | `#` | 1 (hookable) |
| 2 | DEATH | `X` | 10 |
| 3 | FREEZE | `~` | 9 |
| 4 | NOHOOK | `%` | 3 (non-hookable solid) |
| 5 | ENTITY | `!` | spawns, teleporters, start/finish |

Renders ASCII representations for inspection and debugging.

### pathfind.py
BFS from spawn with 8-directional movement. Key details:
- **Passable tiles**: air, freeze, nohook, entity (NOT solid, NOT death)
- **Teleporter awareness**: reads the tele layer, pairs tele-in/tele-out by channel number. When BFS reaches a tele-in, it jumps to all tele-outs on the same channel
- **Step numbers**: each visited tile gets its BFS distance from spawn
- **Checkpoint detection**: horizontal runs of standable tiles (solid/freeze/nohook) with air directly above, minimum 3 tiles wide
- **Checkpoint ordering**: sorted by when BFS first reached the air above them (gameplay order, not spatial order)
- **Checkpoint merging**: consecutive checkpoints within 50 tiles (Euclidean) are merged
- **Voronoi assignment**: each tile is assigned to the checkpoint with the largest `bfs_order <= tile_step`. Binary search for efficiency. Bounding boxes around owned tiles become segment rectangles.

### analyze.py
Per-segment statistics: tile composition percentages, openness score (air ratio), flow direction (which sides have openings), passage width distributions.

### cluster.py
Groups the 47K+ segments into ~20 challenge types based on structural similarity. Each cluster gets a human-readable label (e.g., "Winding freeze corridor descent") and a description. Outputs a `PatternLibrary` with `Cluster` objects.

## Calibration Pipeline

Bridges analysis and generation. Maps each cluster to walker parameters.

### calibration.py
Extracts parameter distributions from segments within each cluster:
- Average passage width -> `inner_size`
- Average freeze ratio -> `outer_margin`
- Directional bias -> `momentum_prob`
- Dimensions -> segment `width` and `height`

### config_mapping.py
Defines `get_walker_config(cluster_label)` which returns a `WalkerConfig` dataclass. Currently hand-tuned from calibration statistics. Each cluster label maps to specific walker parameters that reproduce its characteristic geometry.

## Generation Pipeline

Orchestrated by LangGraph in `graph.py`. Produces one `.map` file per run.

### State Schema

```python
class PipelineState(TypedDict, total=False):
    # User inputs
    user_difficulty: str           # "easy" | "medium" | "hard"
    user_n_segments: int           # number of challenge segments
    user_theme: Optional[str]      # visual theme name

    # LLM plan
    challenge_sequence: list[str]  # ordered cluster labels
    difficulty_progression: list[str]
    visual_theme: str

    # Segment processing
    segments: list[dict]           # per-segment config + grid + validation
    current_segment_index: int

    # Assembly + export
    assembled_grid: np.ndarray     # full stitched grid
    entities: list[tuple[int,int,int]]  # (x, y, tile_id)
    visual_layers: list            # VisualLayer objects from automapper
    output_path: Optional[str]

    # Retry limits
    max_seed_retries: int          # per-segment seed retries (default 3)
    max_param_retries: int         # parameter adjustment retries
```

### Node Functions

**llm_plan_node**: Loads the cluster vocabulary from the PatternLibrary. Sends it to GPT-4o with the user's difficulty and segment count request. Parses the structured JSON response containing `challenge_sequence`, `difficulty_progression`, and `visual_theme`. Validates that all cluster labels exist (fuzzy-matches unknown labels to closest known). If `challenge_sequence` is already provided (testing mode), this node is a no-op.

**init_segments_node**: For each challenge in the sequence, looks up the WalkerConfig via `get_walker_config()`, computes segment dimensions via `get_segment_dimensions()`, assigns a random seed, and creates the segment dict.

**walker_node**: For each segment, calls `generate_segment_grid()` which creates a fully-solid grid, places waypoints along the flow axis with jitter, and runs the Walker. Returns the carved grid.

**validate_segment_node**: Runs BFS from entry to exit. If reachability < 100%, retries with a new seed. After 3 failures, could escalate to parameter adjustment (not yet implemented).

**assemble_node**: Calls `stitch_segments()` to stack segment grids vertically. Then:
- `build_spawn_segment()`: walker-carved lobby with solid platforms, spawn entities, no freeze
- `build_finish_segment()`: walker-carved lobby with finish entities
- Places start line tiles at the corridor entrance below spawn
- Adds BFS-pathfinded arrow quads (arrows point toward the finish)
- Adds gradient background quad (theme-specific colors)

**automap_node**: Calls `apply_theme()` which parses DDNet `.rules` files and generates visual tile layers for both the solid tileset and freeze tileset.

**export_node**: Calls `write_map()` which creates the twmap Map object with all layers (game, visual, arrows, markers, background) and saves the `.map` file.

### Key Design Decision

The LLM makes ONE creative call at the start of the pipeline. Everything downstream is deterministic given the seed. This means:
- Same challenge sequence + same seeds = identical map (reproducible)
- Failed validation only needs to change the seed, not re-call the LLM
- The LLM stays in its strength zone (creative sequencing) and never touches spatial data

## Module Dependencies

```
graph.py
  |- llm.py (OpenAI API)
  |- config_mapping.py
  |- walker.py
  |- postprocess.py
  |- validate.py (uses bfs.py)
  |- assemble.py (uses walker.py for lobbies)
  |- automap.py

assemble.py
  |- walker.py (lobby generation)
  |- bfs.py (arrow pathfinding)

automap.py
  |- (standalone, reads .rules files)

cluster.py
  |- analyze.py
  |- extract.py
  |- pathfind.py
```
