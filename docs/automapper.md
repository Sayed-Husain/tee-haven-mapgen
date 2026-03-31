# DDNet Automapper Engine

The automapper converts the invisible physics/game layer into visual tile layers that players actually see. Without it, players see their character floating in empty space with invisible walls. The automapper is what makes generated maps look like real game environments.

## How DDNet Tilesets Work

DDNet maps have two types of tile layers:
- **Game layer**: invisible, defines physics (solid, freeze, air, death). This is what the walker generates.
- **Visual tile layers**: visible, reference tileset images (grass_main.png, desert_main.png). Each tile is an index into a tileset grid.

A tileset PNG is a grid of 16x16 tile images (each 64x64 pixels). Index 0 is empty, index 1 is a fill tile, and indices 2+ are edge/corner variants. The automapper decides which visual tile index to assign to each game tile.

## Rules File Format

DDNet ships `.rules` files that define pattern-matching rules. Our engine parses these files and evaluates them against the game grid.

### Structure

```
[ConfigName]          <- Named config section (e.g., [Grass], [Desert])

Index 1               <- Default: assign tile index 1
                         (no conditions = always matches)

NewRun                <- Start a new evaluation pass

Index 2               <- Assign tile index 2 if conditions match
Pos -1 0 FULL         <- Left neighbor must be filled
Pos 1 0 EMPTY         <- Right neighbor must be empty
                         (this is a right-edge tile)

Index 2 XFLIP         <- Same tile, horizontally flipped
Pos -1 0 EMPTY        <- Left neighbor empty
Pos 1 0 FULL          <- Right neighbor filled
                         (this is a left-edge tile)
```

### Rule Components

**Index N [XFLIP] [YFLIP] [ROTATE]**: which visual tile to place, with optional transform flags.

**Pos X Y CONDITION**: check the tile at offset (X, Y) from the current position.

Conditions:
- **FULL**: the tile at this position must exist (be a filled tile in the source)
- **EMPTY**: the tile must not exist (air in the source)
- **INDEX values**: the tile must have one of the listed visual indices (checks output, not source)
- **NOTINDEX values**: the tile must NOT have any of the listed indices

### Random Directive

`#random N` assigns a probability weight. Used for tile variation -- multiple rules with different weights produce visual variety (different fill patterns, subtle edge variations).

## Implementation (automap.py)

### Parsing: parse_rules_file()

Reads `.rules` text files line by line. Builds a dictionary of `AutomapConfig` objects, each containing a list of `Run` objects, each containing a list of `Rule` objects with conditions.

Handles edge cases in the DDNet format:
- Comments and blank lines
- Rules without explicit conditions (always match)
- Multiple configs per file (e.g., [Grass] and [Grass dark])
- Random probability weights

### Evaluation: apply_rules()

Two-pass evaluation per config. This is the core of the automapper.

**Run 1** typically has one unconditional rule: "set all filled tiles to index 1." This creates the base layer.

**Run 2** has dozens of conditional rules that check neighbors to assign edge, corner, and transition tiles. These override the base from Run 1.

#### Critical Semantic Detail

The hardest bug to find was the distinction between source and output checks:

- **FULL/EMPTY** conditions check the **source grid** (does a tile exist in the original game layer?)
- **INDEX/NOTINDEX** conditions check the **output grid** (what visual tile was assigned by previous rules in this run?)

If FULL/EMPTY checked the output instead of the source, edge detection would fail silently -- Run 1 fills everything with index 1, so FULL would always be true.

#### Last Rule Wins

Rules are evaluated in order. When multiple rules match, the last one's tile index is assigned. No early break. This allows general rules to be placed first, with specific edge/corner overrides later.

#### Air Tile Processing

Some rules need to place decorations on air tiles adjacent to solid (e.g., grass tips hanging from the bottom of solid terrain, or snow caps on top). The engine processes air tiles that neighbor source tiles, checking `Pos 0 0 EMPTY` rules.

### Theme System

Each theme defines layers in `THEME_LAYERS`:

```python
THEME_LAYERS = {
    "grass": [
        ("grass_main.rules", "grass_main.png", "Default", SOLID),
        ("basic_freeze.rules", "basic_freeze.png", "Freeze soft corners", FREEZE),
    ],
    "desert": [
        ("desert_main.rules", "desert_main.png", "Desert", SOLID),
        ("basic_freeze.rules", "basic_freeze.png", "Freeze soft corners", FREEZE),
    ],
    # ... 7 themes total
}
```

Each entry specifies:
- Rules file to parse
- Tileset PNG to reference
- Config name within the rules file
- Which game tile type this layer maps (SOLID or FREEZE)

### Color Tinting

Some tilesets are shape-only (white/grey tiles). The engine applies RGBA color tinting per layer so the same tileset can produce different visual styles:

```python
THEME_COLORS = {
    "walls": {"solid": (80, 70, 60, 255)},     # dark brown
    "round": {"solid": (60, 50, 40, 255)},     # darker brown
    "gores_classic": {
        "solid": (15, 12, 10, 255),            # near-black
        "freeze": (220, 220, 220, 120),        # translucent white
    },
}
```

### Background Gradients

Each theme has a `THEME_BACKGROUNDS` entry defining top and bottom RGBA colors. A single quad layer covers the full map area. The DDNet renderer interpolates between corner colors for a smooth vertical gradient.

```python
THEME_BACKGROUNDS = {
    "grass":  {"top": (100, 160, 220, 255), "bottom": (30, 60, 40, 255)},
    "desert": {"top": (200, 160, 100, 255), "bottom": (80, 50, 30, 255)},
    "winter": {"top": (170, 180, 195, 255), "bottom": (50, 55, 70, 255)},
    # ...
}
```

## Output

`apply_theme()` returns a list of `VisualLayer` objects:

```python
@dataclass
class VisualLayer:
    indices: np.ndarray    # 2D array of tile indices
    flags: np.ndarray      # 2D array of transform flags
    tileset_path: str      # path to tileset PNG
    color: tuple[int, ...]  # RGBA tint
```

These are passed to `assemble.py` which creates twmap `Tiles` layers for each visual layer in the final `.map` file.

## Supported Themes

| Theme | Solid Tileset | Character |
|-------|--------------|-----------|
| grass | grass_main | Green terrain, blue sky gradient |
| desert | desert_main | Sandy orange, warm sunset gradient |
| winter | winter_main | Pale grey-blue, overcast gradient |
| jungle | jungle_main | Deep green, forest canopy gradient |
| walls | ddnet_walls (tinted) | Dark brown brick, underground gradient |
| round | round_tiles (tinted) | Rounded edges, dark chocolate gradient |
| gores_classic | round_tiles (near-black) | Iconic Gores look, translucent white freeze |

All themes share the same freeze layer (basic_freeze tileset with soft corner rules), tinted per theme for visual consistency.
