"""DDNet automapper — converts game layer tiles to visual tiles.

Parses DDNet .rules files and applies pattern-matching to generate
visual tile layers. Each rule checks a tile's neighbors and assigns
a visual tile index from the tileset.

The automapper is what makes maps visible to players. Without it,
players only see the physics layer (invisible in-game).

Usage:
    from mapgen.automap import apply_theme
    visual_grid, visual_flags = apply_theme(game_grid, "grass")
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .extract import AIR, SOLID, FREEZE, DEATH, NOHOOK

# Tile flags (same as DDNet)
TILEFLAG_XFLIP = 1
TILEFLAG_YFLIP = 2
TILEFLAG_ROTATE = 8

# Where tileset assets live (relative to project root)
DATA_DIR = Path(__file__).parent.parent.parent / "data"
RULES_DIR = DATA_DIR / "automapper"
TILESET_DIR = DATA_DIR / "tilesets"

# Theme → list of layer specs. Each layer spec is a dict with:
#   rules, tileset, config: automapper rule file, tileset image, config name
#   tile_cat: which game tile category to automap (SOLID, FREEZE, etc.)
#   base_index: starting tile index (must survive rules cleanup step)
#   color: (R, G, B, A) layer tint — needed for white/shape-only tilesets
#
# Tilesets with actual colors (grass, desert, etc.) use default white tint.
# Shape-only tilesets (round_tiles, basic_freeze) need explicit color.
THEME_LAYERS = {
    "grass": [
        {"rules": "grass_main.rules", "tileset": "grass_main.png", "config": "Default",
         "tile_cat": SOLID, "base_index": 1, "color": (255, 255, 255, 255)},
        {"rules": "basic_freeze.rules", "tileset": "basic_freeze.png", "config": "Freeze soft corners",
         "tile_cat": FREEZE, "base_index": 4, "color": (200, 220, 255, 200)},
    ],
    "desert": [
        {"rules": "desert_main.rules", "tileset": "desert_main.png", "config": "Desert",
         "tile_cat": SOLID, "base_index": 1, "color": (255, 255, 255, 255)},
        {"rules": "basic_freeze.rules", "tileset": "basic_freeze.png", "config": "Freeze soft corners",
         "tile_cat": FREEZE, "base_index": 4, "color": (200, 220, 255, 200)},
    ],
    "winter": [
        {"rules": "winter_main.rules", "tileset": "winter_main.png", "config": "Winter",
         "tile_cat": SOLID, "base_index": 1, "color": (255, 255, 255, 255)},
        {"rules": "basic_freeze.rules", "tileset": "basic_freeze.png", "config": "Freeze soft corners",
         "tile_cat": FREEZE, "base_index": 4, "color": (180, 210, 255, 180)},
    ],
    "jungle": [
        {"rules": "jungle_main.rules", "tileset": "jungle_main.png", "config": "Jungle",
         "tile_cat": SOLID, "base_index": 1, "color": (255, 255, 255, 255)},
        {"rules": "basic_freeze.rules", "tileset": "basic_freeze.png", "config": "Freeze soft corners",
         "tile_cat": FREEZE, "base_index": 4, "color": (200, 220, 255, 200)},
    ],
    "round": [
        {"rules": "round_tiles.rules", "tileset": "round_tiles.png", "config": "DDNet",
         "tile_cat": SOLID, "base_index": 1, "color": (60, 50, 40, 255)},
        {"rules": "basic_freeze.rules", "tileset": "basic_freeze.png", "config": "Freeze soft corners",
         "tile_cat": FREEZE, "base_index": 4, "color": (200, 220, 255, 200)},
    ],
    "walls": [
        {"rules": "ddnet_walls.rules", "tileset": "ddnet_walls.png", "config": "Basic Walls",
         "tile_cat": SOLID, "base_index": 1, "color": (80, 70, 60, 255)},
        {"rules": "basic_freeze.rules", "tileset": "basic_freeze.png", "config": "Freeze soft corners",
         "tile_cat": FREEZE, "base_index": 4, "color": (200, 220, 255, 200)},
    ],
    "gores_classic": [
        {"rules": "round_tiles.rules", "tileset": "round_tiles.png", "config": "DDNet",
         "tile_cat": SOLID, "base_index": 1, "color": (30, 30, 30, 255)},
        {"rules": "basic_freeze.rules", "tileset": "basic_freeze.png", "config": "Freeze soft corners",
         "tile_cat": FREEZE, "base_index": 4, "color": (255, 255, 255, 150)},
    ],
}


# ── Rule data structures ─────────────────────────────────────────

@dataclass
class PosCondition:
    """One neighbor condition: check tile at (dx, dy)."""
    dx: int
    dy: int
    mode: str  # "empty", "full", "index", "notindex"
    indices: list[int] = field(default_factory=list)


@dataclass
class IndexRule:
    """One output rule: if all conditions match, place this tile."""
    tile_index: int
    flags: int = 0
    conditions: list[PosCondition] = field(default_factory=list)
    random_prob: float = 1.0  # 1.0 = always apply


@dataclass
class Run:
    """One processing pass — rules applied sequentially."""
    rules: list[IndexRule] = field(default_factory=list)


@dataclass
class Config:
    """One automapper configuration (theme variant)."""
    name: str
    runs: list[Run] = field(default_factory=list)


# ── Parser ────────────────────────────────────────────────────────

def parse_rules_file(path: str | Path) -> dict[str, Config]:
    """Parse a DDNet .rules file into Config objects.

    Returns dict mapping config name → Config.
    """
    path = Path(path)
    configs: dict[str, Config] = {}
    current_config: Config | None = None
    current_run: Run | None = None
    current_rule: IndexRule | None = None

    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # New configuration section
        if line.startswith("[") and line.endswith("]"):
            name = line[1:-1]
            current_config = Config(name=name)
            configs[name] = current_config
            current_run = Run()
            current_config.runs.append(current_run)
            current_rule = None
            continue

        if current_config is None:
            continue

        # New processing pass
        if line == "NewRun":
            current_run = Run()
            current_config.runs.append(current_run)
            current_rule = None
            continue

        # Index rule
        if line.startswith("Index "):
            parts = line.split()
            tile_idx = int(parts[1])
            flags = 0
            for flag in parts[2:]:
                if flag == "XFLIP":
                    flags |= TILEFLAG_XFLIP
                elif flag == "YFLIP":
                    flags |= TILEFLAG_YFLIP
                elif flag == "ROTATE":
                    flags |= TILEFLAG_ROTATE
            current_rule = IndexRule(tile_index=tile_idx, flags=flags)
            current_run.rules.append(current_rule)
            continue

        # Position condition
        if line.startswith("Pos ") and current_rule is not None:
            match = re.match(r"Pos\s+(-?\d+)\s+(-?\d+)\s+(\w+)(.*)", line)
            if not match:
                continue
            dx, dy = int(match.group(1)), int(match.group(2))
            cond_type = match.group(3)
            rest = match.group(4).strip()

            if cond_type == "EMPTY":
                current_rule.conditions.append(
                    PosCondition(dx, dy, "empty")
                )
            elif cond_type == "FULL":
                current_rule.conditions.append(
                    PosCondition(dx, dy, "full")
                )
            elif cond_type in ("INDEX", "NOTINDEX"):
                # Parse index list: "1 OR 2 OR 3"
                indices = []
                for token in rest.replace("OR", " ").split():
                    try:
                        indices.append(int(token))
                    except ValueError:
                        pass
                mode = "index" if cond_type == "INDEX" else "notindex"
                current_rule.conditions.append(
                    PosCondition(dx, dy, mode, indices)
                )
            continue

        # Random probability
        if line.startswith("Random ") and current_rule is not None:
            val = line.split()[1]
            if val.endswith("%"):
                current_rule.random_prob = float(val[:-1]) / 100.0
            else:
                current_rule.random_prob = 1.0 / float(val)
            continue

    return configs


# ── Automapper engine ─────────────────────────────────────────────

def apply_rules(
    source_grid: np.ndarray,
    config: Config,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply automapper rules to produce visual tile indices + flags.

    Two-pass approach:
    1. Source tiles (non-zero): first matching rule wins (standard DDNet)
    2. Adjacent air tiles: rules with 'Pos 0 0 EMPTY' place edge/corner
       decorations on air tiles next to source tiles

    Args:
        source_grid: 2D array where non-zero values are the base tile index
            for tiles to automap. Zero = empty (air).
        config: parsed Config with runs and rules.
        seed: for deterministic random decorations.

    Returns:
        (visual_indices, visual_flags) — both same shape as source_grid.
    """
    h, w = source_grid.shape
    output = np.zeros((h, w), dtype=np.int32)
    flags = np.zeros((h, w), dtype=np.int32)

    # Pre-compute which rules have "Pos 0 0 EMPTY" (apply to air tiles)
    def _rule_checks_self_empty(rule: IndexRule) -> bool:
        return any(c.dx == 0 and c.dy == 0 and c.mode == "empty"
                   for c in rule.conditions)

    # Pre-compute air tiles adjacent to source tiles
    source_mask = source_grid != 0
    adjacent_to_source = np.zeros((h, w), dtype=bool)
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        shifted = np.roll(np.roll(source_mask, dy, axis=0), dx, axis=1)
        adjacent_to_source |= shifted
    adjacent_air = adjacent_to_source & ~source_mask

    for run_idx, run in enumerate(config.runs):
        read_grid = output.copy()

        for y in range(h):
            for x in range(w):
                is_source = source_mask[y, x]
                is_adjacent_air = adjacent_air[y, x]

                if not is_source and not is_adjacent_air:
                    continue

                for rule in run.rules:
                    # Air tiles: only apply rules that check Pos 0 0 EMPTY
                    if not is_source and not _rule_checks_self_empty(rule):
                        continue

                    if _matches_rule(read_grid, source_grid, x, y, w, h,
                                     rule, run_idx):
                        if rule.random_prob < 1.0:
                            if not _passes_random(seed, run_idx, x, y,
                                                   rule.random_prob):
                                continue

                        output[y, x] = rule.tile_index
                        flags[y, x] = rule.flags
                        # No break — last matching rule wins (DDNet behavior)
                        # Base rules (no conditions) set default, then
                        # conditional rules override with edge/corner variants

    return output, flags


def _matches_rule(
    read_grid: np.ndarray,
    source_grid: np.ndarray,
    x: int, y: int,
    w: int, h: int,
    rule: IndexRule,
    run_idx: int,
) -> bool:
    """Check if all conditions of a rule match at position (x, y).

    FULL/EMPTY always check the source grid (does a tile exist?).
    INDEX/NOTINDEX check the output grid (what visual index is assigned?).
    This matches DDNet's automapper behavior.
    """
    for cond in rule.conditions:
        cx = x + cond.dx
        cy = y + cond.dy

        if 0 <= cx < w and 0 <= cy < h:
            # FULL/EMPTY: always check source (tile existence)
            source_val = int(source_grid[cy, cx])
            # INDEX/NOTINDEX: check output from previous runs
            output_val = int(read_grid[cy, cx]) if run_idx > 0 else source_val
        else:
            source_val = -1
            output_val = -1

        if cond.mode == "empty":
            if source_val != 0:
                return False
        elif cond.mode == "full":
            if source_val == 0:
                return False
        elif cond.mode == "index":
            if output_val not in cond.indices:
                return False
        elif cond.mode == "notindex":
            if output_val in cond.indices:
                return False

    return True


def _passes_random(seed: int, run: int, x: int, y: int, prob: float) -> bool:
    """Deterministic random check using position hash."""
    h = hashlib.md5(f"{seed}:{run}:{x}:{y}".encode()).digest()
    val = int.from_bytes(h[:4], "little") / 0xFFFFFFFF
    return val < prob


# ── Public API ────────────────────────────────────────────────────

@dataclass
class VisualLayer:
    """One visual tile layer ready for export."""
    indices: np.ndarray       # tile indices (h, w)
    flags: np.ndarray         # tile flags (h, w)
    tileset_path: str         # path to tileset .png
    color: tuple[int, int, int, int] = (255, 255, 255, 255)  # RGBA tint


def apply_theme(
    game_grid: np.ndarray,
    theme: str = "grass",
) -> list[VisualLayer]:
    """Apply a visual theme to a game layer grid.

    Creates one visual layer per tile category (solid, freeze, etc.).
    Each layer uses its own tileset and automapper rules.

    Args:
        game_grid: 2D array with our tile categories (AIR, SOLID, etc.)
        theme: theme name from THEME_LAYERS

    Returns:
        List of VisualLayer objects (one per tileset).
    """
    if theme not in THEME_LAYERS:
        raise ValueError(f"Unknown theme '{theme}'. Available: {list(THEME_LAYERS)}")

    layers = []
    for layer_spec in THEME_LAYERS[theme]:
        rules_path = RULES_DIR / layer_spec["rules"]
        tileset_path = TILESET_DIR / layer_spec["tileset"]
        config_name = layer_spec["config"]
        tile_cat = layer_spec["tile_cat"]
        base_index = layer_spec["base_index"]
        layer_color = tuple(layer_spec.get("color", (255, 255, 255, 255)))

        if not rules_path.exists() or not tileset_path.exists():
            print(f"  Automapper: skipping {layer_spec['tileset']} (files not found)")
            continue

        configs = parse_rules_file(rules_path)
        if config_name not in configs:
            print(f"  Automapper: config '{config_name}' not found in {layer_spec['rules']}")
            continue

        config = configs[config_name]

        # Build source grid: target tile category → base_index, else → 0
        # base_index must survive the rules cleanup step (e.g., freeze=4
        # is in the NOTINDEX whitelist so Run 1 won't clear it)
        source = np.zeros_like(game_grid, dtype=np.int32)
        source[game_grid == tile_cat] = base_index

        indices, flags = apply_rules(source, config)

        # Only add layer if it has any non-zero tiles
        if np.any(indices > 0):
            layers.append(VisualLayer(
                indices=indices,
                flags=flags,
                tileset_path=str(tileset_path),
                color=layer_color,
            ))

    return layers
