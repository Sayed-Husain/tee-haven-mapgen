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

# Theme → (rules file, tileset image, config name)
THEME_MAP = {
    "grass": ("grass_main.rules", "grass_main.png", "Default"),
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

    Args:
        source_grid: 2D array where non-zero = solid (will be automapped).
            Zero = empty (air). This maps our game categories to the
            automapper's FULL/EMPTY distinction.
        config: parsed Config with runs and rules.
        seed: for deterministic random decorations.

    Returns:
        (visual_indices, visual_flags) — both same shape as source_grid.
    """
    h, w = source_grid.shape
    # Start with zeros (empty)
    output = np.zeros((h, w), dtype=np.int32)
    flags = np.zeros((h, w), dtype=np.int32)

    for run_idx, run in enumerate(config.runs):
        # Each run reads from the current output state
        read_grid = output.copy()

        for y in range(h):
            for x in range(w):
                # Only process non-empty tiles (solid in source)
                if source_grid[y, x] == 0:
                    continue

                for rule in run.rules:
                    if _matches_rule(read_grid, source_grid, x, y, w, h,
                                     rule, run_idx):
                        # Random check
                        if rule.random_prob < 1.0:
                            if not _passes_random(seed, run_idx, x, y,
                                                   rule.random_prob):
                                continue

                        output[y, x] = rule.tile_index
                        flags[y, x] = rule.flags
                        break  # first matching rule wins

    return output, flags


def _matches_rule(
    read_grid: np.ndarray,
    source_grid: np.ndarray,
    x: int, y: int,
    w: int, h: int,
    rule: IndexRule,
    run_idx: int,
) -> bool:
    """Check if all conditions of a rule match at position (x, y)."""
    for cond in rule.conditions:
        cx = x + cond.dx
        cy = y + cond.dy

        # Out of bounds → index -1
        if 0 <= cx < w and 0 <= cy < h:
            if run_idx == 0:
                # First run reads from source grid
                check_idx = int(source_grid[cy, cx])
            else:
                # Subsequent runs read from previous output
                check_idx = int(read_grid[cy, cx])
        else:
            check_idx = -1

        if cond.mode == "empty":
            if check_idx != 0:
                return False
        elif cond.mode == "full":
            if check_idx == 0:
                return False
        elif cond.mode == "index":
            if check_idx not in cond.indices:
                return False
        elif cond.mode == "notindex":
            if check_idx in cond.indices:
                return False

    return True


def _passes_random(seed: int, run: int, x: int, y: int, prob: float) -> bool:
    """Deterministic random check using position hash."""
    h = hashlib.md5(f"{seed}:{run}:{x}:{y}".encode()).digest()
    val = int.from_bytes(h[:4], "little") / 0xFFFFFFFF
    return val < prob


# ── Public API ────────────────────────────────────────────────────

def apply_theme(
    game_grid: np.ndarray,
    theme: str = "grass",
) -> tuple[np.ndarray, np.ndarray, str]:
    """Apply a visual theme to a game layer grid.

    Converts game categories (SOLID/AIR/FREEZE/etc.) to a source grid
    for the automapper (non-zero = solid = will get visual tiles),
    then applies the theme's rules.

    Args:
        game_grid: 2D array with our tile categories (AIR, SOLID, etc.)
        theme: theme name from THEME_MAP

    Returns:
        (visual_indices, visual_flags, tileset_path)
    """
    if theme not in THEME_MAP:
        raise ValueError(f"Unknown theme '{theme}'. Available: {list(THEME_MAP)}")

    rules_file, tileset_file, config_name = THEME_MAP[theme]
    rules_path = RULES_DIR / rules_file
    tileset_path = TILESET_DIR / tileset_file

    if not rules_path.exists():
        raise FileNotFoundError(f"Rules file not found: {rules_path}")
    if not tileset_path.exists():
        raise FileNotFoundError(f"Tileset not found: {tileset_path}")

    # Parse rules
    configs = parse_rules_file(rules_path)
    if config_name not in configs:
        raise ValueError(f"Config '{config_name}' not found in {rules_path}. "
                         f"Available: {list(configs)}")

    config = configs[config_name]

    # Convert game grid to automapper source:
    # SOLID → 1 (will be automapped)
    # Everything else → 0 (empty/air)
    source = np.zeros_like(game_grid, dtype=np.int32)
    source[game_grid == SOLID] = 1

    visual_indices, visual_flags = apply_rules(source, config)

    return visual_indices, visual_flags, str(tileset_path)
