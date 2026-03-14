"""Analyze segmented real maps into annotated examples for LLM context.

Reads the *_segments.txt files produced by the segmentation pipeline,
computes tile composition, detects features (freeze corridors, nohook
boxes, death zones), classifies flow direction, and generates natural
language descriptions.

The output is an ExampleLibrary — a curated set of diverse, annotated
segments that teach the LLM what real Gores gameplay looks like.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from mapgen.extract import (
    AIR, SOLID, DEATH, FREEZE, NOHOOK, ENTITY,
    ASCII_CHARS, CATEGORY_NAMES,
)
from mapgen.pathfind import _is_passable


# ── Reverse ASCII mapping: char → tile category ───────────────────

_CHAR_TO_TILE = {ch: cat for cat, ch in ASCII_CHARS.items()}


# ── Data structures ───────────────────────────────────────────────

@dataclass
class SegmentAnalysis:
    """Rich annotation for a single real-map segment."""
    source_map: str
    segment_index: int
    width: int
    height: int

    # Tile composition (percentages)
    tile_pcts: dict[str, float]   # {"air": 42.0, "solid": 28.0, ...}

    # Detected features
    has_freeze: bool
    has_death: bool
    has_nohook: bool

    # Flow characteristics
    primary_flow: str       # "vertical" | "horizontal" | "winding"
    entry_side: str         # "top" | "bottom" | "left" | "right"
    exit_side: str
    path_complexity: str    # "straight" | "L-shaped" | "zigzag"

    # Complexity metrics
    openness: float         # ratio of passable tiles to total (0.0 - 1.0)

    # Natural language description
    description: str

    # The ASCII grid
    ascii_grid: str


@dataclass
class ExampleLibrary:
    """Curated set of annotated examples for LLM prompting."""
    examples: list[SegmentAnalysis]
    generated_at: str           # ISO timestamp
    source_maps: list[str]
    total_segments_analyzed: int


# ── Grid reconstruction from ASCII ───────────────────────────────

def _ascii_to_grid(ascii_text: str) -> np.ndarray:
    """Convert an ASCII grid string back to a numpy tile array."""
    lines = ascii_text.strip().split("\n")
    if not lines:
        return np.array([], dtype=np.uint8)

    h = len(lines)
    w = max(len(line) for line in lines)

    grid = np.full((h, w), SOLID, dtype=np.uint8)
    for y, line in enumerate(lines):
        for x, ch in enumerate(line):
            grid[y, x] = _CHAR_TO_TILE.get(ch, SOLID)

    return grid


# ── Tile composition ─────────────────────────────────────────────

def _tile_composition(grid: np.ndarray) -> dict[str, float]:
    """Compute percentage of each tile type."""
    total = grid.size
    if total == 0:
        return {}

    pcts = {}
    for cat, name in CATEGORY_NAMES.items():
        count = int(np.sum(grid == cat))
        pcts[name] = round(count / total * 100, 1)
    return pcts


# ── Feature detection ────────────────────────────────────────────

def _has_tile_type(grid: np.ndarray, tile_type: int, min_count: int = 3) -> bool:
    """Check if the grid has at least min_count tiles of a given type."""
    return int(np.sum(grid == tile_type)) >= min_count


# ── Entry/exit side detection ────────────────────────────────────

def _detect_border_openings(grid: np.ndarray) -> dict[str, int]:
    """Find which borders have runs of passable tiles (potential openings).

    Returns dict mapping side → length of longest passable run on that border.
    """
    h, w = grid.shape
    openings = {}

    for side, tiles in [
        ("top",    [grid[0, x] for x in range(w)]),
        ("bottom", [grid[h - 1, x] for x in range(w)]),
        ("left",   [grid[y, 0] for y in range(h)]),
        ("right",  [grid[y, w - 1] for y in range(h)]),
    ]:
        # Find longest run of passable tiles
        max_run = 0
        current_run = 0
        for t in tiles:
            if _is_passable(int(t)):
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        openings[side] = max_run

    return openings


def _detect_entry_exit(grid: np.ndarray) -> tuple[str, str]:
    """Determine the most likely entry and exit sides.

    Heuristic: the border with the most passable tiles near the top
    of the grid is the entry; the one near the bottom is the exit.
    For left/right borders, uses the upper half vs lower half.
    """
    openings = _detect_border_openings(grid)

    # Sort by opening size, pick top 2 candidates
    candidates = sorted(openings.items(), key=lambda x: x[1], reverse=True)
    candidates = [(side, size) for side, size in candidates if size >= 2]

    if len(candidates) == 0:
        return "top", "bottom"
    elif len(candidates) == 1:
        entry = candidates[0][0]
        # Exit is the opposite side
        opposite = {"top": "bottom", "bottom": "top",
                    "left": "right", "right": "left"}
        return entry, opposite[entry]
    else:
        # Entry is typically top or left, exit is bottom or right
        # Use position preference: top > left > right > bottom for entry
        entry_pref = {"top": 4, "left": 3, "right": 2, "bottom": 1}
        exit_pref  = {"bottom": 4, "right": 3, "left": 2, "top": 1}

        entry = max(candidates, key=lambda x: entry_pref.get(x[0], 0))[0]
        # Exit: pick the best remaining candidate that isn't the entry
        exit_candidates = [(s, sz) for s, sz in candidates if s != entry]
        if exit_candidates:
            exit_side = max(exit_candidates, key=lambda x: exit_pref.get(x[0], 0))[0]
        else:
            opposite = {"top": "bottom", "bottom": "top",
                        "left": "right", "right": "left"}
            exit_side = opposite[entry]

        return entry, exit_side


# ── Flow classification ──────────────────────────────────────────

def _classify_flow(grid: np.ndarray, entry_side: str, exit_side: str) -> tuple[str, str]:
    """Classify the primary flow direction and path complexity.

    Uses BFS from entry border to understand how the passable space
    is shaped.

    Returns:
        (primary_flow, path_complexity)
        primary_flow: "vertical" | "horizontal" | "winding"
        path_complexity: "straight" | "L-shaped" | "zigzag"
    """
    h, w = grid.shape

    # Determine flow from entry→exit relationship
    vertical_pairs = {("top", "bottom"), ("bottom", "top")}
    horizontal_pairs = {("left", "right"), ("right", "left")}
    corner_pairs = {("top", "left"), ("top", "right"),
                    ("bottom", "left"), ("bottom", "right"),
                    ("left", "top"), ("left", "bottom"),
                    ("right", "top"), ("right", "bottom")}

    if (entry_side, exit_side) in vertical_pairs:
        base_flow = "vertical"
    elif (entry_side, exit_side) in horizontal_pairs:
        base_flow = "horizontal"
    else:
        base_flow = "diagonal"

    # Analyze path shape — vectorized for speed on large grids
    passable_mask = np.isin(grid, [AIR, FREEZE, ENTITY])
    passable_ys, passable_xs = np.where(passable_mask)

    if len(passable_ys) == 0:
        return "vertical", "straight"

    # Compute the centroid of passable space at each row (vectorized)
    row_centroids: dict[int, list[int]] = {}
    for y, x in zip(passable_ys, passable_xs):
        y_int = int(y)
        if y_int not in row_centroids:
            row_centroids[y_int] = []
        row_centroids[y_int].append(int(x))

    if len(row_centroids) < 3:
        return base_flow, "straight"

    # Track how the centroid moves vertically
    sorted_rows = sorted(row_centroids.keys())
    centroids = [np.mean(row_centroids[r]) for r in sorted_rows]

    # Count direction changes in centroid movement
    direction_changes = 0
    for i in range(2, len(centroids)):
        prev_dir = centroids[i - 1] - centroids[i - 2]
        curr_dir = centroids[i] - centroids[i - 1]
        # A direction change: moving left then right (or vice versa)
        if prev_dir * curr_dir < -1.0:  # threshold to avoid noise
            direction_changes += 1

    # Classify complexity
    if direction_changes == 0:
        complexity = "straight"
    elif direction_changes == 1:
        complexity = "L-shaped"
    else:
        complexity = "zigzag"

    # Upgrade flow to "winding" if there are multiple direction changes
    if direction_changes >= 2:
        flow = "winding"
    else:
        flow = base_flow

    return flow, complexity


# ── Description generator ────────────────────────────────────────

def _generate_description(analysis: dict) -> str:
    """Generate a natural language description from computed features.

    Uses a template approach — not LLM-generated, just structured text
    that captures the key characteristics a designer would notice.
    """
    parts = []

    # Size and shape
    w, h = analysis["width"], analysis["height"]
    if w > h * 1.5:
        shape = "wide"
    elif h > w * 1.5:
        shape = "tall"
    else:
        shape = "square"
    parts.append(f"A {w}x{h} {shape} segment from {analysis['source_map']}.")

    # Flow
    flow = analysis["primary_flow"]
    complexity = analysis["path_complexity"]
    entry = analysis["entry_side"]
    exit_side = analysis["exit_side"]
    parts.append(f"{complexity.capitalize()} {flow} flow from {entry} to {exit_side}.")

    # Features
    features = []
    pcts = analysis["tile_pcts"]
    if analysis["has_freeze"] and pcts.get("freeze", 0) > 5:
        features.append(f"freeze corridors ({pcts['freeze']}% freeze)")
    if analysis["has_nohook"] and pcts.get("nohook", 0) > 1:
        features.append(f"nohook sections ({pcts['nohook']}% nohook)")
    if analysis["has_death"] and pcts.get("death", 0) > 0.5:
        features.append(f"death zones ({pcts['death']}% death)")

    if features:
        parts.append("Features: " + ", ".join(features) + ".")

    # Openness
    openness = analysis["openness"]
    if openness > 0.5:
        parts.append(f"Open layout ({openness:.0%} passable) with spacious chambers.")
    elif openness > 0.3:
        parts.append(f"Moderate density ({openness:.0%} passable) with defined corridors.")
    else:
        parts.append(f"Dense layout ({openness:.0%} passable) with tight passages.")

    return " ".join(parts)


# ── Main analysis function ───────────────────────────────────────

def analyze_segment(
    grid: np.ndarray,
    source_map: str,
    index: int,
    ascii_grid: str | None = None,
) -> SegmentAnalysis:
    """Compute features and generate description for one segment.

    Args:
        grid: 2D numpy tile array for the segment.
        source_map: Name of the source map (e.g. "Simpler").
        index: Segment index within that map.
        ascii_grid: Pre-computed ASCII text. If None, generated from grid.

    Returns:
        SegmentAnalysis with all computed features and description.
    """
    h, w = grid.shape
    tile_pcts = _tile_composition(grid)
    has_freeze = _has_tile_type(grid, FREEZE)
    has_death = _has_tile_type(grid, DEATH)
    has_nohook = _has_tile_type(grid, NOHOOK)

    entry_side, exit_side = _detect_entry_exit(grid)
    primary_flow, path_complexity = _classify_flow(grid, entry_side, exit_side)

    # Openness: fraction of passable tiles (vectorized for speed)
    passable_mask = np.isin(grid, [AIR, FREEZE, ENTITY])
    openness = float(passable_mask.sum()) / grid.size if grid.size > 0 else 0.0

    # Build ASCII grid string (skip if already provided)
    if ascii_grid is None:
        from mapgen.extract import grid_to_ascii
        ascii_grid = grid_to_ascii(grid)

    # Generate description
    analysis_dict = {
        "source_map": source_map,
        "width": w, "height": h,
        "tile_pcts": tile_pcts,
        "has_freeze": has_freeze,
        "has_death": has_death,
        "has_nohook": has_nohook,
        "primary_flow": primary_flow,
        "entry_side": entry_side,
        "exit_side": exit_side,
        "path_complexity": path_complexity,
        "openness": openness,
    }
    description = _generate_description(analysis_dict)

    return SegmentAnalysis(
        source_map=source_map,
        segment_index=index,
        width=w,
        height=h,
        tile_pcts=tile_pcts,
        has_freeze=has_freeze,
        has_death=has_death,
        has_nohook=has_nohook,
        primary_flow=primary_flow,
        entry_side=entry_side,
        exit_side=exit_side,
        path_complexity=path_complexity,
        openness=round(openness, 3),
        description=description,
        ascii_grid=ascii_grid,
    )


# ── Segment file iteration (streaming, low-memory) ───────────────

def _iter_segments_from_file(
    txt_file: Path,
) -> list[tuple[str, int, str]]:
    """Parse one *_segments.txt file into (source_map, index, ascii_text) tuples.

    Returns only lightweight data — no numpy arrays — so we can process
    thousands of files without blowing up memory.
    """
    source_map = txt_file.stem.replace("_segments", "")
    content = txt_file.read_text(encoding="utf-8")
    parts = content.split("=" * 60)
    results: list[tuple[str, int, str]] = []

    for i in range(1, len(parts) - 1, 2):
        header = parts[i].strip()
        grid_text = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if not grid_text:
            continue

        idx_match = re.search(r"Segment\s+(\d+):", header)
        if not idx_match:
            continue
        seg_idx = int(idx_match.group(1))
        results.append((source_map, seg_idx, grid_text))

    return results


# ── Example selection ────────────────────────────────────────────

def _select_diverse_examples(
    analyses: list[SegmentAnalysis],
    target_count: int = 5,
) -> list[SegmentAnalysis]:
    """Select a diverse set of representative segments.

    Strategy:
    1. Filter: remove tiny segments (< 20x15) and oversized ones (> 100 wide)
       — the LLM can't process 900-character-wide ASCII
    2. Enforce ONE segment per source map for maximum diversity
    3. Cover all three flow types (vertical, horizontal, winding)
    4. Cover different path complexities (straight, L-shaped, zigzag)
    5. Cover different openness ranges (tight, moderate, open)
    6. Prefer segments with interesting features (freeze, death, nohook)
    """
    # Filter: reasonable size for LLM context (not too tiny, not too huge)
    candidates = [a for a in analyses
                  if a.width >= 20 and a.height >= 15
                  and a.width <= 100 and a.height <= 100]

    if len(candidates) <= target_count:
        return candidates

    def _score(a: SegmentAnalysis) -> tuple:
        """Higher = more interesting as an example."""
        return (
            a.has_freeze + a.has_nohook + a.has_death,  # feature count
            a.path_complexity == "zigzag",                # prefer complex
            a.path_complexity == "L-shaped",
            -abs(a.openness - 0.4),                       # prefer moderate openness
        )

    selected: list[SegmentAnalysis] = []
    used_maps: set[str] = set()

    # Pass 1: one segment per flow type, each from a different map
    for flow_type in ["vertical", "horizontal", "winding"]:
        if len(selected) >= target_count:
            break
        flow_cands = [a for a in candidates
                      if a.primary_flow == flow_type and a.source_map not in used_maps]
        if flow_cands:
            best = max(flow_cands, key=_score)
            selected.append(best)
            used_maps.add(best.source_map)

    # Pass 2: cover missing complexity types
    covered_complexity = {a.path_complexity for a in selected}
    for complexity in ["zigzag", "L-shaped", "straight"]:
        if len(selected) >= target_count:
            break
        if complexity in covered_complexity:
            continue
        comp_cands = [a for a in candidates
                      if a.path_complexity == complexity and a.source_map not in used_maps]
        if comp_cands:
            best = max(comp_cands, key=_score)
            selected.append(best)
            used_maps.add(best.source_map)

    # Pass 3: cover openness diversity (tight, moderate, open)
    openness_ranges = [
        ("tight", 0.0, 0.3),
        ("moderate", 0.3, 0.5),
    ]
    for label, low, high in openness_ranges:
        if len(selected) >= target_count:
            break
        # Only add if we don't already have one in this range
        has_range = any(low <= ex.openness < high for ex in selected)
        if has_range:
            continue
        range_cands = [a for a in candidates
                       if low <= a.openness < high and a.source_map not in used_maps]
        if range_cands:
            best = max(range_cands, key=_score)
            selected.append(best)
            used_maps.add(best.source_map)

    # Pass 4: fill remaining slots with best unused segments from new maps
    if len(selected) < target_count:
        remaining = [a for a in candidates
                     if a not in selected and a.source_map not in used_maps]
        remaining.sort(key=_score, reverse=True)
        for a in remaining:
            if len(selected) >= target_count:
                break
            selected.append(a)
            used_maps.add(a.source_map)

    return selected[:target_count]


# ── Library building and caching ─────────────────────────────────

def build_example_library(
    output_dir: Path = Path("maps/output"),
    target_count: int = 5,
    cache_path: Path | None = None,
) -> ExampleLibrary:
    """Analyze all segments and select a diverse representative set.

    Uses a streaming two-pass approach to handle large datasets (30k+
    segments) without blowing up memory:

    Pass 1: Parse each file, convert ASCII to grid, analyze, but only
    keep the lightweight SegmentAnalysis (with ascii_grid='' placeholder).
    This avoids storing 31k grids + ASCII strings in memory.

    Pass 2: After selecting the best examples, re-read only the files
    that contain selected segments to fill in their ASCII grids.

    Args:
        output_dir: Directory containing *_segments.txt files.
        target_count: How many examples to select.
        cache_path: Where to save the JSON cache.

    Returns:
        ExampleLibrary with curated, annotated examples.
    """
    import time

    t0 = time.time()

    if cache_path is None:
        cache_path = output_dir / "example_library.json"

    files = sorted(output_dir.glob("*_segments.txt"))
    total_files = len(files)
    print(f"Found {total_files} segment files in {output_dir}")

    # ── Pass 1: streaming analysis (low-memory) ──────────────────
    # Parse each file, analyze segments, keep only metadata
    # ASCII grids are set to "" — we'll fill them in Pass 2
    analyses: list[SegmentAnalysis] = []
    source_maps: set[str] = set()
    total_segments = 0

    for file_idx, txt_file in enumerate(files):
        file_segments = _iter_segments_from_file(txt_file)

        for source_map, seg_idx, ascii_text in file_segments:
            source_maps.add(source_map)

            # Convert ASCII to grid for analysis
            grid = _ascii_to_grid(ascii_text)
            if grid.size == 0:
                continue

            # Analyze but DON'T store the ASCII grid (save memory)
            analysis = analyze_segment(
                grid, source_map, seg_idx,
                ascii_grid="",  # placeholder — filled in Pass 2
            )
            analyses.append(analysis)
            total_segments += 1

            # Let the grid be garbage collected
            del grid

        # Progress logging every 100 files
        if (file_idx + 1) % 100 == 0 or (file_idx + 1) == total_files:
            elapsed = time.time() - t0
            print(f"  [{file_idx + 1}/{total_files}] "
                  f"{total_segments} segments analyzed "
                  f"({elapsed:.0f}s)")

    print(f"\nPass 1 complete: {total_segments} segments from "
          f"{len(source_maps)} maps in {time.time() - t0:.1f}s")

    # ── Select diverse representatives ───────────────────────────
    selected = _select_diverse_examples(analyses, target_count)
    print(f"Selected {len(selected)} representative examples")

    # Free the big list — we only need the selected examples now
    del analyses

    # ── Pass 2: fill ASCII grids for selected examples ───────────
    # Group selected by source_map so we only re-read needed files
    needed: dict[str, list[int]] = {}  # source_map → [segment_indices]
    for ex in selected:
        needed.setdefault(ex.source_map, []).append(ex.segment_index)

    print(f"Pass 2: re-reading {len(needed)} files for ASCII grids...")

    for txt_file in files:
        source_map = txt_file.stem.replace("_segments", "")
        if source_map not in needed:
            continue

        target_indices = set(needed[source_map])
        file_segments = _iter_segments_from_file(txt_file)

        for src, seg_idx, ascii_text in file_segments:
            if seg_idx in target_indices:
                # Find the matching SegmentAnalysis and fill in its grid
                for ex in selected:
                    if ex.source_map == source_map and ex.segment_index == seg_idx:
                        ex.ascii_grid = ascii_text
                        break

    # Verify all selected examples have their ASCII
    for ex in selected:
        if not ex.ascii_grid:
            print(f"  WARNING: Could not find ASCII for "
                  f"{ex.source_map} seg {ex.segment_index}")

    library = ExampleLibrary(
        examples=selected,
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        source_maps=sorted(source_maps),
        total_segments_analyzed=total_segments,
    )

    # Save cache
    _save_library(library, cache_path)

    elapsed = time.time() - t0
    print(f"\nLibrary built in {elapsed:.1f}s")

    return library


def _save_library(library: ExampleLibrary, path: Path) -> None:
    """Save the example library to a JSON cache file."""
    data = {
        "generated_at": library.generated_at,
        "source_maps": library.source_maps,
        "total_segments_analyzed": library.total_segments_analyzed,
        "examples": [asdict(ex) for ex in library.examples],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_example_library(
    cache_path: Path = Path("maps/output/example_library.json"),
) -> ExampleLibrary | None:
    """Load cached library. Returns None if cache doesn't exist."""
    if not cache_path.exists():
        return None

    data = json.loads(cache_path.read_text(encoding="utf-8"))

    examples = []
    for ex in data.get("examples", []):
        examples.append(SegmentAnalysis(**ex))

    return ExampleLibrary(
        examples=examples,
        generated_at=data.get("generated_at", ""),
        source_maps=data.get("source_maps", []),
        total_segments_analyzed=data.get("total_segments_analyzed", 0),
    )
