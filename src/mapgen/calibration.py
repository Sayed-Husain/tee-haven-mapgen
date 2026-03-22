"""Mine real Gores map segments for walker calibration data.

Processes the 892 *_segments.txt files produced by the analysis pipeline,
extracting statistical distributions that constrain the walker generator:
- Passage width distributions (how wide are air corridors?)
- Tile type ratios per difficulty level
- Openness ranges
- Segment dimension distributions

These numbers replace hardcoded guesses in the walker, ensuring generated
maps match the physical characteristics of real Gores maps.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

from mapgen.extract import AIR, SOLID, DEATH, FREEZE, NOHOOK, ENTITY
from mapgen.analyze import (
    _ascii_to_grid,
    _tile_composition,
    _iter_segments_from_file,
    _detect_entry_exit,
    _has_tile_type,
)

logger = logging.getLogger(__name__)


# ── Data structures ──────────────────────────────────────────────

@dataclass
class CalibrationProfile:
    """Statistical profile for one difficulty level.

    All ranges are (p25, p75) — the interquartile range — so they
    represent what "typical" looks like, filtering out outliers.
    """
    difficulty: str

    # Tile composition ranges — {tile_name: (p25_pct, p75_pct)}
    tile_ratios: dict[str, tuple[float, float]]

    # How open the segment is (ratio of passable tiles, 0.0–1.0)
    openness_range: tuple[float, float]

    # Passage width distribution — {width_in_tiles: frequency_0_to_1}
    passage_width_dist: dict[int, float]
    median_passage_width: float

    # Segment dimensions — ((w_p25, h_p25), (w_p75, h_p75))
    segment_dimensions: tuple[tuple[int, int], tuple[int, int]]

    # How many segments contributed to this profile
    sample_count: int = 0


# ── Passage width measurement ────────────────────────────────────

def measure_passage_widths(grid: np.ndarray) -> list[int]:
    """Scan rows and columns for contiguous air runs bordered by non-air.

    Uses numpy vectorized operations for speed (47K+ segments).

    Returns all measured passage widths as a flat list.
    """
    h, w = grid.shape
    # Boolean mask: True where passable (AIR or ENTITY)
    passable_mask = (grid == AIR) | (grid == ENTITY)
    widths: list[int] = []

    # Horizontal scan: find runs in each row using diff
    for y in range(h):
        row = passable_mask[y]
        _extract_runs(row, widths)

    # Vertical scan: find runs in each column
    for x in range(w):
        col = passable_mask[:, x]
        _extract_runs(col, widths)

    return widths


def _extract_runs(arr: np.ndarray, widths: list[int]) -> None:
    """Extract run lengths of True values from a 1D boolean array.

    Appends passage widths (runs of True bordered by False/boundary) to widths.
    Uses numpy diff for O(n) performance.
    """
    n = len(arr)
    if n == 0:
        return

    # Pad with False at boundaries to detect runs at edges
    padded = np.empty(n + 2, dtype=bool)
    padded[0] = False
    padded[1:-1] = arr
    padded[-1] = False

    # Find transitions: False→True (start) and True→False (end)
    diff = np.diff(padded.astype(np.int8))
    starts = np.where(diff == 1)[0]   # indices where runs begin
    ends = np.where(diff == -1)[0]    # indices where runs end

    for s, e in zip(starts, ends):
        run_len = e - s
        if run_len > 0:
            widths.append(int(run_len))


# ── Difficulty classification ────────────────────────────────────

def classify_segment_difficulty(
    tile_pcts: dict[str, float],
    has_death: bool,
    has_freeze: bool,
    openness: float,
) -> str:
    """Heuristic difficulty classifier based on measurable properties.

    Real segments don't have difficulty labels, so we infer them:
    - Easy: spacious (openness > 0.45), no death tiles
    - Hard: tight (openness < 0.25), death or heavy freeze (>20%)
    - Medium: everything else

    This is deliberately simple — it's a rough partition for computing
    per-difficulty calibration ranges, not a precise difficulty score.
    """
    freeze_pct = tile_pcts.get("freeze", 0.0)

    if openness > 0.45 and not has_death:
        return "easy"
    elif openness < 0.25 or (has_death and freeze_pct > 15.0):
        return "hard"
    else:
        return "medium"


# ── Profile computation ──────────────────────────────────────────

def _percentile(values: list[float], p: float) -> float:
    """Compute percentile without scipy dependency."""
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = p / 100.0 * (len(sorted_v) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_v) - 1)
    frac = idx - lo
    return sorted_v[lo] * (1 - frac) + sorted_v[hi] * frac


def build_calibration_profiles(
    output_dir: Path | str = Path("maps/output"),
    cache_path: Path | str | None = None,
    max_segments: int = 5000,
) -> dict[str, CalibrationProfile]:
    """Stream through segment files and compute per-difficulty profiles.

    This is the main calibration entry point. It:
    1. Parses segments from *_segments.txt files (up to max_segments)
    2. Converts ASCII → grid, measures passage widths and tile ratios
    3. Classifies difficulty heuristically
    4. Computes percentile distributions per difficulty level
    5. Caches results to JSON for fast reloads

    Args:
        max_segments: Cap on total segments to process. 5000 is more than
            enough for statistically sound distributions while keeping
            runtime under 2 minutes.

    Returns {"easy": CalibrationProfile, "medium": ..., "hard": ...}
    """
    output_dir = Path(output_dir)
    if cache_path is None:
        cache_path = output_dir / "calibration_profiles.json"
    else:
        cache_path = Path(cache_path)

    # Collectors per difficulty
    collectors: dict[str, dict] = {
        diff: {
            "tile_pcts_lists": {name: [] for name in ["air", "solid", "death", "freeze", "nohook", "entity"]},
            "openness_values": [],
            "passage_widths_all": [],
            "widths_list": [],
            "heights_list": [],
            "count": 0,
        }
        for diff in ("easy", "medium", "hard")
    }

    # Find all segment files
    seg_files = sorted(output_dir.glob("*_segments.txt"))
    if not seg_files:
        logger.warning("No segment files found in %s", output_dir)
        return _empty_profiles()

    logger.info("Processing %d segment files...", len(seg_files))
    total_segments = 0

    for f_idx, seg_file in enumerate(seg_files):
        if total_segments >= max_segments:
            logger.info("Reached segment cap (%d), stopping early.", max_segments)
            break

        segments = _iter_segments_from_file(seg_file)

        for source_map, seg_idx, ascii_text in segments:
            if total_segments >= max_segments:
                break
            grid = _ascii_to_grid(ascii_text)
            h, w = grid.shape

            # Skip tiny segments (not representative)
            if w < 10 or h < 10:
                continue

            total_segments += 1

            # Compute metrics
            tile_pcts = _tile_composition(grid)
            air_pct = tile_pcts.get("air", 0.0) + tile_pcts.get("entity", 0.0)
            openness = air_pct / 100.0
            has_death = _has_tile_type(grid, DEATH, min_count=3)
            has_freeze = _has_tile_type(grid, FREEZE, min_count=3)
            passage_widths = measure_passage_widths(grid)

            # Classify difficulty
            diff = classify_segment_difficulty(tile_pcts, has_death, has_freeze, openness)
            coll = collectors[diff]

            # Accumulate
            for name, pct in tile_pcts.items():
                if name in coll["tile_pcts_lists"]:
                    coll["tile_pcts_lists"][name].append(pct)
            coll["openness_values"].append(openness)
            coll["passage_widths_all"].extend(passage_widths)
            coll["widths_list"].append(w)
            coll["heights_list"].append(h)
            coll["count"] += 1

        if (f_idx + 1) % 100 == 0:
            logger.info("  Processed %d / %d files...", f_idx + 1, len(seg_files))

    logger.info(
        "Done. %d segments total: easy=%d, medium=%d, hard=%d",
        total_segments,
        collectors["easy"]["count"],
        collectors["medium"]["count"],
        collectors["hard"]["count"],
    )

    # Build profiles
    profiles: dict[str, CalibrationProfile] = {}

    for diff, coll in collectors.items():
        if coll["count"] == 0:
            profiles[diff] = _default_profile(diff)
            continue

        # Tile ratio IQR
        tile_ratios = {}
        for name, values in coll["tile_pcts_lists"].items():
            if values:
                tile_ratios[name] = (
                    round(_percentile(values, 25), 1),
                    round(_percentile(values, 75), 1),
                )
            else:
                tile_ratios[name] = (0.0, 0.0)

        # Openness IQR
        openness_range = (
            round(_percentile(coll["openness_values"], 25), 3),
            round(_percentile(coll["openness_values"], 75), 3),
        )

        # Passage width distribution (use Counter for O(n) instead of O(n*k))
        pw_all = coll["passage_widths_all"]
        pw_dist: dict[int, float] = {}
        if pw_all:
            total_pw = len(pw_all)
            pw_counts = Counter(pw_all)
            for width, count in pw_counts.items():
                pw_dist[width] = round(count / total_pw, 4)
            # Use numpy for efficient median on large arrays
            median_pw = float(np.median(pw_all))
        else:
            median_pw = 3.0

        # Segment dimensions IQR
        w_list = coll["widths_list"]
        h_list = coll["heights_list"]
        seg_dims = (
            (int(_percentile(w_list, 25)), int(_percentile(h_list, 25))),
            (int(_percentile(w_list, 75)), int(_percentile(h_list, 75))),
        )

        profiles[diff] = CalibrationProfile(
            difficulty=diff,
            tile_ratios=tile_ratios,
            openness_range=openness_range,
            passage_width_dist=dict(sorted(pw_dist.items())),
            median_passage_width=median_pw,
            segment_dimensions=seg_dims,
            sample_count=coll["count"],
        )

    # Cache to JSON
    _save_profiles(profiles, cache_path)
    logger.info("Calibration profiles saved to %s", cache_path)

    return profiles


def _save_profiles(profiles: dict[str, CalibrationProfile], path: Path) -> None:
    """Serialize profiles to JSON."""
    data = {}
    for diff, prof in profiles.items():
        data[diff] = {
            "difficulty": prof.difficulty,
            "tile_ratios": {k: list(v) for k, v in prof.tile_ratios.items()},
            "openness_range": list(prof.openness_range),
            "passage_width_dist": {str(k): v for k, v in prof.passage_width_dist.items()},
            "median_passage_width": prof.median_passage_width,
            "segment_dimensions": [list(d) for d in prof.segment_dimensions],
            "sample_count": prof.sample_count,
        }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_calibration_profiles(
    cache_path: Path | str = Path("maps/output/calibration_profiles.json"),
) -> dict[str, CalibrationProfile] | None:
    """Load cached calibration profiles. Returns None if cache doesn't exist."""
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return None

    raw = json.loads(cache_path.read_text(encoding="utf-8"))
    profiles: dict[str, CalibrationProfile] = {}

    for diff, data in raw.items():
        profiles[diff] = CalibrationProfile(
            difficulty=data["difficulty"],
            tile_ratios={k: tuple(v) for k, v in data["tile_ratios"].items()},
            openness_range=tuple(data["openness_range"]),
            passage_width_dist={int(k): v for k, v in data["passage_width_dist"].items()},
            median_passage_width=data["median_passage_width"],
            segment_dimensions=tuple(tuple(d) for d in data["segment_dimensions"]),
            sample_count=data.get("sample_count", 0),
        )

    return profiles


# ── Defaults / fallbacks ─────────────────────────────────────────

def _empty_profiles() -> dict[str, CalibrationProfile]:
    """Return default profiles when no segment data is available."""
    return {diff: _default_profile(diff) for diff in ("easy", "medium", "hard")}


def _default_profile(difficulty: str) -> CalibrationProfile:
    """Hardcoded fallback profile based on Gores design knowledge.

    Used when no segment files are available for calibration.
    These values are educated guesses — real calibration data is better.
    """
    defaults = {
        "easy": {
            "tile_ratios": {"air": (35.0, 50.0), "solid": (30.0, 45.0),
                            "freeze": (0.0, 10.0), "death": (0.0, 0.0),
                            "nohook": (0.0, 5.0), "entity": (0.0, 1.0)},
            "openness_range": (0.35, 0.50),
            "median_passage_width": 4.0,
            "segment_dimensions": ((30, 20), (60, 50)),
        },
        "medium": {
            "tile_ratios": {"air": (25.0, 40.0), "solid": (35.0, 50.0),
                            "freeze": (5.0, 20.0), "death": (0.0, 5.0),
                            "nohook": (0.0, 10.0), "entity": (0.0, 1.0)},
            "openness_range": (0.25, 0.40),
            "median_passage_width": 3.0,
            "segment_dimensions": ((25, 20), (55, 45)),
        },
        "hard": {
            "tile_ratios": {"air": (15.0, 30.0), "solid": (40.0, 55.0),
                            "freeze": (10.0, 25.0), "death": (0.0, 10.0),
                            "nohook": (0.0, 15.0), "entity": (0.0, 1.0)},
            "openness_range": (0.15, 0.30),
            "median_passage_width": 2.0,
            "segment_dimensions": ((20, 15), (50, 40)),
        },
    }

    d = defaults.get(difficulty, defaults["medium"])
    return CalibrationProfile(
        difficulty=difficulty,
        tile_ratios=d["tile_ratios"],
        openness_range=d["openness_range"],
        passage_width_dist={2: 0.3, 3: 0.35, 4: 0.2, 5: 0.1, 6: 0.05},
        median_passage_width=d["median_passage_width"],
        segment_dimensions=d["segment_dimensions"],
        sample_count=0,
    )


# ── CLI entry point ──────────────────────────────────────────────

def print_calibration_summary(profiles: dict[str, CalibrationProfile]) -> None:
    """Print a human-readable summary of calibration profiles."""
    for diff in ("easy", "medium", "hard"):
        prof = profiles.get(diff)
        if not prof:
            continue

        print(f"\n{'=' * 50}")
        print(f"  {diff.upper()} (N={prof.sample_count} segments)")
        print(f"{'=' * 50}")

        print(f"  Tile ratios (p25 – p75):")
        for name, (lo, hi) in sorted(prof.tile_ratios.items()):
            if hi > 0:
                print(f"    {name:>8s}: {lo:5.1f}% – {hi:5.1f}%")

        print(f"  Openness: {prof.openness_range[0]:.3f} – {prof.openness_range[1]:.3f}")
        print(f"  Median passage width: {prof.median_passage_width:.1f} tiles")

        # Top 5 passage widths
        top_widths = sorted(prof.passage_width_dist.items(), key=lambda x: -x[1])[:5]
        if top_widths:
            print(f"  Passage width distribution (top 5):")
            for w, freq in top_widths:
                bar = "#" * int(freq * 40)
                print(f"    {w:2d} tiles: {freq:5.1%} {bar}")

        dim_lo, dim_hi = prof.segment_dimensions
        print(f"  Segment dimensions: {dim_lo[0]}-{dim_hi[0]} x {dim_lo[1]}-{dim_hi[1]}")
