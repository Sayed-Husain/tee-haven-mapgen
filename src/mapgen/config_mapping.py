"""Map cluster challenge types to walker parameters.

Each of the 20 clusters from pattern_library.json represents a distinct
Gores gameplay pattern. This module maps those patterns to WalkerConfig
parameters that reproduce similar geometry.

The mapping is manually calibrated based on cluster centroid statistics
(air%, solid%, freeze%, openness, flow direction) and playtesting.
Future improvement: auto-calibrate from real segment analysis data.
"""

from __future__ import annotations

from .walker import WalkerConfig


# ── Cluster → WalkerConfig mapping ──────────────────────────────
#
# Cluster centroids (from pattern_library.json analysis):
#
# High air (60-76%): open sections, large inner kernel, thin freeze
# Medium air (40-50%): standard corridors, medium kernel
# Low air (20-30%): tight passages, small kernel, thick freeze
#
# Key parameters and what they control:
#   inner_size: passage width (bigger = more air, easier)
#   outer_margin: freeze border thickness (bigger = more freeze, harder)
#   momentum_prob: direction predictability (higher = straighter paths)
#   size_range: kernel mutation range (narrower = more uniform corridors)

# Default config that produces good playable maps (from playtesting)
DEFAULT_CONFIG = {
    "inner_size": 6,
    "outer_margin": 1,
    "inner_circularity": 0.3,
    "momentum_prob": 0.65,
    "size_range": (4, 7),
    "size_mutate_prob": 0.03,
    "fade_min_size": 4,
    "width": 100,
    "height": 100,
    "n_waypoints": 5,
}


CLUSTER_TO_CONFIG: dict[str, dict] = {
    # ── High air clusters (open sections) ────────────────────────
    "high air zigzag traverse": {
        **DEFAULT_CONFIG,
        "inner_size": 7,
        "outer_margin": 1,
        "momentum_prob": 0.5,  # more winding
        "size_range": (5, 8),
    },
    "open air freeze zigzag descent": {
        **DEFAULT_CONFIG,
        "inner_size": 7,
        "outer_margin": 1,
        "momentum_prob": 0.55,
        "size_range": (5, 8),
    },
    "open air zigzag descent": {
        **DEFAULT_CONFIG,
        "inner_size": 7,
        "outer_margin": 1,
        "momentum_prob": 0.5,
        "size_range": (5, 9),
    },
    "Winding Air Ascent Traverse": {
        **DEFAULT_CONFIG,
        "inner_size": 7,
        "outer_margin": 1,
        "momentum_prob": 0.55,
        "size_range": (5, 8),
    },

    # ── Medium air clusters (standard corridors) ─────────────────
    "Winding freeze corridor ascent": {
        **DEFAULT_CONFIG,
        "inner_size": 5,
        "outer_margin": 2,
        "momentum_prob": 0.6,
        "size_range": (4, 6),
    },
    "Winding freeze corridor descent": {
        **DEFAULT_CONFIG,
        "inner_size": 5,
        "outer_margin": 2,
        "momentum_prob": 0.6,
        "size_range": (4, 6),
    },
    "zigzag freeze corridor traverse": {
        **DEFAULT_CONFIG,
        "inner_size": 5,
        "outer_margin": 2,
        "momentum_prob": 0.55,
        "size_range": (4, 6),
    },
    "vertical solid freeze corridor": {
        **DEFAULT_CONFIG,
        "inner_size": 5,
        "outer_margin": 1,
        "momentum_prob": 0.75,  # straighter (vertical)
        "size_range": (4, 6),
    },
    "nohook zigzag descent": {
        **DEFAULT_CONFIG,
        "inner_size": 5,
        "outer_margin": 2,
        "momentum_prob": 0.55,
        "size_range": (4, 6),
    },
    "solid corridor freeze traverse": {
        **DEFAULT_CONFIG,
        "inner_size": 5,
        "outer_margin": 1,
        "momentum_prob": 0.7,
        "size_range": (4, 6),
    },

    # ── Low air clusters (tight passages) ────────────────────────
    "tight solid corridor traversal": {
        **DEFAULT_CONFIG,
        "inner_size": 4,
        "outer_margin": 1,
        "momentum_prob": 0.65,
        "size_range": (3, 5),
    },
    "dense solid zigzag corridor": {
        **DEFAULT_CONFIG,
        "inner_size": 4,
        "outer_margin": 1,
        "momentum_prob": 0.55,
        "size_range": (3, 5),
    },
    "tight freeze corridor ascent": {
        **DEFAULT_CONFIG,
        "inner_size": 4,
        "outer_margin": 2,
        "momentum_prob": 0.65,
        "size_range": (3, 5),
    },
    "tight freeze corridor traversal": {
        **DEFAULT_CONFIG,
        "inner_size": 4,
        "outer_margin": 2,
        "momentum_prob": 0.7,
        "size_range": (3, 5),
    },
    "tight solid corridor with freeze": {
        **DEFAULT_CONFIG,
        "inner_size": 4,
        "outer_margin": 2,
        "momentum_prob": 0.65,
        "size_range": (3, 5),
    },

    # ── Freeze-heavy clusters ────────────────────────────────────
    "narrow freeze corridor descent": {
        **DEFAULT_CONFIG,
        "inner_size": 4,
        "outer_margin": 3,
        "momentum_prob": 0.7,
        "size_range": (3, 5),
    },
    "dense freeze corridor traverse": {
        **DEFAULT_CONFIG,
        "inner_size": 5,
        "outer_margin": 2,
        "momentum_prob": 0.65,
        "size_range": (4, 6),
    },

    # ── Special clusters ─────────────────────────────────────────
    "Winding Air Chamber with Death": {
        **DEFAULT_CONFIG,
        "inner_size": 6,
        "outer_margin": 1,
        "momentum_prob": 0.5,
        "size_range": (5, 7),
        # TODO: add death tile placement post-processing
    },
    "punishing nohook zigzag descent": {
        **DEFAULT_CONFIG,
        "inner_size": 4,
        "outer_margin": 1,
        "momentum_prob": 0.5,
        "size_range": (3, 5),
        # TODO: add nohook tile conversion post-processing
    },
    "Winding freeze corridor with death": {
        **DEFAULT_CONFIG,
        "inner_size": 5,
        "outer_margin": 2,
        "momentum_prob": 0.6,
        "size_range": (4, 6),
        # TODO: add death tile placement post-processing
    },
}


def get_walker_config(cluster_label: str, difficulty: str = "medium") -> WalkerConfig:
    """Get WalkerConfig for a cluster type with difficulty adjustment.

    Difficulty modifiers:
        easy:   +1 inner_size, -1 outer_margin (wider, less freeze)
        medium: no change
        hard:   -1 inner_size, +1 outer_margin (tighter, more freeze)
    """
    raw = CLUSTER_TO_CONFIG.get(cluster_label, DEFAULT_CONFIG).copy()

    # Extract non-WalkerConfig fields
    raw.pop("width", None)
    raw.pop("height", None)
    raw.pop("n_waypoints", None)

    # Apply difficulty modifiers
    if difficulty == "easy":
        raw["inner_size"] = min(raw.get("inner_size", 6) + 1, 9)
        raw["outer_margin"] = max(raw.get("outer_margin", 1) - 1, 0)
        sr = raw.get("size_range", (4, 7))
        raw["size_range"] = (min(sr[0] + 1, sr[1]), sr[1] + 1)
    elif difficulty == "hard":
        raw["inner_size"] = max(raw.get("inner_size", 6) - 1, 2)
        raw["outer_margin"] = min(raw.get("outer_margin", 1) + 1, 4)
        sr = raw.get("size_range", (4, 7))
        raw["size_range"] = (max(sr[0] - 1, 1), max(sr[1] - 1, sr[0]))

    return WalkerConfig(**raw)


def get_segment_dimensions(cluster_label: str) -> tuple[int, int]:
    """Get recommended width x height for a cluster type."""
    raw = CLUSTER_TO_CONFIG.get(cluster_label, DEFAULT_CONFIG)
    return raw.get("width", 100), raw.get("height", 100)
