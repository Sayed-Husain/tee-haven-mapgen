"""Cluster all real-map segments into gameplay pattern archetypes.

This is the "training data" pipeline:
  1. Parse every *_segments.txt file (from batch_segment.py)
  2. For each segment, compute a lightweight feature vector (reusing analyze.py)
  3. Run KMeans clustering across all segments
  4. Pick the best k via silhouette score
  5. Persist the full cluster library to JSON

The output is a PatternLibrary — every segment tagged with a cluster ID,
plus cluster metadata (centroid, size, representative segments).

*_segments.txt  -->  Feature Extraction  -->  KMeans Cluster  -->  pattern_library.json
  (30k segs)

WHY CLUSTERING?
───────────────
We have ~30k segments from 916 real Gores maps. Feeding all of them to
the LLM would blow up the context window. Instead, we group similar
segments into ~40-60 clusters. Each cluster represents a "gameplay
pattern" (e.g., "tight freeze corridor descent", "wide hookable ceiling
traverse"). At generation time, we retrieve a few representative
examples per relevant cluster — the LLM gets quality over quantity.

WHY KMEANS?
───────────
KMeans is simple, deterministic (given a seed), and fast for 30k points
in 16 dimensions. We try multiple values of k and pick the one with the
best silhouette score — a measure of how well-separated the clusters are.
HDBSCAN would auto-pick k, but it's harder to debug and explain.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np

# Reuse the analysis functions we already built
from mapgen.analyze import (
    _ascii_to_grid,
    _tile_composition,
    _has_tile_type,
    _detect_entry_exit,
    _classify_flow,
    _iter_segments_from_file,
)
from mapgen.extract import FREEZE, DEATH, NOHOOK


# ── Data structures ───────────────────────────────────────────────

@dataclass
class SegmentFeatures:
    """Lightweight feature record for one segment.

    This is intentionally smaller than SegmentAnalysis — no ASCII grid,
    no description. We need to hold ~30k of these in memory during
    clustering, so every byte counts.
    """
    source_map: str
    segment_index: int
    width: int
    height: int

    # Tile composition (percentages)
    tile_pcts: dict[str, float]

    # Boolean features
    has_freeze: bool
    has_death: bool
    has_nohook: bool

    # Flow characteristics
    primary_flow: str       # "vertical" | "horizontal" | "winding"
    path_complexity: str    # "straight" | "L-shaped" | "zigzag"
    entry_side: str
    exit_side: str
    openness: float

    # Assigned after clustering (None until then)
    cluster_id: int | None = None


@dataclass
class ClusterInfo:
    """Metadata for one cluster."""
    id: int
    size: int                           # number of segments in this cluster
    centroid: list[float]               # 16-dim centroid vector
    label: str | None = None            # human-readable label (set by label.py)
    description: str | None = None      # one-sentence description (set by label.py)
    representative_indices: list[int] = field(default_factory=list)
    # ^ indices into the segments list for the 3 closest-to-centroid members


@dataclass
class PatternLibrary:
    """The full pattern library — segments + clusters."""
    segments: list[SegmentFeatures]
    clusters: list[ClusterInfo]
    n_clusters: int
    silhouette_score: float
    generated_at: str                   # ISO timestamp
    source_dir: str                     # where the segment files came from


# ── Feature vector construction ───────────────────────────────────

# Column order for the feature vector (16 dimensions):
#
#  0-4:  tile percentages (air, solid, freeze, death, nohook)
#  5-7:  boolean features (has_freeze, has_death, has_nohook)
#  8-10: flow one-hot (vertical, horizontal, winding)
# 11-13: complexity one-hot (straight, L-shaped, zigzag)
#    14: openness (0.0 - 1.0)
#    15: aspect ratio = width / height (captures shape)
#
# WHY ONE-HOT ENCODING?
# ─────────────────────
# Categorical features like "vertical" can't be fed into KMeans as a
# single integer (vertical=0, horizontal=1, winding=2) because KMeans
# uses Euclidean distance — it would think horizontal is "closer" to
# vertical than winding is. One-hot encoding (3 binary columns) gives
# each category equal distance from the others.

FLOW_CATEGORIES = ["vertical", "horizontal", "winding"]
COMPLEXITY_CATEGORIES = ["straight", "L-shaped", "zigzag"]
TILE_KEYS = ["air", "solid", "freeze", "death", "nohook"]

FEATURE_DIM = 16  # total dimensions


def _features_to_vector(feat: SegmentFeatures) -> np.ndarray:
    """Convert a SegmentFeatures into a 16-dim float vector.

    This is the representation KMeans will cluster on.
    """
    vec = np.zeros(FEATURE_DIM, dtype=np.float64)

    # Tile percentages (0-100 scale, will be normalized later)
    for i, key in enumerate(TILE_KEYS):
        vec[i] = feat.tile_pcts.get(key, 0.0)

    # Boolean features (0 or 1)
    vec[5] = float(feat.has_freeze)
    vec[6] = float(feat.has_death)
    vec[7] = float(feat.has_nohook)

    # Flow one-hot
    for i, cat in enumerate(FLOW_CATEGORIES):
        vec[8 + i] = 1.0 if feat.primary_flow == cat else 0.0

    # Complexity one-hot
    for i, cat in enumerate(COMPLEXITY_CATEGORIES):
        vec[11 + i] = 1.0 if feat.path_complexity == cat else 0.0

    # Openness (0.0 - 1.0)
    vec[14] = feat.openness

    # Aspect ratio (width / height) — captures shape
    vec[15] = feat.width / max(feat.height, 1)

    return vec


# ── Feature extraction (Pass 1) ──────────────────────────────────

def extract_all_features(
    output_dir: Path = Path("maps/output"),
    min_size: int = 10,
) -> list[SegmentFeatures]:
    """Parse all segment files and compute features for every segment.

    This is a streaming pass — we parse each file, compute features,
    then discard the grid. Only the lightweight SegmentFeatures survives.

    Args:
        output_dir: Directory containing *_segments.txt files.
        min_size: Skip segments smaller than this in either dimension.
                  Tiny fragments aren't useful patterns.

    Returns:
        List of SegmentFeatures for all valid segments.
    """
    files = sorted(output_dir.glob("*_segments.txt"))
    print(f"Found {len(files)} segment files in {output_dir}")

    if not files:
        print("  No segment files found. Run 'mapgen segment' or "
              "'scripts/batch_segment.py' first.")
        return []

    features: list[SegmentFeatures] = []
    skipped = 0
    t0 = time.time()

    for file_idx, txt_file in enumerate(files):
        file_segments = _iter_segments_from_file(txt_file)

        for source_map, seg_idx, ascii_text in file_segments:
            grid = _ascii_to_grid(ascii_text)
            if grid.size == 0:
                skipped += 1
                continue

            h, w = grid.shape

            # Skip tiny segments — they're usually boundary artifacts
            if w < min_size or h < min_size:
                skipped += 1
                del grid
                continue

            # Compute features (reusing analyze.py functions)
            tile_pcts = _tile_composition(grid)
            has_freeze = _has_tile_type(grid, FREEZE)
            has_death = _has_tile_type(grid, DEATH)
            has_nohook = _has_tile_type(grid, NOHOOK)

            entry_side, exit_side = _detect_entry_exit(grid)
            primary_flow, path_complexity = _classify_flow(
                grid, entry_side, exit_side,
            )

            # Openness: fraction of passable tiles
            from mapgen.extract import AIR, ENTITY
            passable_mask = np.isin(grid, [AIR, FREEZE, ENTITY])
            openness = float(passable_mask.sum()) / grid.size

            feat = SegmentFeatures(
                source_map=source_map,
                segment_index=seg_idx,
                width=w,
                height=h,
                tile_pcts=tile_pcts,
                has_freeze=has_freeze,
                has_death=has_death,
                has_nohook=has_nohook,
                primary_flow=primary_flow,
                path_complexity=path_complexity,
                entry_side=entry_side,
                exit_side=exit_side,
                openness=round(openness, 4),
            )
            features.append(feat)

            # Let the grid be garbage collected immediately
            del grid

        # Progress logging
        if (file_idx + 1) % 100 == 0 or (file_idx + 1) == len(files):
            elapsed = time.time() - t0
            print(f"  [{file_idx + 1}/{len(files)}] "
                  f"{len(features)} segments extracted "
                  f"({skipped} skipped, {elapsed:.0f}s)")

    elapsed = time.time() - t0
    print(f"\nFeature extraction complete: {len(features)} segments "
          f"({skipped} skipped) in {elapsed:.1f}s")

    return features


# ── Clustering (Pass 2) ──────────────────────────────────────────

def cluster_segments(
    features: list[SegmentFeatures],
    k_candidates: list[int] | None = None,
    random_state: int = 42,
) -> tuple[list[SegmentFeatures], list[ClusterInfo], int, float]:
    """Run KMeans with multiple k values, pick the best via silhouette score.

    HOW KMEANS WORKS (brief):
    ─────────────────────────
    KMeans partitions N data points into k clusters. Each cluster has a
    "centroid" (the mean of all its members). The algorithm:
      1. Pick k random initial centroids
      2. Assign each point to its closest centroid
      3. Recompute centroids as the mean of assigned points
      4. Repeat 2-3 until stable

    HOW SILHOUETTE SCORE WORKS:
    ───────────────────────────
    For each point, silhouette measures:
      a = mean distance to OTHER points in the SAME cluster (cohesion)
      b = mean distance to points in the NEAREST OTHER cluster (separation)
      silhouette = (b - a) / max(a, b)

    Range: -1 to +1. Higher = better separated clusters.
    We try multiple k values and pick the one with the highest score.

    WHY STANDARDSCALER?
    ───────────────────
    Our features have very different scales:
      - tile_pcts: 0-100 (percentages)
      - booleans: 0 or 1
      - openness: 0.0-1.0
      - aspect_ratio: 0.1-10+
    Without scaling, tile_pcts would dominate distance calculations
    (a difference of 50% in air content would dwarf a 0.5 difference
    in openness). StandardScaler normalizes each feature to mean=0,
    std=1 so they all contribute equally.

    Args:
        features: List of SegmentFeatures from extract_all_features().
        k_candidates: Values of k to try. Default: [20, 30, 40, 50, 60, 80].
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (updated_features, clusters, best_k, best_score).
        Features have cluster_id set. Clusters have centroids and reps.
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score

    if k_candidates is None:
        k_candidates = [20, 30, 40, 50, 60, 80]

    n = len(features)
    print(f"\nClustering {n} segments...")

    # Remove k values larger than the number of segments
    k_candidates = [k for k in k_candidates if k < n]
    if not k_candidates:
        print("  Too few segments to cluster meaningfully.")
        return features, [], 0, 0.0

    # Build the feature matrix (N x 16)
    X = np.array([_features_to_vector(f) for f in features])

    # Standardize: each column gets mean=0, std=1
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Try each k, compute silhouette score
    best_k = k_candidates[0]
    best_score = -1.0
    best_labels = None
    best_model = None

    print(f"  Trying k = {k_candidates}...")

    for k in k_candidates:
        # n_init=10: run KMeans 10 times with different random seeds,
        # keep the best. Prevents getting stuck in a bad local minimum.
        model = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=10,
            max_iter=300,
        )
        labels = model.fit_predict(X_scaled)

        # Silhouette on a sample if dataset is large (>10k is slow)
        if n > 10_000:
            # Sample 10k points for silhouette (it's O(n²))
            rng = np.random.RandomState(random_state)
            sample_idx = rng.choice(n, 10_000, replace=False)
            score = silhouette_score(
                X_scaled[sample_idx], labels[sample_idx],
            )
        else:
            score = silhouette_score(X_scaled, labels)

        print(f"    k={k:3d} -> silhouette={score:.4f}")

        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels
            best_model = model

    print(f"  Best: k={best_k} (silhouette={best_score:.4f})")

    # Assign cluster IDs to features
    for i, feat in enumerate(features):
        feat.cluster_id = int(best_labels[i])

    # Build cluster info
    # Centroids are in SCALED space — transform back to original space
    # for human-readable centroid values
    centroids_scaled = best_model.cluster_centers_
    centroids_original = scaler.inverse_transform(centroids_scaled)

    clusters: list[ClusterInfo] = []
    for cid in range(best_k):
        mask = best_labels == cid
        size = int(mask.sum())

        # Find 3 representatives closest to centroid (in scaled space)
        member_indices = np.where(mask)[0]
        if len(member_indices) == 0:
            continue

        distances = np.linalg.norm(
            X_scaled[member_indices] - centroids_scaled[cid], axis=1,
        )
        # argsort gives indices sorted by distance (closest first)
        closest = np.argsort(distances)[:3]
        rep_indices = [int(member_indices[i]) for i in closest]

        clusters.append(ClusterInfo(
            id=cid,
            size=size,
            centroid=centroids_original[cid].tolist(),
            representative_indices=rep_indices,
        ))

    # Sort clusters by size (largest first) for readability
    clusters.sort(key=lambda c: c.size, reverse=True)

    return features, clusters, best_k, best_score


# ── Persistence ───────────────────────────────────────────────────

def save_pattern_library(
    features: list[SegmentFeatures],
    clusters: list[ClusterInfo],
    n_clusters: int,
    silhouette_score: float,
    source_dir: str,
    output_path: Path,
) -> PatternLibrary:
    """Save the full pattern library to JSON.

    The JSON structure is designed for both machine consumption
    (the LangGraph pipeline reads it at generation time) and human
    browsing (mapgen patterns command).
    """
    library = PatternLibrary(
        segments=features,
        clusters=clusters,
        n_clusters=n_clusters,
        silhouette_score=silhouette_score,
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        source_dir=str(source_dir),
    )

    data = {
        "n_clusters": library.n_clusters,
        "silhouette_score": round(library.silhouette_score, 4),
        "generated_at": library.generated_at,
        "source_dir": library.source_dir,
        "clusters": [asdict(c) for c in library.clusters],
        # Segments: store compactly (skip None cluster_id default)
        "segments": [
            {
                "source_map": f.source_map,
                "segment_index": f.segment_index,
                "width": f.width,
                "height": f.height,
                "tile_pcts": f.tile_pcts,
                "has_freeze": f.has_freeze,
                "has_death": f.has_death,
                "has_nohook": f.has_nohook,
                "primary_flow": f.primary_flow,
                "path_complexity": f.path_complexity,
                "entry_side": f.entry_side,
                "exit_side": f.exit_side,
                "openness": f.openness,
                "cluster_id": f.cluster_id,
            }
            for f in library.segments
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Pattern library saved to {output_path} ({size_mb:.1f} MB)")

    return library


def load_pattern_library(
    path: Path = Path("maps/output/pattern_library.json"),
) -> PatternLibrary | None:
    """Load a previously saved pattern library.

    Returns None if the file doesn't exist.
    """
    if not path.exists():
        return None

    data = json.loads(path.read_text(encoding="utf-8"))

    segments = [
        SegmentFeatures(**seg) for seg in data.get("segments", [])
    ]

    clusters = [
        ClusterInfo(
            id=c["id"],
            size=c["size"],
            centroid=c["centroid"],
            label=c.get("label"),
            description=c.get("description"),
            representative_indices=c.get("representative_indices", []),
        )
        for c in data.get("clusters", [])
    ]

    return PatternLibrary(
        segments=segments,
        clusters=clusters,
        n_clusters=data.get("n_clusters", 0),
        silhouette_score=data.get("silhouette_score", 0.0),
        generated_at=data.get("generated_at", ""),
        source_dir=data.get("source_dir", ""),
    )


# ── Main pipeline ────────────────────────────────────────────────

def build_pattern_library(
    output_dir: Path = Path("maps/output"),
    cache_path: Path | None = None,
    k_candidates: list[int] | None = None,
) -> PatternLibrary:
    """Full pipeline: extract features → cluster → save.

    This is the function called by `mapgen cluster`.

    Args:
        output_dir: Directory with *_segments.txt files.
        cache_path: Where to save the JSON. Default: output_dir/pattern_library.json.
        k_candidates: Values of k to try for KMeans.

    Returns:
        The built PatternLibrary.
    """
    t0 = time.time()

    if cache_path is None:
        cache_path = output_dir / "pattern_library.json"

    # Pass 1: Extract features from all segments
    features = extract_all_features(output_dir)

    if not features:
        print("No segments found. Cannot build pattern library.")
        return PatternLibrary(
            segments=[], clusters=[], n_clusters=0,
            silhouette_score=0.0, generated_at="",
            source_dir=str(output_dir),
        )

    # Pass 2: Cluster
    features, clusters, best_k, score = cluster_segments(
        features, k_candidates=k_candidates,
    )

    # Save
    library = save_pattern_library(
        features, clusters, best_k, score,
        source_dir=str(output_dir),
        output_path=cache_path,
    )

    elapsed = time.time() - t0
    print(f"\nPattern library built in {elapsed:.1f}s")
    print(f"  {len(features)} segments -> {best_k} clusters "
          f"(silhouette={score:.4f})")

    return library
