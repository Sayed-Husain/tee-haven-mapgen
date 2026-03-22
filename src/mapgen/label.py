"""LLM-powered labeling for segment clusters.

After clustering (cluster.py), each cluster is just a group of
similar segments with no human-readable name. This module sends
representative segments from each cluster to an LLM and asks:
"What gameplay pattern do these segments share?"

The result: every cluster gets a short label (3-6 words) and a
one-sentence description. These labels become the vocabulary the
generation pipeline uses — instead of "cluster 12", the LLM sees
"tight freeze corridor descent".

pattern_library.json (unlabeled) --> LLM labels each cluster --> pattern_library.json (labeled)

WHY LLM LABELING?
-----------------
We could hand-label 50 clusters ourselves, but:
  1. It's tedious and subjective
  2. If we re-cluster (different k, new maps), all labels are lost
  3. The LLM can see the ASCII grid and describe what it SEES — often
     more accurate than our manual categorization
  4. It's cheap: ~50 API calls × ~500 tokens = ~25k tokens total

The user reviews and can edit labels after — the LLM provides a
solid first draft.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from openai import OpenAI

from mapgen.cluster import (
    PatternLibrary,
    ClusterInfo,
    SegmentFeatures,
    load_pattern_library,
    TILE_KEYS,
)
from mapgen.analyze import _iter_segments_from_file


# ── Label generation prompt ───────────────────────────────────────

LABEL_SYSTEM_PROMPT = """\
You are a Teeworlds Gores map analyst. You study segments extracted \
from real Gores maps and identify the gameplay pattern they represent.

## Gores Gameplay Context

- Players navigate by jumping, hooking solid surfaces, and using momentum.
- FREEZE tiles (~) temporarily freeze the player but are traversable.
- NOHOOK surfaces (%) cannot be hooked -- player must use other surfaces.
- DEATH tiles (X) kill instantly.
- Segments flow from entry to exit -- the player must find a path through.
- Hookable SOLID surfaces (#) are what the player grabs onto with the hook.

## Your Task

Given 2-3 representative segments from the same cluster, identify the \
shared gameplay pattern. You will also see what makes this cluster \
DIFFERENT from the global average -- use that to make the label unique.

Respond with JSON:

{
  "label": "3-6 word pattern name",
  "description": "One sentence describing what the player experiences."
}

## Label Guidelines

- EVERY cluster label must be UNIQUE. If two clusters seem similar, look \
  harder at the distinguishing features section and the ASCII grids.
- Focus on the GAMEPLAY EXPERIENCE, not just tile composition.
  Good: "ceiling hook traverse" (describes what the player DOES)
  Bad: "many solid tiles" (describes what the data LOOKS like)
- The DISTINGUISHING FEATURES section tells you exactly what makes this \
  cluster unusual. Build your label around those differences.
  Example: if freeze is 3x the average, the label should mention freeze.
  Example: if nohook is 10x the average, the label should center on nohook.
  Example: if it has death tiles but most clusters don't, mention death.
  Example: if air% is very high (>70%), it's wide open chambers.
  Example: if solid% is very high (>50%), it's dense tight corridors.
- Be specific. "freeze section" is too vague -- "narrow freeze corridor \
  descent" tells us the shape, the tile type, AND the direction.
- Use these vocabulary categories:
  Movement: traverse, descent, ascent, drop, climb, swing, weave
  Shape: corridor, chamber, maze, tunnel, shaft, gap, bridge
  Tile: freeze, nohook, death, hookable, solid
  Density: tight, wide, open, narrow, dense, spacious, cramped
  Difficulty: precise, forgiving, punishing, technical
"""


def _build_label_prompt(
    cluster: ClusterInfo,
    representative_segments: list[tuple[SegmentFeatures, str]],
    global_averages: dict[str, float] | None = None,
) -> str:
    """Build the user prompt for labeling one cluster.

    Shows the LLM:
    1. What makes this cluster DIFFERENT from the average (key differentiator)
    2. Cluster stats (size, centroid features)
    3. 2-3 representative ASCII grids with their computed features

    Args:
        cluster: The cluster to label.
        representative_segments: List of (features, ascii_grid) tuples.
        global_averages: Average tile pcts across ALL segments for comparison.
    """
    parts = []

    parts.append(f"## Cluster {cluster.id} ({cluster.size} segments)\n")

    # Compute and show distinguishing features vs global average
    centroid = cluster.centroid
    if len(centroid) >= 16 and global_averages:
        parts.append("### DISTINGUISHING FEATURES (what makes this cluster unique):\n")

        tile_names = ["air", "solid", "freeze", "death", "nohook"]
        diffs = []
        for idx, name in enumerate(tile_names):
            cluster_val = centroid[idx]
            global_val = global_averages.get(name, 0)
            if global_val > 0.5:  # only compare if meaningful
                ratio = cluster_val / global_val
                diff = cluster_val - global_val
                if abs(diff) > 3:  # significant difference (>3 percentage points)
                    if ratio > 1.5:
                        diffs.append(f"- {name.upper()} is {ratio:.1f}x the average "
                                    f"({cluster_val:.0f}% vs avg {global_val:.0f}%) "
                                    f"-- UNUSUALLY HIGH")
                    elif ratio < 0.5:
                        diffs.append(f"- {name.upper()} is {ratio:.1f}x the average "
                                    f"({cluster_val:.0f}% vs avg {global_val:.0f}%) "
                                    f"-- UNUSUALLY LOW")
                    elif diff > 0:
                        diffs.append(f"- {name.upper()}: {cluster_val:.0f}% "
                                    f"(avg {global_val:.0f}%, +{diff:.0f}pp above)")
                    else:
                        diffs.append(f"- {name.upper()}: {cluster_val:.0f}% "
                                    f"(avg {global_val:.0f}%, {diff:.0f}pp below)")
            elif cluster_val > 1:
                # Tile type is rare globally but present here
                diffs.append(f"- {name.upper()}: {cluster_val:.0f}% "
                            f"(rare -- global avg is only {global_val:.1f}%)")

        if not diffs:
            diffs.append("- No extreme outliers -- look at the ASCII grids "
                        "for structural/shape differences")

        parts.extend(diffs)
        parts.append("")

    # Show cluster centroid stats
    if len(centroid) >= 16:
        parts.append("**Cluster averages:**")
        parts.append(f"- Tile mix: air={centroid[0]:.0f}%, "
                     f"solid={centroid[1]:.0f}%, "
                     f"freeze={centroid[2]:.0f}%, "
                     f"death={centroid[3]:.0f}%, "
                     f"nohook={centroid[4]:.0f}%")
        parts.append(f"- Openness: {centroid[14]:.2f}")
        parts.append(f"- Aspect ratio (w/h): {centroid[15]:.2f}")
        parts.append("")

    # Show representative segments
    for i, (feat, ascii_grid) in enumerate(representative_segments):
        parts.append(f"### Representative {i + 1}: "
                     f"{feat.source_map} segment {feat.segment_index}")
        parts.append(f"- Size: {feat.width}x{feat.height}")
        parts.append(f"- Flow: {feat.primary_flow}, {feat.path_complexity}")
        parts.append(f"- Entry: {feat.entry_side} -> Exit: {feat.exit_side}")

        feature_tags = []
        if feat.has_freeze:
            feature_tags.append(f"freeze ({feat.tile_pcts.get('freeze', 0):.0f}%)")
        if feat.has_death:
            feature_tags.append(f"death ({feat.tile_pcts.get('death', 0):.0f}%)")
        if feat.has_nohook:
            feature_tags.append(f"nohook ({feat.tile_pcts.get('nohook', 0):.0f}%)")
        if feature_tags:
            parts.append(f"- Features: {', '.join(feature_tags)}")

        # Truncate very large ASCII grids to save tokens
        lines = ascii_grid.strip().split("\n")
        if len(lines) > 30:
            lines = lines[:30] + [f"... ({len(lines) - 30} more rows)"]
        # Also truncate very wide lines
        truncated_lines = []
        for line in lines:
            if len(line) > 80:
                truncated_lines.append(line[:80] + "...")
            else:
                truncated_lines.append(line)

        parts.append(f"```\n{chr(10).join(truncated_lines)}\n```")
        parts.append("")

    parts.append(
        "Respond with JSON containing 'label' and 'description'. "
        "The label MUST reflect this cluster's distinguishing features."
    )

    return "\n".join(parts)


# ── ASCII grid retrieval ──────────────────────────────────────────

def _fetch_ascii_grids(
    segments: list[SegmentFeatures],
    indices: list[int],
    output_dir: Path,
) -> list[tuple[SegmentFeatures, str]]:
    """Retrieve ASCII grids for specific segments by re-reading their files.

    We don't store ASCII grids in the pattern library (too much memory).
    When we need them for labeling, we re-read just the relevant files.

    Args:
        segments: Full segments list from the pattern library.
        indices: Indices into segments list for the segments we want.
        output_dir: Directory containing *_segments.txt files.

    Returns:
        List of (SegmentFeatures, ascii_grid_string) tuples.
    """
    # Group by source map to minimize file reads
    needed: dict[str, list[tuple[int, int]]] = {}  # source_map → [(seg_idx, list_idx)]
    for list_idx in indices:
        feat = segments[list_idx]
        needed.setdefault(feat.source_map, []).append(
            (feat.segment_index, list_idx)
        )

    # Read only needed files
    results: dict[int, str] = {}  # list_idx → ascii_grid

    for txt_file in output_dir.glob("*_segments.txt"):
        source_map = txt_file.stem.replace("_segments", "")
        if source_map not in needed:
            continue

        target_pairs = needed[source_map]
        target_seg_indices = {seg_idx for seg_idx, _ in target_pairs}

        file_segments = _iter_segments_from_file(txt_file)
        for src, seg_idx, ascii_text in file_segments:
            if seg_idx in target_seg_indices:
                # Find which list_idx this corresponds to
                for target_seg_idx, list_idx in target_pairs:
                    if target_seg_idx == seg_idx:
                        results[list_idx] = ascii_text
                        break

    # Build output in the requested order
    output = []
    for list_idx in indices:
        ascii_grid = results.get(list_idx, "")
        if ascii_grid:
            output.append((segments[list_idx], ascii_grid))

    return output


# ── Global averages for differentiation ──────────────────────────

def _compute_global_averages(clusters: list[ClusterInfo]) -> dict[str, float]:
    """Compute size-weighted average tile percentages across all clusters.

    Each cluster centroid contains the average feature vector for its
    members. We combine all centroids weighted by cluster size to get
    the global baseline. This lets each cluster's prompt say things like
    "freeze is 3x the average" -- making labels more distinctive.

    Args:
        clusters: All clusters from the pattern library.

    Returns:
        Dict like {"air": 48.2, "solid": 28.1, "freeze": 20.5, ...}
    """
    tile_names = ["air", "solid", "freeze", "death", "nohook"]
    totals = {name: 0.0 for name in tile_names}
    total_weight = 0

    for cluster in clusters:
        if len(cluster.centroid) < 5:
            continue
        weight = cluster.size
        for idx, name in enumerate(tile_names):
            totals[name] += cluster.centroid[idx] * weight
        total_weight += weight

    if total_weight == 0:
        return {name: 0.0 for name in tile_names}

    return {name: totals[name] / total_weight for name in tile_names}


# ── Main labeling pipeline ────────────────────────────────────────

def label_clusters(
    library_path: Path = Path("maps/output/pattern_library.json"),
    output_dir: Path = Path("maps/output"),
    model: str = "gpt-4o-mini",
    skip_labeled: bool = True,
) -> PatternLibrary:
    """Label all clusters in the pattern library using an LLM.

    For each unlabeled cluster:
    1. Get 2-3 representative segments (closest to centroid)
    2. Fetch their ASCII grids from segment files
    3. Send to LLM with labeling prompt
    4. Parse the response and update the cluster

    The updated library is saved back to the same JSON file.

    Args:
        library_path: Path to pattern_library.json.
        output_dir: Directory with *_segments.txt files (for ASCII grids).
        model: OpenAI model to use. gpt-4o-mini is cheap and good enough
               for this classification task.
        skip_labeled: If True, don't re-label clusters that already have labels.
                      Useful when re-running after manual edits.

    Returns:
        Updated PatternLibrary with labels.
    """
    library = load_pattern_library(library_path)
    if library is None:
        print(f"No pattern library found at {library_path}")
        print("Run 'mapgen cluster' first.")
        return None

    client = OpenAI(max_retries=3)

    clusters_to_label = [
        c for c in library.clusters
        if not skip_labeled or c.label is None
    ]

    if not clusters_to_label:
        print("All clusters already labeled. Use --force to re-label.")
        return library

    # Compute global averages across ALL clusters (weighted by size)
    # so each cluster can see how it differs from the norm
    global_averages = _compute_global_averages(library.clusters)

    print(f"Labeling {len(clusters_to_label)} clusters "
          f"(model: {model})...\n")
    print(f"Global averages: air={global_averages['air']:.0f}%, "
          f"solid={global_averages['solid']:.0f}%, "
          f"freeze={global_averages['freeze']:.0f}%, "
          f"death={global_averages['death']:.1f}%, "
          f"nohook={global_averages['nohook']:.1f}%\n")

    t0 = time.time()
    labeled_count = 0

    for i, cluster in enumerate(clusters_to_label):
        # Fetch ASCII grids for representative segments
        reps = _fetch_ascii_grids(
            library.segments,
            cluster.representative_indices,
            output_dir,
        )

        if not reps:
            print(f"  Cluster {cluster.id}: no ASCII grids found, skipping")
            continue

        # Build prompt with global context for differentiation
        user_prompt = _build_label_prompt(cluster, reps, global_averages)

        # Call LLM
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": LABEL_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.3,  # low temp for consistent classification
            )

            result = json.loads(response.choices[0].message.content)
            cluster.label = result.get("label", f"cluster_{cluster.id}")
            cluster.description = result.get("description", "")
            labeled_count += 1

            print(f"  [{i + 1}/{len(clusters_to_label)}] "
                  f"Cluster {cluster.id} ({cluster.size} segs): "
                  f"{cluster.label}")

        except Exception as e:
            print(f"  [{i + 1}/{len(clusters_to_label)}] "
                  f"Cluster {cluster.id}: ERROR - {str(e)[:80]}")
            cluster.label = f"unlabeled_cluster_{cluster.id}"
            cluster.description = f"Labeling failed: {str(e)[:100]}"

        # Brief pause between API calls to be respectful
        if i < len(clusters_to_label) - 1:
            time.sleep(0.5)

    # Deduplication pass: re-label clusters that got duplicate labels.
    # Since each cluster is labeled independently, the LLM can't see
    # what labels it already assigned. We fix that here by telling it
    # which labels are taken and asking for a unique alternative.
    used_labels = {}  # lowercase label -> list of cluster objects
    for c in library.clusters:
        if c.label:
            key = c.label.lower().strip()
            used_labels.setdefault(key, []).append(c)

    duplicates = {k: v for k, v in used_labels.items() if len(v) > 1}
    if duplicates:
        all_labels = [c.label for c in library.clusters if c.label]
        print(f"\nDedup pass: {len(duplicates)} duplicate labels found")

        for dup_label, dup_clusters in duplicates.items():
            # Keep the first (largest) cluster's label, re-label the rest
            dup_clusters.sort(key=lambda c: c.size, reverse=True)
            for cluster in dup_clusters[1:]:
                reps = _fetch_ascii_grids(
                    library.segments,
                    cluster.representative_indices,
                    output_dir,
                )
                if not reps:
                    continue

                user_prompt = _build_label_prompt(
                    cluster, reps, global_averages
                )
                # Add dedup constraint
                user_prompt += (
                    "\n\nIMPORTANT: These labels are ALREADY TAKEN "
                    "by other clusters. You MUST choose something different:\n"
                    + "\n".join(f"- {l}" for l in all_labels if l != cluster.label)
                )

                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": LABEL_SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                        response_format={"type": "json_object"},
                        temperature=0.7,  # higher temp for more creative alternatives
                    )
                    result = json.loads(response.choices[0].message.content)
                    new_label = result.get("label", cluster.label)
                    cluster.label = new_label
                    cluster.description = result.get(
                        "description", cluster.description
                    )
                    all_labels.append(new_label)
                    print(f"  Cluster {cluster.id}: '{dup_label}' -> '{new_label}'")
                except Exception as e:
                    print(f"  Cluster {cluster.id}: dedup failed - {str(e)[:60]}")

                time.sleep(0.5)

    elapsed = time.time() - t0
    print(f"\nLabeled {labeled_count}/{len(clusters_to_label)} clusters "
          f"in {elapsed:.1f}s")

    # Save updated library
    _save_labels_to_library(library, library_path)

    return library


def _save_labels_to_library(library: PatternLibrary, path: Path) -> None:
    """Update the pattern library JSON with new labels.

    Instead of rewriting the entire file, we load it, update just the
    cluster labels, and save. This preserves any manual edits the user
    might have made to other fields.
    """
    data = json.loads(path.read_text(encoding="utf-8"))

    # Build a lookup by cluster ID
    label_map = {c.id: c for c in library.clusters}

    for cluster_data in data.get("clusters", []):
        cid = cluster_data["id"]
        if cid in label_map:
            cluster_data["label"] = label_map[cid].label
            cluster_data["description"] = label_map[cid].description

    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Labels saved to {path}")
