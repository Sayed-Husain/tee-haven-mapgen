"""Batch-segment all OpenGores maps for the analysis library.

Runs the segmentation pipeline (extract -> pathfind -> segment) on every
.map file in the opengores-repo and saves results to maps/output/.

Usage:
    python scripts/batch_segment.py [--limit N] [--skip-existing]

This feeds into `mapgen analyze` which reads all *_segments.txt files
to build the example library for LLM generation.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add src/ to path so we can import mapgen modules
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))


def main():
    parser = argparse.ArgumentParser(description="Batch-segment OpenGores maps")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process only first N maps (0 = all)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip maps that already have segment output")
    parser.add_argument("--maps-dir", type=str,
                        default=str(Path(__file__).resolve().parent.parent / "maps" / "opengores-repo" / "maps"),
                        help="Directory containing .map files")
    parser.add_argument("--output-dir", type=str,
                        default=str(Path(__file__).resolve().parent.parent / "maps" / "output"),
                        help="Output directory for segment files")
    args = parser.parse_args()

    maps_dir = Path(args.maps_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect all .map files
    map_files = sorted(maps_dir.glob("*.map"))
    if args.limit > 0:
        map_files = map_files[:args.limit]

    total = len(map_files)
    print(f"Found {total} maps in {maps_dir}")
    print(f"Output: {out_dir}")
    print(f"Skip existing: {args.skip_existing}")
    print(f"{'='*60}\n")

    # Late imports (heavy modules)
    from mapgen.extract import load_game_layer, grid_to_ascii
    from mapgen.pathfind import segment_map

    succeeded = 0
    failed = 0
    skipped = 0
    total_segments = 0
    failed_maps: list[tuple[str, str]] = []

    t0 = time.time()

    for i, map_path in enumerate(map_files):
        name = map_path.stem
        progress = f"[{i+1}/{total}]"

        # Check if already processed
        seg_file = out_dir / f"{name}_segments.txt"
        if args.skip_existing and seg_file.exists():
            skipped += 1
            continue

        try:
            # Load game layer
            grid = load_game_layer(str(map_path))
            h, w = grid.shape

            if h < 10 or w < 10:
                raise ValueError(f"Map too small: {w}x{h}")

            # Run full segmentation pipeline
            segments = segment_map(
                grid,
                map_path=str(map_path),
                source_name=name,
            )

            if not segments:
                raise ValueError("No segments found")

            # Save in the same format as cmd_segment (for analyze.py compatibility)
            lines = []
            lines.append(f"Map: {name}  ({w}x{h})")
            lines.append(f"Segments: {len(segments)}")
            lines.append("")

            for seg in segments:
                sh, sw = seg.grid.shape
                lines.append("=" * 60)
                lines.append(
                    f"Segment {seg.index}: "
                    f"x=[{seg.x_start}-{seg.x_end}] y=[{seg.y_start}-{seg.y_end}]  "
                    f"({sw}x{sh})"
                )
                if hasattr(seg, 'checkpoint') and seg.checkpoint:
                    lines.append(
                        f"Checkpoint: platform at y={seg.checkpoint.y}, "
                        f"x=[{seg.checkpoint.x_start}-{seg.checkpoint.x_end}] "
                        f"(width={seg.checkpoint.width})"
                    )
                lines.append("=" * 60)
                lines.append(grid_to_ascii(seg.grid))
                lines.append("")

            seg_file.write_text("\n".join(lines), encoding="utf-8")

            n_segs = len(segments)
            total_segments += n_segs
            succeeded += 1

            # Progress logging
            if (i + 1) % 50 == 0 or (i + 1) == total:
                elapsed = time.time() - t0
                rate = (succeeded + failed) / elapsed if elapsed > 0 else 0
                print(f"  {progress} {name}: {n_segs} segments "
                      f"[{succeeded} ok, {failed} fail, {skipped} skip — "
                      f"{rate:.1f} maps/s, {total_segments} total segs]")

        except Exception as e:
            failed += 1
            err_msg = str(e)[:80]
            failed_maps.append((name, err_msg))

            # Log first few failures, then periodically
            if failed <= 5 or (i + 1) % 50 == 0:
                print(f"  {progress} {name}: FAILED — {err_msg}")

    # Summary
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Batch segmentation complete in {elapsed:.1f}s")
    print(f"  Succeeded: {succeeded} maps -> {total_segments} segments")
    print(f"  Failed:    {failed}")
    print(f"  Skipped:   {skipped}")
    print(f"  Total:     {total}")

    if failed_maps:
        print(f"\nFailed maps ({len(failed_maps)}):")
        for name, err in failed_maps[:30]:
            print(f"  - {name}: {err}")
        if len(failed_maps) > 30:
            print(f"  ... and {len(failed_maps) - 30} more")

    print(f"\nSegment files saved to: {out_dir}")
    print(f"Next: run `mapgen analyze` to build the example library.")


if __name__ == "__main__":
    main()
