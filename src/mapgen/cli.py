"""CLI entry point for the mapgen tool."""

import argparse
from pathlib import Path
import sys

from dotenv import load_dotenv

from mapgen.extract import load_game_layer, grid_to_ascii, print_grid_stats
from mapgen.floors import detect_floors
from mapgen.pathfind import segment_map
from mapgen.visualize import render_segments


def cmd_extract(args: argparse.Namespace) -> None:
    """Extract and display the game layer as ASCII."""
    grid = load_game_layer(args.map_file)
    h, w = grid.shape

    lines = []
    lines.append(f"Map: {args.map_file}  ({w}x{h})\n")

    from io import StringIO
    import contextlib
    buf = StringIO()
    with contextlib.redirect_stdout(buf):
        print_grid_stats(grid)
    lines.append(buf.getvalue())

    lines.append(grid_to_ascii(grid))

    output = "\n".join(lines)

    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"Written to {args.output}")
    else:
        print(output)


def _get_source_name(map_file: str) -> str:
    """Extract the filename from a path."""
    return map_file.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]


def cmd_floors(args: argparse.Namespace) -> None:
    """Detect and display horizontal floors in the map."""
    import numpy as np
    from mapgen.extract import AIR

    grid = load_game_layer(args.map_file)
    h, w = grid.shape

    print(f"Map: {args.map_file}  ({w}x{h})")

    floors = detect_floors(grid)
    print(f"Detected {len(floors)} floors:\n")

    for f in floors:
        air_pct = np.mean(f.grid == AIR) * 100
        print(f"  Floor {f.index}: rows {f.y_start}-{f.y_end}  "
              f"(height={f.height}, air={air_pct:.0f}%)")

    print(f"\nDivider bands:")
    prev_end = 0
    for f in floors:
        if f.y_start > prev_end:
            band_h = f.y_start - prev_end
            print(f"  rows {prev_end}-{f.y_start}  (height={band_h}, solid wall)")
        prev_end = f.y_end
    if prev_end < h:
        band_h = h - prev_end
        print(f"  rows {prev_end}-{h}  (height={band_h}, solid wall)")


def cmd_segment(args: argparse.Namespace) -> None:
    """Segment the map using algorithmic path tracing."""
    grid = load_game_layer(args.map_file)
    h, w = grid.shape
    source = _get_source_name(args.map_file)
    stem = Path(source).stem

    print(f"Map: {args.map_file}  ({w}x{h})")
    print(f"Tracing player path and detecting checkpoints...")

    segments = segment_map(
        grid, map_path=args.map_file, source_name=source,
        min_checkpoint_distance=args.distance,
    )

    # Write text output
    lines = []
    lines.append(f"Map: {source}  ({w}x{h})")
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
        elif hasattr(seg, 'description') and seg.description:
            lines.append(f"Description: {seg.description}")
        lines.append("=" * 60)
        lines.append(grid_to_ascii(seg.grid))
        lines.append("")

    txt_output = "\n".join(lines)

    if args.output:
        txt_path = Path(args.output)
    else:
        txt_path = Path("maps/output") / f"{stem}_segments.txt"

    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text(txt_output, encoding="utf-8")

    # Render PNG visualization
    png_path = txt_path.with_suffix(".png")
    render_segments(grid, segments, png_path)

    print(f"\nFound {len(segments)} segments")
    print(f"  Text: {txt_path}")
    print(f"  Image: {png_path}")


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        prog="mapgen",
        description="AI-assisted Teeworlds map generation tool",
    )
    sub = parser.add_subparsers(dest="command")

    p_extract = sub.add_parser("extract", help="Extract game layer as ASCII")
    p_extract.add_argument("map_file", help="Path to a .map file")
    p_extract.add_argument("-o", "--output", help="Output file (default: print to terminal)")

    p_segment = sub.add_parser("segment", help="Segment map via path tracing")
    p_segment.add_argument("map_file", help="Path to a .map file")
    p_segment.add_argument("-o", "--output", help="Output file (default: maps/output/<name>_segments.txt)")
    p_segment.add_argument("-d", "--distance", type=int, default=50,
                           help="Min distance between checkpoints (default: 50)")

    p_floors = sub.add_parser("floors", help="Detect horizontal floors in a map")
    p_floors.add_argument("map_file", help="Path to a .map file")

    args = parser.parse_args()

    if args.command == "extract":
        cmd_extract(args)
    elif args.command == "segment":
        cmd_segment(args)
    elif args.command == "floors":
        cmd_floors(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
