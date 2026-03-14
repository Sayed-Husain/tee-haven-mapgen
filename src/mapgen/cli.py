"""CLI entry point for the mapgen tool."""

import argparse
from pathlib import Path
import sys

from dotenv import load_dotenv

import json

from mapgen.extract import load_game_layer, grid_to_ascii, print_grid_stats
from mapgen.floors import detect_floors
from mapgen.pathfind import segment_map
from mapgen.visualize import render_segments, grid_to_image


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


def cmd_build(args: argparse.Namespace) -> None:
    """Build a segment from a blueprint JSON file."""
    from mapgen.schema import validate_blueprint, BlueprintError
    from mapgen.builder import build_grid
    from mapgen.validate import validate_segment

    # Load and validate blueprint
    raw = json.loads(Path(args.blueprint).read_text(encoding="utf-8"))
    try:
        bp = validate_blueprint(raw)
    except BlueprintError as e:
        print(f"Invalid blueprint:\n{e}", file=sys.stderr)
        sys.exit(1)

    print(f"Blueprint: {bp.width}x{bp.height}, {bp.difficulty}")
    print(f"  Entry: {bp.entry.side} x={bp.entry.x} w={bp.entry.width}")
    print(f"  Exit:  {bp.exit.side} x={bp.exit.x} w={bp.exit.width}")
    print(f"  Checkpoint: y={bp.checkpoint.y} x={bp.checkpoint.x} w={bp.checkpoint.width}")
    print(f"  Obstacles: {len(bp.obstacles)}")

    # Build the grid
    grid = build_grid(bp)

    # ASCII output
    if args.ascii:
        print()
        print(grid_to_ascii(grid))

    # Validation
    if args.validate:
        result = validate_segment(grid, bp.entry, bp.exit)
        print(f"\nValidation:")
        print(f"  Playable: {result.playable}")
        print(f"  Path length: {result.path_length} steps")
        print(f"  Reachable: {result.reachable_pct}% "
              f"({result.total_reachable}/{result.total_passable} tiles)")

    # File outputs
    stem = Path(args.blueprint).stem
    if args.output:
        out_dir = Path(args.output)
    else:
        out_dir = Path("maps/output")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save ASCII
    txt_path = out_dir / f"{stem}_built.txt"
    txt_path.write_text(grid_to_ascii(grid), encoding="utf-8")

    # Save PNG
    png_path = out_dir / f"{stem}_built.png"
    img = grid_to_image(grid)
    img.save(str(png_path))

    print(f"\nOutput:")
    print(f"  Text:  {txt_path}")
    print(f"  Image: {png_path}")


def cmd_analyze(args: argparse.Namespace) -> None:
    """Analyze real segments and build the annotated example library."""
    from mapgen.analyze import build_example_library

    output_dir = Path(args.output) if args.output else Path("maps/output")
    cache_path = output_dir / "example_library.json"

    library = build_example_library(
        output_dir=output_dir,
        target_count=args.count,
        cache_path=cache_path,
    )

    print(f"\nAnalyzed {library.total_segments_analyzed} segments "
          f"from {len(library.source_maps)} maps")
    print(f"Selected {len(library.examples)} representative examples:")
    for ex in library.examples:
        print(f"\n  {ex.source_map} seg {ex.segment_index}: "
              f"{ex.width}x{ex.height}, {ex.primary_flow} flow, "
              f"{ex.path_complexity} path")
        if args.verbose:
            print(f"    {ex.description}")
    print(f"\nLibrary saved to {cache_path}")


def cmd_generate(args: argparse.Namespace) -> None:
    """Generate a complete map using LLM + builder pipeline."""
    from mapgen.llm import generate_blueprint, load_annotated_examples
    from mapgen.validate import validate_segment
    from mapgen.assemble import assemble_map

    import time

    n_segments = args.segments
    difficulty = args.difficulty
    theme = args.theme
    model = args.model
    dry_run = args.dry_run

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    stem = f"generated_{timestamp}"

    out_dir = Path(args.output) if args.output else Path("maps/output")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load annotated examples from the analysis library
    annotated_examples = load_annotated_examples(3)
    if annotated_examples:
        print(f"Using {len(annotated_examples)} annotated examples for LLM context")
    else:
        print("No example library found. Run 'mapgen analyze' first for better results.")

    # Generate segments one by one
    built_segments: list[tuple] = []  # (blueprint, grid)
    entry_constraint = None  # first segment has no constraint

    for i in range(n_segments):
        print(f"\n{'='*40}")
        print(f"Segment {i + 1}/{n_segments}")
        print(f"{'='*40}")

        # Generate blueprint + validated grid
        # (generate_blueprint now returns both — grid may be gap-bridged)
        from mapgen.schema import BlueprintError
        try:
            bp, grid = generate_blueprint(
                difficulty=difficulty,
                entry_constraint=entry_constraint,
                theme=theme,
                model=model,
                annotated_examples=annotated_examples,
            )
        except BlueprintError as e:
            print(f"  FAILED: {str(e)[:120]}...")
            print(f"  Skipping segment {i + 1} — will retry with simpler constraints")
            # Retry once with no entry constraint (let LLM choose freely)
            try:
                bp, grid = generate_blueprint(
                    difficulty=difficulty,
                    entry_constraint=None,
                    theme=theme,
                    model=model,
                    annotated_examples=annotated_examples,
                )
            except BlueprintError:
                print(f"  FAILED again. Stopping generation.")
                break

        print(f"  Blueprint: {bp.width}x{bp.height}")
        print(f"  Entry: {bp.entry.side} x={bp.entry.x} w={bp.entry.width}")
        print(f"  Exit:  {bp.exit.side} x={bp.exit.x} w={bp.exit.width}")
        print(f"  Obstacles: {len(bp.obstacles)}")
        if bp.design_notes:
            # Show first line of design notes
            first_line = bp.design_notes.split("\n")[0][:80]
            print(f"  Design: {first_line}...")

        # Validate returned grid (should always pass — for logging)
        result = validate_segment(grid, bp.entry, bp.exit)
        status = "PASS" if result.playable else "FAIL"
        print(f"  Validation: {status} (path={result.path_length}, reachable={result.reachable_pct}%)")

        # Save individual segment files
        bp_dir = out_dir / f"{stem}_blueprints"
        bp_dir.mkdir(parents=True, exist_ok=True)

        # Blueprint JSON (now includes design_notes)
        bp_json = {
            "width": bp.width, "height": bp.height, "difficulty": bp.difficulty,
            "entry": {"side": bp.entry.side, "x": bp.entry.x, "y": bp.entry.y, "width": bp.entry.width},
            "exit": {"side": bp.exit.side, "x": bp.exit.x, "y": bp.exit.y, "width": bp.exit.width},
            "checkpoint": {"x": bp.checkpoint.x, "y": bp.checkpoint.y, "width": bp.checkpoint.width},
            "obstacles": [
                {"type": o.type, "x": o.x, "y": o.y, "width": o.width, "height": o.height, "params": o.params}
                for o in bp.obstacles
            ],
            "description": bp.description,
            "design_notes": bp.design_notes,
        }
        json_path = bp_dir / f"segment_{i}.json"
        json_path.write_text(json.dumps(bp_json, indent=2), encoding="utf-8")

        # ASCII
        txt_path = bp_dir / f"segment_{i}.txt"
        txt_path.write_text(grid_to_ascii(grid), encoding="utf-8")

        # PNG
        png_path = bp_dir / f"segment_{i}.png"
        img = grid_to_image(grid)
        img.save(str(png_path))

        built_segments.append((bp, grid))

        # Constrain next segment's entry to match this exit
        # (keep vertical stacking — assembler handles top→bottom)
        # When exit is on bottom, pass the exact x position.
        # When exit is on a side (left/right), just pass the width
        # and let the LLM choose x — the assembler carves borders anyway.
        if bp.exit.side == "bottom":
            entry_constraint = {
                "side": "top",
                "x": bp.exit.x,
                "width": bp.exit.width,
            }
        else:
            entry_constraint = {
                "side": "top",
                "width": bp.exit.width,
            }

    if dry_run:
        print(f"\n{'='*40}")
        print(f"Dry run complete. {n_segments} blueprints saved to {bp_dir}")
        return

    # Assemble into .map
    print(f"\n{'='*40}")
    print(f"Assembling {n_segments} segments...")
    map_path = out_dir / f"{stem}.map"
    assemble_map(built_segments, str(map_path))

    # Overview PNG
    from mapgen.extract import load_game_layer
    full_grid = load_game_layer(str(map_path))
    overview_path = out_dir / f"{stem}_overview.png"
    overview_img = grid_to_image(full_grid)
    overview_img.save(str(overview_path))

    print(f"\nOutput:")
    print(f"  Map:        {map_path}")
    print(f"  Overview:   {overview_path}")
    print(f"  Blueprints: {bp_dir}/")


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

    p_build = sub.add_parser("build", help="Build a segment from a blueprint JSON")
    p_build.add_argument("blueprint", help="Path to a blueprint JSON file")
    p_build.add_argument("-o", "--output", help="Output directory (default: maps/output/)")
    p_build.add_argument("--ascii", action="store_true", help="Print ASCII grid to terminal")
    p_build.add_argument("--validate", action="store_true", help="Run BFS playability check")

    p_analyze = sub.add_parser("analyze", help="Analyze real segments and build example library")
    p_analyze.add_argument("-o", "--output", help="Output directory (default: maps/output/)")
    p_analyze.add_argument("-n", "--count", type=int, default=5,
                           help="Number of representative examples to select (default: 5)")
    p_analyze.add_argument("--verbose", action="store_true",
                           help="Print full description for each example")

    p_gen = sub.add_parser("generate", help="Generate a complete map via LLM + builder")
    p_gen.add_argument("-n", "--segments", type=int, default=5, help="Number of segments (default: 5)")
    p_gen.add_argument("-d", "--difficulty", default="medium",
                       choices=["easy", "medium", "hard"], help="Difficulty (default: medium)")
    p_gen.add_argument("-o", "--output", help="Output directory (default: maps/output/)")
    p_gen.add_argument("--theme", help="Theme hint for LLM (e.g. 'vertical drops', 'freeze maze')")
    p_gen.add_argument("--dry-run", action="store_true", help="Generate blueprints only, no .map")
    p_gen.add_argument("--no-validate", action="store_true", help="Skip BFS validation")
    p_gen.add_argument("--model", default="gpt-4o", help="OpenAI model (default: gpt-4o)")

    args = parser.parse_args()

    if args.command == "extract":
        cmd_extract(args)
    elif args.command == "segment":
        cmd_segment(args)
    elif args.command == "floors":
        cmd_floors(args)
    elif args.command == "build":
        cmd_build(args)
    elif args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "generate":
        cmd_generate(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
