"""LLM integration for blueprint generation.

Sends prompts to OpenAI, parses JSON responses, validates via
schema.validate_blueprint(), and retries on failure.  The LLM
designs segment layouts; it never touches raw tiles.

Prompt philosophy:
  - System: physics facts, schema format, obstacle vocabulary,
    design PRINCIPLES (not prescriptive rules)
  - User: difficulty, theme, entry/exit constraints, annotated
    examples from real Gores maps with descriptions
  - Retry: schema errors AND playability failures feed back to
    the LLM with ASCII visualization of what it built
"""

from __future__ import annotations

import json
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
from openai import OpenAI

from mapgen.schema import (
    Blueprint,
    BlueprintError,
    OBSTACLE_TYPES,
    DIFFICULTIES,
    MIN_WIDTH, MAX_WIDTH,
    MIN_HEIGHT, MAX_HEIGHT,
    MIN_OPENING, MAX_OPENING,
    MIN_CHECKPOINT_W, MAX_CHECKPOINT_W,
    validate_blueprint,
)


# ── System prompt ───────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a Teeworlds Gores map designer. You create JSON blueprints for \
individual map segments — rectangular gameplay sections that a player \
navigates from entry to exit.

## Gores Physics

- Players move by jumping, hooking solid surfaces, and using momentum.
- FREEZE tiles temporarily freeze the player but are traversable.
- NOHOOK surfaces cannot be hooked — player must use other surfaces.
- DEATH tiles kill instantly — player must avoid them.
- Checkpoints are flat platforms where the player can safely stand.
- The player needs at least 2 tiles of vertical air to stand.

## Blueprint JSON Schema

Return a single JSON object with these fields:

- "design_notes" (string): IMPORTANT — describe your design reasoning FIRST. \
What is the player's journey? How does the path flow? What makes it fun?
- "width" (int, {min_w}-{max_w}): Segment width in tiles
- "height" (int, {min_h}-{max_h}): Segment height in tiles
- "difficulty" (string): "easy", "medium", or "hard"
- "entry" (object): {{"side": "top"|"bottom"|"left"|"right", "x": int, "y": int, "width": int ({min_o}-{max_o})}}
- "exit" (object): same format as entry
- "checkpoint" (object): {{"x": int, "y": int, "width": int ({min_cp}-{max_cp})}}
- "obstacles" (array): list of obstacle objects
- "description" (string): one-line gameplay summary

## Obstacle Types

Each obstacle has: "type", "x", "y", "width", "height", and optional "params".

| Type              | What it does                                           | Key params                        |
|-------------------|--------------------------------------------------------|-----------------------------------|
| platform          | Flat SOLID surface with AIR above (stepping stone)     | —                                 |
| freeze_corridor   | Air corridor bordered by FREEZE tiles                  | direction: horizontal/vertical    |
| death_zone        | Rectangle of DEATH tiles to avoid/jump over            | —                                 |
| hook_point        | 1-2 SOLID tiles in open air (hookable target)          | —                                 |
| wall_gap          | SOLID wall with AIR gap to pass through                | gap_size: int, gap_y: int         |
| nohook_wall       | NOHOOK rectangle (can't hook)                          | —                                 |
| narrow_passage    | Tight AIR corridor between walls                       | direction, gap_width: int (2-4)   |

## How the Builder Works

1. The grid starts completely SOLID. Obstacles CARVE space into it.
2. Entry and exit each carve 3 tiles deep into the border.
3. If two obstacles don't overlap or touch, there's SOLID wall between them = BLOCKED.
4. freeze_corridor height must be >= 3 (horizontal) or width >= 3 (vertical) \
to have air inside.
5. Keep obstacles within bounds: x + width <= segment width, y + height <= segment height.
6. **Obstacle placement order matters.** narrow_passage fills its bounding box \
with SOLID before carving the air strip. If it overlaps an earlier obstacle, \
those carved tiles are OVERWRITTEN. List narrow_passages before the obstacles \
they should connect to, or ensure they don't overlap.

## Design Principles

Think about the PLAYER EXPERIENCE. A Gores segment should feel like a \
mini-adventure — the player enters, faces challenges, and finds the exit.

1. **Path IS the gameplay.** Every obstacle should be part of the route the \
player travels. If an obstacle doesn't affect how the player moves from entry \
to exit, it's wasted space. Don't place obstacles beside the path — place \
them IN the path.

2. **Direction changes create interest.** Real Gores maps rarely go straight \
down. The path might go down, then right, then down again. Use obstacles to \
force the player to change direction. Entry and exit don't need to be at the \
same x-position.

3. **Connected space is everything.** Every obstacle must overlap or touch \
another obstacle (or the entry/exit opening). Before placing each obstacle, \
ask: "Does this connect to the existing air space?" If not, it's an \
unreachable island.

4. **Build a chain, not a tube.** Instead of one big corridor, create a \
CHAIN of connected obstacles: narrow_passage connects to freeze_corridor \
which connects to platform which connects to another narrow_passage. Each \
link in the chain is a mini-challenge.

5. **Overlap means shared coordinates.** Two obstacles connect ONLY if their \
bounding boxes overlap or touch. Example: narrow_passage at (x=5, y=0, w=6, h=15) \
covers y=0-14 and x=5-10. A freeze_corridor at (x=8, y=12, w=10, h=5) covers \
y=12-16 and x=8-17. They overlap at y=12-14, x=8-10 → CONNECTED. But if the \
freeze_corridor started at y=16 instead, there'd be a solid gap → BLOCKED. \
**When building your chain, verify each obstacle's y-range overlaps the previous one.**

6. **Use the full canvas.** Don't center everything at one x-position. A \
40x30 segment can have paths that zigzag from left to right across the full \
width. Vary the x-positions of your obstacles.

7. **Difficulty = precision, not complexity.**
   - easy: wider passages (gap_width 3-4), more platforms, fewer death zones
   - medium: moderate passages, some death/freeze challenges
   - hard: narrow passages (gap_width 2), nohook walls, death zones near path

8. **The checkpoint must be ON the path.** Place it where the player \
naturally passes through, with enough air above to stand.
""".format(
    min_w=MIN_WIDTH, max_w=MAX_WIDTH,
    min_h=MIN_HEIGHT, max_h=MAX_HEIGHT,
    min_o=MIN_OPENING, max_o=MAX_OPENING,
    min_cp=MIN_CHECKPOINT_W, max_cp=MAX_CHECKPOINT_W,
)


# ── Main generation function ────────────────────────────────────────

def generate_blueprint(
    difficulty: str = "medium",
    entry_constraint: dict | None = None,
    theme: str | None = None,
    model: str = "gpt-4o",
    max_retries: int = 3,
    annotated_examples: list | None = None,
) -> tuple[Blueprint, np.ndarray]:
    """Generate a segment blueprint via LLM with playability feedback.

    The retry loop has TWO types of feedback:
      1. Schema errors (BlueprintError) — field validation
      2. Playability failures (BFS unreachable) — the LLM sees the ASCII
         grid it built and a reachability map showing the disconnection

    Args:
        difficulty: easy, medium, or hard.
        entry_constraint: Force entry to match previous segment's exit.
        theme: Optional theme hint.
        model: OpenAI model name.
        max_retries: Retries on validation/playability failure.
        annotated_examples: SegmentAnalysis objects from the example library.

    Returns:
        Tuple of (Blueprint, grid). The grid is the validated numpy array.
        If the LLM couldn't fully connect the path, the grid may have been
        post-processed with algorithmic gap-bridging.

    Raises:
        BlueprintError if all retries and gap-bridging fail.
    """
    # Late imports to avoid circular dependencies
    from mapgen.builder import build_grid
    from mapgen.validate import validate_segment
    from mapgen.extract import grid_to_ascii

    client = OpenAI(max_retries=3)  # retries 429/5xx with backoff

    user_prompt = _build_user_prompt(
        difficulty, entry_constraint, theme, annotated_examples,
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    last_error = None
    last_bp = None
    last_grid = None

    for attempt in range(1 + max_retries):
        if attempt > 0 and last_error:
            messages.append({"role": "user", "content": last_error})
            print(f"  Retry {attempt}/{max_retries} — feeding error back to LLM...")
            time.sleep(2)  # respect rate limits between retries

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.85,
        )

        raw_json = response.choices[0].message.content
        messages.append({"role": "assistant", "content": raw_json})

        # ── Stage 1: Schema validation ──
        try:
            data = json.loads(raw_json)
            bp = validate_blueprint(data)
        except (json.JSONDecodeError, BlueprintError) as e:
            last_error = (
                f"Your previous blueprint had validation errors:\n{e}\n\n"
                f"Please fix these issues and return a corrected JSON blueprint."
            )
            print(f"  Attempt {attempt + 1} schema error: {str(e)[:120]}...")
            continue

        # ── Stage 2: Playability validation (BFS) ──
        grid = build_grid(bp)
        result = validate_segment(grid, bp.entry, bp.exit)

        if result.playable:
            if bp.design_notes:
                print(f"  Design: {bp.design_notes[:100]}...")
            return bp, grid

        last_bp = bp
        last_grid = grid

        # Build detailed feedback so the LLM can SEE what went wrong
        ascii_viz = grid_to_ascii(grid)
        reachability_viz = _build_reachability_viz(grid, bp.entry)
        gap_info = _analyze_gap(grid, bp.entry)

        last_error = (
            f"Your blueprint builds but is NOT PLAYABLE. "
            f"The player cannot reach the exit from the entry.\n\n"
            f"## What you built (ASCII — '.' = air, '#' = solid, "
            f"'~' = freeze, 'X' = death, '%' = nohook):\n"
            f"```\n{ascii_viz}\n```\n\n"
            f"## Reachability ('R' = reachable from entry, '.' = unreachable air):\n"
            f"```\n{reachability_viz}\n```\n\n"
            f"## Stats:\n"
            f"- Reachable: {result.total_reachable}/{result.total_passable} tiles "
            f"({result.reachable_pct}%)\n"
            f"- Entry: {bp.entry.side} side at x={bp.entry.x}\n"
            f"- Exit: {bp.exit.side} side at x={bp.exit.x}\n\n"
            f"## Gap Analysis:\n"
            f"{gap_info}\n\n"
            f"## How to fix:\n"
            f"- Obstacles must OVERLAP or TOUCH to form connected air space\n"
            f"- Look at the 'R' tiles in the reachability map — those are reachable. "
            f"Extend obstacles from that area toward the exit.\n"
            f"- Check that x/y positions of adjacent obstacles actually overlap\n\n"
            f"Return a corrected JSON blueprint."
        )
        print(f"  Attempt {attempt + 1} not playable: "
              f"{result.reachable_pct}% reachable...")

    # ── Fallback: try algorithmic gap-bridging ──
    if last_bp is not None and last_grid is not None:
        print(f"  All retries failed. Attempting algorithmic gap-bridging...")
        bridged_grid, success = _bridge_gap(last_grid, last_bp.entry, last_bp.exit)
        if success:
            print(f"  Gap-bridging succeeded! Map is now playable.")
            return last_bp, bridged_grid

    raise BlueprintError(
        f"Failed after {1 + max_retries} attempts. Last error:\n{last_error}"
    )


# ── Reachability visualization ─────────────────────────────────────

def _build_reachability_viz(grid, entry) -> str:
    """Build a text map showing which tiles are reachable from entry.

    Reachable passable tiles → 'R'
    Unreachable passable tiles → '.' (the LLM can see the disconnection)
    Solid/death/etc → their normal ASCII character
    """
    from mapgen.validate import _opening_tiles
    from mapgen.pathfind import _is_passable
    from mapgen.extract import ASCII_CHARS

    h, w = grid.shape

    # BFS from entry tiles
    entry_tiles = _opening_tiles(entry, h, w)
    visited: set[tuple[int, int]] = set()
    queue: deque[tuple[int, int]] = deque()

    for (x, y) in entry_tiles:
        if 0 <= y < h and 0 <= x < w and _is_passable(int(grid[y, x])):
            visited.add((x, y))
            queue.append((x, y))

    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)]
    while queue:
        x, y = queue.popleft()
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if 0 <= ny < h and 0 <= nx < w and (nx, ny) not in visited:
                if _is_passable(int(grid[ny, nx])):
                    visited.add((nx, ny))
                    queue.append((nx, ny))

    # Build visualization
    rows = []
    for y in range(h):
        row = []
        for x in range(w):
            if (x, y) in visited:
                row.append("R")
            else:
                row.append(ASCII_CHARS.get(int(grid[y, x]), "?"))
        rows.append("".join(row))

    return "\n".join(rows)


# ── Gap analysis ──────────────────────────────────────────────────

def _analyze_gap(grid, entry) -> str:
    """Identify exactly where the disconnection is.

    Gives the LLM specific, actionable feedback:
    - Where the reachable area ends (frontier)
    - Where the nearest unreachable air is
    - Suggested obstacle placement to bridge the gap
    """
    from mapgen.validate import _opening_tiles
    from mapgen.pathfind import _is_passable
    from mapgen.extract import SOLID

    h, w = grid.shape

    # BFS from entry to find reachable tiles
    entry_tiles = _opening_tiles(entry, h, w)
    reachable = set()
    queue = deque()

    for (x, y) in entry_tiles:
        if 0 <= y < h and 0 <= x < w and _is_passable(int(grid[y, x])):
            reachable.add((x, y))
            queue.append((x, y))

    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)]
    while queue:
        x, y = queue.popleft()
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if 0 <= ny < h and 0 <= nx < w and (nx, ny) not in reachable:
                if _is_passable(int(grid[ny, nx])):
                    reachable.add((nx, ny))
                    queue.append((nx, ny))

    # Find unreachable passable tiles
    unreachable = set()
    for y in range(h):
        for x in range(w):
            if (x, y) not in reachable and _is_passable(int(grid[y, x])):
                unreachable.add((x, y))

    if not unreachable:
        return (
            "All passable tiles are reachable. The exit opening itself "
            "may be blocked by solid tiles — check that the exit carves "
            "into your obstacle chain."
        )

    # Find frontier: reachable tiles adjacent to solid
    frontier = []
    for x, y in reachable:
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= ny < h and 0 <= nx < w and int(grid[ny, nx]) == SOLID:
                frontier.append((x, y))
                break

    if not frontier:
        return "No frontier tiles found."

    # Find closest (frontier, unreachable) pair — sample for efficiency
    sample_f = random.sample(frontier, min(80, len(frontier)))
    sample_u = random.sample(list(unreachable), min(80, len(unreachable)))

    min_dist = float('inf')
    best_from = None
    best_to = None

    for fx, fy in sample_f:
        for ux, uy in sample_u:
            dist = abs(fx - ux) + abs(fy - uy)
            if dist < min_dist:
                min_dist = dist
                best_from = (fx, fy)
                best_to = (ux, uy)

    if best_from is None:
        return "Could not identify gap location."

    # Suggest a bridging area
    bx0 = max(0, min(best_from[0], best_to[0]) - 1)
    by0 = max(0, min(best_from[1], best_to[1]) - 1)
    bx1 = max(best_from[0], best_to[0]) + 2
    by1 = max(best_from[1], best_to[1]) + 2

    return (
        f"The reachable area ends near (x={best_from[0]}, y={best_from[1]}). "
        f"The nearest unreachable air is at (x={best_to[0]}, y={best_to[1]}), "
        f"{min_dist} tiles away through solid.\n"
        f"SUGGESTED FIX: Add or extend an obstacle to span "
        f"x={bx0}-{bx1}, y={by0}-{by1}. "
        f"For example, a narrow_passage at x={bx0}, y={by0}, "
        f"width={bx1 - bx0}, height={by1 - by0} with "
        f'direction="{"vertical" if abs(best_from[1] - best_to[1]) > abs(best_from[0] - best_to[0]) else "horizontal"}" '
        f"would bridge this gap."
    )


# ── Algorithmic gap-bridging ─────────────────────────────────────

def _bridge_gap(grid, entry, exit_) -> tuple[np.ndarray, bool]:
    """Bridge gaps between disconnected air regions as a last resort.

    Uses BFS through SOLID tiles to find the shortest path from the
    reachable frontier to unreachable air, then carves a 2-tile-wide
    channel. May attempt multiple bridges for multiple disconnected
    regions.

    This is a safety net — the LLM's creative design is preserved,
    we just add minimal connecting channels where needed.

    Returns:
        (modified_grid, success) — success is True if exit is reachable.
    """
    from mapgen.validate import _opening_tiles, validate_segment
    from mapgen.pathfind import _is_passable
    from mapgen.extract import AIR, SOLID

    grid = grid.copy()
    h, w = grid.shape
    dirs8 = [(-1, 0), (1, 0), (0, -1), (0, 1),
             (-1, -1), (-1, 1), (1, -1), (1, 1)]
    dirs4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    max_bridges = 5

    for bridge_num in range(max_bridges):
        # BFS from entry to find reachable area
        entry_tiles = _opening_tiles(entry, h, w)
        reachable = set()
        queue = deque()

        for (x, y) in entry_tiles:
            if 0 <= y < h and 0 <= x < w and _is_passable(int(grid[y, x])):
                reachable.add((x, y))
                queue.append((x, y))

        while queue:
            x, y = queue.popleft()
            for dx, dy in dirs8:
                nx, ny = x + dx, y + dy
                if 0 <= ny < h and 0 <= nx < w and (nx, ny) not in reachable:
                    if _is_passable(int(grid[ny, nx])):
                        reachable.add((nx, ny))
                        queue.append((nx, ny))

        # Check if exit is already reachable
        exit_tiles = set(_opening_tiles(exit_, h, w))
        if any(t in reachable for t in exit_tiles):
            return grid, True

        # Find unreachable passable tiles
        unreachable = set()
        for y in range(h):
            for x in range(w):
                if (x, y) not in reachable and _is_passable(int(grid[y, x])):
                    unreachable.add((x, y))

        if not unreachable:
            break  # No unreachable air — exit opening itself is blocked

        # Find solid tiles adjacent to reachable air (bridge start points)
        solid_seeds = set()
        for x, y in reachable:
            for dx, dy in dirs4:
                nx, ny = x + dx, y + dy
                if 0 <= ny < h and 0 <= nx < w and int(grid[ny, nx]) == SOLID:
                    solid_seeds.add((nx, ny))

        if not solid_seeds:
            break

        # BFS through SOLID to find shortest path to unreachable air
        parent: dict[tuple[int, int], tuple[int, int] | None] = {}
        bfs_queue: deque[tuple[int, int]] = deque()

        for pos in solid_seeds:
            parent[pos] = None
            bfs_queue.append(pos)

        target = None
        while bfs_queue:
            x, y = bfs_queue.popleft()

            # Check if adjacent to unreachable air
            for dx, dy in dirs4:
                nx, ny = x + dx, y + dy
                if (nx, ny) in unreachable:
                    target = (x, y)
                    break
            if target:
                break

            for dx, dy in dirs4:
                nx, ny = x + dx, y + dy
                if 0 <= ny < h and 0 <= nx < w and (nx, ny) not in parent:
                    if int(grid[ny, nx]) == SOLID:
                        parent[(nx, ny)] = (x, y)
                        bfs_queue.append((nx, ny))

        if target is None:
            break  # No solid path found

        # Trace path back from target to seed
        path = []
        pos = target
        while pos is not None:
            path.append(pos)
            pos = parent[pos]

        # Carve path as AIR with 2-tile width for playability
        for px, py in path:
            if int(grid[py, px]) == SOLID:
                grid[py, px] = AIR
            # Widen: carve one adjacent solid tile
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = px + dx, py + dy
                if 0 <= ny < h and 0 <= nx < w and int(grid[ny, nx]) == SOLID:
                    grid[ny, nx] = AIR
                    break

        print(f"    Bridge {bridge_num + 1}: carved {len(path)} tiles")

    # Final validation
    result = validate_segment(grid, entry, exit_)
    return grid, result.playable


# ── Prompt builders ─────────────────────────────────────────────────

def _build_user_prompt(
    difficulty: str,
    entry_constraint: dict | None,
    theme: str | None,
    annotated_examples: list | None,
) -> str:
    """Build the user prompt with annotated real-map examples.

    The key differences from the old approach:
    - Examples include natural language descriptions BEFORE the ASCII
    - No default "entry on top, exit on bottom" — LLM chooses freely
      (unless chaining constrains the entry)
    - Explicit instruction to fill design_notes first
    """
    parts = []

    parts.append(f"Generate a **{difficulty}** Gores segment blueprint as JSON.")

    if theme:
        parts.append(f"Theme: {theme}")

    if entry_constraint:
        parts.append(
            f"The entry MUST be: side={entry_constraint['side']}, "
            f"x={entry_constraint.get('x', 'any')}, "
            f"width={entry_constraint.get('width', 5)}."
        )
    else:
        parts.append(
            "Choose entry and exit sides that create an interesting path. "
            "They don't have to be on the same side — an entry at top-left "
            "with an exit at bottom-right creates a diagonal journey."
        )

    parts.append(
        "Remember: the grid starts as all SOLID. Your obstacles carve playable space. "
        "Create a connected chain of obstacles from entry through the checkpoint to exit.\n"
        "Fill in design_notes FIRST — describe the player's journey before listing obstacles."
    )

    # Annotated examples from real maps
    if annotated_examples:
        parts.append("\n## Real Gores segments for reference:\n")
        parts.append(
            "Study these segments extracted from real published Gores maps. "
            "Notice how the paths wind, change direction, and use the full "
            "canvas width — not just a straight vertical tube.\n"
        )
        for i, ex in enumerate(annotated_examples):
            # Description FIRST — so the LLM understands the design intent
            parts.append(
                f"### Example {i + 1}: {ex.source_map} segment {ex.segment_index}"
            )
            parts.append(f"**{ex.description}**\n")
            # Compact ASCII (truncate to save tokens)
            lines = ex.ascii_grid.strip().split("\n")
            if len(lines) > 25:
                lines = lines[:25] + [f"... ({len(lines) - 25} more rows)"]
            parts.append(f"```\n{chr(10).join(lines)}\n```")

    return "\n\n".join(parts)


# ── Example loading ────────────────────────────────────────────────

def load_annotated_examples(n: int = 3) -> list:
    """Load annotated examples from the analysis library.

    Returns SegmentAnalysis objects that include both descriptions
    and ASCII grids. Falls back gracefully if no library exists.
    """
    from mapgen.analyze import load_example_library

    library = load_example_library()
    if library and library.examples:
        examples = list(library.examples)
        # Shuffle for variety between generation runs
        random.shuffle(examples)
        return examples[:n]

    return []
