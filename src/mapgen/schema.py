"""Blueprint schema for generated Gores map segments.

A Blueprint describes a segment in high-level terms: dimensions,
entry/exit positions, a checkpoint, and an array of named obstacle
patterns.  The builder (builder.py) converts this into a numpy tile
grid.  The LLM generates these; the algorithm consumes them.

Design philosophy:
- Expressive enough for interesting variety
- Constrained enough that the builder always produces a valid grid
- Clear error messages so the LLM retry loop knows what to fix
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ── Valid values ────────────────────────────────────────────────────

OBSTACLE_TYPES = {
    "platform",
    "freeze_corridor",
    "death_zone",
    "hook_point",
    "wall_gap",
    "nohook_wall",
    "narrow_passage",
}

SIDES = {"top", "bottom", "left", "right"}
DIFFICULTIES = {"easy", "medium", "hard"}

# Dimension limits (tiles)
MIN_WIDTH, MAX_WIDTH = 20, 60
MIN_HEIGHT, MAX_HEIGHT = 15, 50
MIN_OPENING, MAX_OPENING = 3, 8
MIN_CHECKPOINT_W, MAX_CHECKPOINT_W = 3, 12


# ── Dataclasses ─────────────────────────────────────────────────────

@dataclass
class Opening:
    """Entry or exit opening on a segment border."""
    side: str          # top, bottom, left, right
    x: int             # column position (for top/bottom)
    y: int = 0         # row position (for left/right; defaults to 0)
    width: int = 5     # opening width in tiles


@dataclass
class CheckpointSpec:
    """Flat platform where the player can rest and reset double jump."""
    x: int             # leftmost column
    y: int             # row of the platform surface
    width: int = 6     # platform width


@dataclass
class Obstacle:
    """A named obstacle pattern placed on the segment canvas."""
    type: str          # one of OBSTACLE_TYPES
    x: int             # top-left column
    y: int             # top-left row
    width: int = 4     # bounding box width
    height: int = 4    # bounding box height
    params: dict = field(default_factory=dict)  # type-specific params


@dataclass
class Blueprint:
    """Complete description of a generated segment."""
    width: int
    height: int
    difficulty: str
    entry: Opening
    exit: Opening
    checkpoint: CheckpointSpec
    obstacles: list[Obstacle]
    description: str = ""
    design_notes: str = ""    # LLM's reasoning about the design


# ── Validation ──────────────────────────────────────────────────────

class BlueprintError(ValueError):
    """Raised when a blueprint dict is invalid."""
    pass


def validate_blueprint(data: dict) -> Blueprint:
    """Parse a raw JSON dict into a validated Blueprint.

    Raises BlueprintError with a clear message describing exactly
    what's wrong — this message gets fed back to the LLM on retry.
    """
    errors: list[str] = []

    # ── Top-level dimensions ──
    width = data.get("width")
    height = data.get("height")

    if not isinstance(width, int):
        errors.append(f"'width' must be an integer, got {type(width).__name__}")
    elif not (MIN_WIDTH <= width <= MAX_WIDTH):
        errors.append(f"'width' must be {MIN_WIDTH}-{MAX_WIDTH}, got {width}")

    if not isinstance(height, int):
        errors.append(f"'height' must be an integer, got {type(height).__name__}")
    elif not (MIN_HEIGHT <= height <= MAX_HEIGHT):
        errors.append(f"'height' must be {MIN_HEIGHT}-{MAX_HEIGHT}, got {height}")

    # ── Difficulty ──
    difficulty = data.get("difficulty", "medium")
    if difficulty not in DIFFICULTIES:
        errors.append(f"'difficulty' must be one of {DIFFICULTIES}, got '{difficulty}'")

    # ── Entry / Exit ──
    entry = _validate_opening(data.get("entry"), "entry", width, height, errors)
    exit_ = _validate_opening(data.get("exit"), "exit", width, height, errors)

    # ── Checkpoint ──
    checkpoint = _validate_checkpoint(data.get("checkpoint"), width, height, errors)

    # ── Obstacles ──
    raw_obstacles = data.get("obstacles", [])
    if not isinstance(raw_obstacles, list):
        errors.append("'obstacles' must be a list")
        raw_obstacles = []

    obstacles: list[Obstacle] = []
    for i, raw in enumerate(raw_obstacles):
        obs = _validate_obstacle(raw, i, width, height, errors)
        if obs:
            obstacles.append(obs)

    # ── Bail if any errors ──
    if errors:
        raise BlueprintError("Blueprint validation failed:\n  - " + "\n  - ".join(errors))

    return Blueprint(
        width=width,
        height=height,
        difficulty=difficulty,
        entry=entry,
        exit=exit_,
        checkpoint=checkpoint,
        obstacles=obstacles,
        description=data.get("description", ""),
        design_notes=data.get("design_notes", ""),
    )


# ── Helpers ─────────────────────────────────────────────────────────

def _validate_opening(
    raw: dict | None, name: str,
    seg_w: int | None, seg_h: int | None,
    errors: list[str],
) -> Opening:
    """Validate an entry or exit opening."""
    if not isinstance(raw, dict):
        errors.append(f"'{name}' must be an object, got {type(raw).__name__}")
        return Opening(side="top", x=0)

    side = raw.get("side", "top")
    if side not in SIDES:
        errors.append(f"'{name}.side' must be one of {SIDES}, got '{side}'")

    x = raw.get("x", 0)
    y = raw.get("y", 0)
    w = raw.get("width", 5)

    if not isinstance(w, int) or not (MIN_OPENING <= w <= MAX_OPENING):
        errors.append(f"'{name}.width' must be {MIN_OPENING}-{MAX_OPENING}, got {w}")

    # Bounds check against segment dimensions (if available)
    if isinstance(seg_w, int) and isinstance(x, int):
        if side in ("top", "bottom") and x + w > seg_w:
            errors.append(
                f"'{name}' opening x={x} + width={w} exceeds segment width={seg_w}"
            )
    if isinstance(seg_h, int) and isinstance(y, int):
        if side in ("left", "right") and y + w > seg_h:
            errors.append(
                f"'{name}' opening y={y} + width={w} exceeds segment height={seg_h}"
            )

    return Opening(side=side, x=int(x), y=int(y), width=int(w))


def _validate_checkpoint(
    raw: dict | None,
    seg_w: int | None, seg_h: int | None,
    errors: list[str],
) -> CheckpointSpec:
    """Validate a checkpoint specification."""
    if not isinstance(raw, dict):
        errors.append(f"'checkpoint' must be an object, got {type(raw).__name__}")
        return CheckpointSpec(x=0, y=0)

    x = raw.get("x", 0)
    y = raw.get("y", 0)
    w = raw.get("width", 6)

    if not isinstance(w, int) or not (MIN_CHECKPOINT_W <= w <= MAX_CHECKPOINT_W):
        errors.append(
            f"'checkpoint.width' must be {MIN_CHECKPOINT_W}-{MAX_CHECKPOINT_W}, got {w}"
        )

    if isinstance(seg_w, int) and isinstance(x, int) and isinstance(w, int):
        if x + w > seg_w:
            errors.append(
                f"checkpoint x={x} + width={w} exceeds segment width={seg_w}"
            )

    if isinstance(seg_h, int) and isinstance(y, int):
        if y < 0 or y >= (seg_h or 999):
            errors.append(f"checkpoint y={y} out of bounds (height={seg_h})")

    return CheckpointSpec(x=int(x), y=int(y), width=int(w))


def _validate_obstacle(
    raw: dict | None, index: int,
    seg_w: int | None, seg_h: int | None,
    errors: list[str],
) -> Obstacle | None:
    """Validate a single obstacle entry."""
    if not isinstance(raw, dict):
        errors.append(f"obstacles[{index}] must be an object")
        return None

    obs_type = raw.get("type", "")
    if obs_type not in OBSTACLE_TYPES:
        errors.append(
            f"obstacles[{index}].type '{obs_type}' unknown. "
            f"Valid: {sorted(OBSTACLE_TYPES)}"
        )

    x = int(raw.get("x", 0))
    y = int(raw.get("y", 0))
    w = int(raw.get("width", 4))
    h = int(raw.get("height", 4))
    params = raw.get("params", {})

    if not isinstance(params, dict):
        errors.append(f"obstacles[{index}].params must be an object")
        params = {}

    # Bounds check
    if isinstance(seg_w, int) and x + w > seg_w:
        errors.append(
            f"obstacles[{index}] ({obs_type}) x={x} + width={w} "
            f"exceeds segment width={seg_w}"
        )
    if isinstance(seg_h, int) and y + h > seg_h:
        errors.append(
            f"obstacles[{index}] ({obs_type}) y={y} + height={h} "
            f"exceeds segment height={seg_h}"
        )

    return Obstacle(type=obs_type, x=x, y=y, width=w, height=h, params=params)
