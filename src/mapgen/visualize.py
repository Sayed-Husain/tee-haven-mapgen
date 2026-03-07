"""Render game-layer grids as colored PNG images with segment overlays."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mapgen.extract import AIR, SOLID, DEATH, FREEZE, NOHOOK, ENTITY

# ── Tile colors (RGB) ────────────────────────────────────────────
TILE_COLORS = {
    AIR:    (26,  26,  46),   # dark navy
    SOLID:  (140, 140, 140),  # gray
    DEATH:  (220,  40,  40),  # red
    FREEZE: (0,   180, 220),  # cyan
    NOHOOK: (220, 200,  50),  # yellow
    ENTITY: (50,  220,  80),  # green
}

# ── Segment tint colors (RGBA with alpha) ────────────────────────
SEGMENT_TINTS = [
    (255, 100, 100, 50),   # red
    (100, 100, 255, 50),   # blue
    (100, 255, 100, 50),   # green
    (255, 200, 50,  50),   # yellow
    (200, 100, 255, 50),   # purple
    (255, 150, 50,  50),   # orange
    (50,  255, 200, 50),   # teal
    (255, 100, 200, 50),   # pink
]

SCALE = 4            # pixels per tile
BORDER_WIDTH = 2     # segment boundary line width
LABEL_COLOR = (255, 255, 255)
BORDER_COLOR = (255, 255, 255, 200)
CHECKPOINT_COLOR = (255, 255, 0, 180)  # yellow for checkpoint markers


def grid_to_image(grid: np.ndarray) -> Image.Image:
    """Convert a 2D tile grid to a colored RGB image."""
    height, width = grid.shape
    img_w, img_h = width * SCALE, height * SCALE

    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    for tile_id, color in TILE_COLORS.items():
        mask = grid == tile_id
        rgb[mask] = color

    img = Image.fromarray(rgb, mode="RGB")
    img = img.resize((img_w, img_h), Image.Resampling.NEAREST)
    return img


def render_segments(
    grid: np.ndarray,
    segments: list,
    output_path: str | Path,
) -> Path:
    """Render the full map with colored rectangular segment overlays.

    Supports both column-based segments (with y_start/y_end) and
    legacy column-only segments.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    base = grid_to_image(grid).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    height, width = grid.shape

    for seg in segments:
        tint = SEGMENT_TINTS[seg.index % len(SEGMENT_TINTS)]

        # Support both rectangular and column-only segments
        x0 = seg.x_start * SCALE
        x1 = seg.x_end * SCALE
        y0 = getattr(seg, 'y_start', 0) * SCALE
        y1 = getattr(seg, 'y_end', height) * SCALE

        # Semi-transparent tint rectangle
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], fill=tint)

        # Border around the segment
        draw.rectangle(
            [x0, y0, x1 - 1, y1 - 1],
            outline=BORDER_COLOR,
            width=BORDER_WIDTH,
        )

        # Segment label at the top-left of the rectangle
        label = f"S{seg.index}"
        draw.text((x0 + 4, y0 + 2), label, fill=LABEL_COLOR)

        # Mark checkpoint if available (from pathfind-based segments)
        cp = getattr(seg, 'checkpoint', None)
        if cp is not None and hasattr(cp, 'tiles'):
            for cx, cy in cp.tiles:
                px = cx * SCALE
                py = cy * SCALE
                draw.rectangle(
                    [px, py, px + SCALE - 1, py + SCALE - 1],
                    fill=CHECKPOINT_COLOR,
                )

    result = Image.alpha_composite(base, overlay)
    result = result.convert("RGB")
    result.save(out)

    return out
