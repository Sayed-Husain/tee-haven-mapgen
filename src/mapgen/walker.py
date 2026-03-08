"""Two-kernel walker for Gores map generation.

The walker carves passages through a solid grid by walking from waypoint
to waypoint, applying two kernels per step:

    1. Outer kernel → places FREEZE tiles (barrier padding)
    2. Inner kernel → places AIR tiles (playable space)

This two-phase approach guarantees that every air passage is surrounded
by freeze, which is the defining visual and gameplay characteristic of
real Gores maps. Freeze borders emerge naturally from the carving
process — no post-hoc placement needed.

Two-kernel approach developed for organic Gores corridor generation.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

from .extract import AIR, SOLID, FREEZE


# ── Kernel ───────────────────────────────────────────────────────

@dataclass
class Kernel:
    """2D carving footprint with controllable shape.

    The kernel is an NxN boolean mask centered on the walker's position.
    Circularity controls the shape:
        0.0 = full square (all cells included)
        1.0 = circle inscribed in the square (corners excluded)

    Why circularity matters:
        Square kernels create boxy corridors with sharp 90° corners.
        Circular kernels create rounder, more organic passages.
        Most real Gores maps use a mix — the walker mutates circularity
        as it walks, creating natural variation.
    """
    size: int
    circularity: float = 0.0
    mask: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.mask = self._build_mask()

    def _build_mask(self) -> np.ndarray:
        """Build the 2D boolean mask.

        For each cell in the NxN grid, compute distance from center.
        Include the cell if distance <= radius, where radius is
        interpolated between inscribed circle and bounding box diagonal
        based on circularity.
        """
        n = self.size
        if n <= 0:
            return np.zeros((1, 1), dtype=bool)

        mask = np.zeros((n, n), dtype=bool)
        center = (n - 1) / 2.0

        # Radius range: circle inscribed in square ↔ square diagonal
        min_radius = center  # inscribed circle
        max_radius = np.sqrt(2) * center  # diagonal to corner

        # circularity=1 → tight circle, circularity=0 → full square
        radius = self.circularity * min_radius + (1 - self.circularity) * max_radius

        for y in range(n):
            for x in range(n):
                dist = np.sqrt((x - center) ** 2 + (y - center) ** 2)
                if dist <= radius:
                    mask[y, x] = True

        return mask

    @property
    def half(self) -> int:
        """Half-size for centering the kernel on a position."""
        return self.size // 2


# ── Walker Configuration ─────────────────────────────────────────

@dataclass
class WalkerConfig:
    """Controls walker behavior. Different configs = different challenge types.

    These parameters directly map to the gameplay feel:
        - inner_size: passage width (1=claustrophobic, 5=open)
        - outer_margin: freeze padding thickness (more = more punishing)
        - momentum: direction predictability (high=straight, low=winding)
        - shift_weights: how strongly the walker favors moving toward goal
    """
    # Kernel sizes
    inner_size: int = 4          # inner (air) kernel size
    outer_margin: int = 2        # extra tiles for outer (freeze) kernel
    inner_circularity: float = 0.3
    outer_circularity: float = 0.0

    # Movement
    momentum_prob: float = 0.6   # probability of continuing same direction
    shift_weights: tuple[float, ...] = (0.5, 0.225, 0.2, 0.075)
    # ^ [toward_goal, lateral1, lateral2, away_from_goal]

    # Kernel mutation (per step)
    size_mutate_prob: float = 0.05     # chance of changing inner size per step
    size_range: tuple[int, int] = (2, 5)  # min/max inner kernel size
    circ_mutate_prob: float = 0.03     # chance of changing circularity

    # Waypoint navigation
    waypoint_reached_dist: float = 10.0  # squared distance to consider reached

    # Fade (gradual kernel growth at start)
    fade_steps: int = 30         # steps to grow from fade_min to inner_size
    fade_min_size: int = 2       # starting kernel size during fade


# ── Direction helpers ─────────────────────────────────────────────

UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRECTIONS = [UP, DOWN, LEFT, RIGHT]


def _rate_directions(
    pos: tuple[int, int], goal: tuple[int, int],
) -> list[tuple[float, tuple[int, int]]]:
    """Rate all 4 directions by squared distance to goal after moving.

    Returns sorted list of (distance, direction) — best direction first.
    """
    rated = []
    for d in DIRECTIONS:
        new_y = pos[0] + d[0]
        new_x = pos[1] + d[1]
        dist_sq = (new_y - goal[0]) ** 2 + (new_x - goal[1]) ** 2
        rated.append((dist_sq, d))
    rated.sort(key=lambda x: x[0])
    return rated


# ── The Walker ────────────────────────────────────────────────────

class Walker:
    """Probabilistic path-carving walker.

    Usage:
        grid = np.full((height, width), SOLID, dtype=np.uint8)
        walker = Walker(grid, waypoints, config)
        walker.walk()
        # grid is now carved with air passages surrounded by freeze
    """

    def __init__(
        self,
        grid: np.ndarray,
        waypoints: list[tuple[int, int]],
        config: WalkerConfig | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.grid = grid
        self.h, self.w = grid.shape
        self.waypoints = list(waypoints)
        self.config = config or WalkerConfig()
        self.rng = rng or np.random.default_rng()

        # Walker state
        self.pos = waypoints[0]
        self.current_waypoint_idx = 1
        self.last_direction: tuple[int, int] | None = None
        self.steps = 0
        self.finished = False

        # Build initial kernels
        self._rebuild_kernels(self.config.inner_size)

    def _rebuild_kernels(self, inner_size: int):
        """Rebuild inner and outer kernels with current sizes."""
        outer_size = inner_size + self.config.outer_margin * 2
        self.inner_kernel = Kernel(inner_size, self.config.inner_circularity)
        self.outer_kernel = Kernel(outer_size, self.config.outer_circularity)

    def _apply_kernel(self, pos: tuple[int, int], kernel: Kernel, tile: int):
        """Stamp a kernel onto the grid at the given position.

        For AIR: overwrites SOLID and FREEZE (carving takes priority).
        For FREEZE: only overwrites SOLID (freeze doesn't erase air).

        Preserves a 1-tile solid border around the map edge.
        """
        half = kernel.half
        for ky in range(kernel.size):
            for kx in range(kernel.size):
                if not kernel.mask[ky, kx]:
                    continue
                gy = pos[0] + ky - half
                gx = pos[1] + kx - half
                if 1 <= gy < self.h - 1 and 1 <= gx < self.w - 1:
                    current = self.grid[gy, gx]
                    if tile == AIR:
                        self.grid[gy, gx] = AIR
                    elif tile == FREEZE and current == SOLID:
                        self.grid[gy, gx] = FREEZE

    def _pick_direction(self) -> tuple[int, int]:
        """Choose next movement direction using momentum + weighted sampling.

        1. With probability momentum_prob, keep going same direction.
        2. Otherwise, rate directions by distance to goal and sample
           using shift_weights.
        """
        goal = self.waypoints[self.current_waypoint_idx]

        # Momentum: reuse last direction
        if (self.last_direction is not None
                and self.rng.random() < self.config.momentum_prob):
            ny = self.pos[0] + self.last_direction[0]
            nx = self.pos[1] + self.last_direction[1]
            margin = self.outer_kernel.half + 2
            if margin <= ny < self.h - margin and margin <= nx < self.w - margin:
                return self.last_direction

        # Rate directions by distance to goal
        rated = _rate_directions(self.pos, goal)

        # Apply shift weights
        weights = np.array(self.config.shift_weights[:len(rated)], dtype=np.float64)

        # Filter out-of-bounds directions
        margin = self.outer_kernel.half + 2
        valid_mask = np.ones(len(rated), dtype=bool)
        for i, (_, d) in enumerate(rated):
            ny = self.pos[0] + d[0]
            nx = self.pos[1] + d[1]
            if not (margin <= ny < self.h - margin and margin <= nx < self.w - margin):
                valid_mask[i] = False

        weights = weights * valid_mask
        if weights.sum() == 0:
            return rated[0][1]

        weights /= weights.sum()
        idx = self.rng.choice(len(rated), p=weights)
        return rated[idx][1]

    def _mutate_kernel(self):
        """Randomly adjust kernel size and circularity for variation."""
        cfg = self.config

        if self.rng.random() < cfg.size_mutate_prob:
            new_size = self.rng.integers(cfg.size_range[0], cfg.size_range[1] + 1)
            self._rebuild_kernels(int(new_size))

        if self.rng.random() < cfg.circ_mutate_prob:
            self.inner_kernel = Kernel(
                self.inner_kernel.size,
                float(self.rng.uniform(0.0, 0.8)),
            )

    def _check_waypoint_reached(self):
        """Advance to next waypoint if close enough to current goal."""
        if self.current_waypoint_idx >= len(self.waypoints):
            self.finished = True
            return

        goal = self.waypoints[self.current_waypoint_idx]
        dist_sq = (self.pos[0] - goal[0]) ** 2 + (self.pos[1] - goal[1]) ** 2

        if dist_sq <= self.config.waypoint_reached_dist:
            self.current_waypoint_idx += 1
            if self.current_waypoint_idx >= len(self.waypoints):
                self.finished = True

    def _get_fade_size(self) -> int:
        """Compute kernel size during fade phase (gradual growth at start)."""
        if self.steps >= self.config.fade_steps:
            return self.config.inner_size

        t = self.steps / max(self.config.fade_steps, 1)
        size = self.config.fade_min_size + t * (self.config.inner_size - self.config.fade_min_size)
        return max(1, int(round(size)))

    def step(self):
        """Execute one walker step: check waypoint → mutate → move → carve."""
        self._check_waypoint_reached()
        if self.finished:
            return

        # Fade: use smaller kernel at start
        fade_size = self._get_fade_size()
        if self.steps < self.config.fade_steps:
            self._rebuild_kernels(fade_size)
        else:
            self._mutate_kernel()

        # Pick direction and move
        direction = self._pick_direction()
        self.pos = (self.pos[0] + direction[0], self.pos[1] + direction[1])
        self.last_direction = direction

        # Carve: outer (freeze) first, then inner (air)
        self._apply_kernel(self.pos, self.outer_kernel, FREEZE)
        self._apply_kernel(self.pos, self.inner_kernel, AIR)

        self.steps += 1

    def walk(self, max_steps: int = 50000) -> int:
        """Run the walker until all waypoints are reached or max_steps."""
        while not self.finished and self.steps < max_steps:
            self.step()
        return self.steps


# ── Waypoint generation ──────────────────────────────────────────

def generate_waypoints(
    width: int,
    height: int,
    n_waypoints: int = 5,
    margin: int = 10,
    rng: np.random.Generator | None = None,
) -> list[tuple[int, int]]:
    """Generate a top-to-bottom waypoint sequence with horizontal variation."""
    if rng is None:
        rng = np.random.default_rng()

    waypoints = []
    y_positions = np.linspace(margin, height - margin, n_waypoints + 2).astype(int)

    for i, y in enumerate(y_positions):
        if i == 0 or i == len(y_positions) - 1:
            x = width // 2
        else:
            x = int(rng.integers(margin, width - margin))
        waypoints.append((int(y), x))

    return waypoints


def generate_sub_waypoints(
    waypoints: list[tuple[int, int]],
    max_dist: float = 30.0,
    shift_range: float = 8.0,
    rng: np.random.Generator | None = None,
) -> list[tuple[int, int]]:
    """Insert sub-waypoints between main waypoints for smoother paths.

    Breaks long segments into shorter hops with random lateral offsets.
    """
    if rng is None:
        rng = np.random.default_rng()

    result = [waypoints[0]]

    for i in range(len(waypoints) - 1):
        start = waypoints[i]
        end = waypoints[i + 1]

        dist = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        n_subs = max(0, int(dist / max_dist) - 1)

        for j in range(1, n_subs + 1):
            t = j / (n_subs + 1)
            y = int(start[0] + t * (end[0] - start[0]))
            x = int(start[1] + t * (end[1] - start[1]))
            y += int(rng.uniform(-shift_range, shift_range))
            x += int(rng.uniform(-shift_range, shift_range))
            result.append((y, x))

        result.append(end)

    return result


# ── High-level generation ────────────────────────────────────────

def generate_segment_grid(
    width: int = 80,
    height: int = 80,
    config: WalkerConfig | None = None,
    n_waypoints: int = 5,
    seed: int | None = None,
    entry_x: int | None = None,
    exit_x: int | None = None,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Generate a single segment grid using the walker.

    Returns (grid, waypoints) tuple. The grid starts fully SOLID and
    the walker carves air passages with freeze borders.

    entry_x/exit_x: horizontal positions for entry (top) and exit
    (bottom). If None, defaults to center. These ensure the walker's
    path connects to the actual entry/exit openings.
    """
    rng = np.random.default_rng(seed)
    config = config or WalkerConfig()

    grid = np.full((height, width), SOLID, dtype=np.uint8)

    margin = max(config.inner_size + config.outer_margin + 5, 10)

    # Fix entry/exit positions so walker path connects to openings
    ex = entry_x if entry_x is not None else width // 2
    xx = exit_x if exit_x is not None else width // 2

    waypoints = generate_waypoints(width, height, n_waypoints, margin, rng)
    # Override first and last waypoints to match entry/exit
    waypoints[0] = (margin, ex)
    waypoints[-1] = (height - margin, xx)

    waypoints = generate_sub_waypoints(
        waypoints, max_dist=25.0, shift_range=6.0, rng=rng,
    )

    walker = Walker(grid, waypoints, config, rng)
    walker.walk()

    # Carve entry and exit corridors to connect to map edges
    _carve_corridor(grid, (margin, ex), (2, ex))
    _carve_corridor(grid, (height - margin, xx), (height - 3, xx))

    return grid, waypoints


def _carve_corridor(
    grid: np.ndarray, start: tuple[int, int], end: tuple[int, int], width: int = 3,
):
    """Carve a straight air corridor between two points.

    Used to connect the walker's path to the map entry/exit edges.
    """
    h, w = grid.shape
    y0, x0 = start
    y1, x1 = end
    half = width // 2

    # Vertical corridor
    min_y, max_y = min(y0, y1), max(y0, y1)
    for y in range(min_y, max_y + 1):
        for dx in range(-half - 1, half + 2):
            gx = x0 + dx
            if 1 <= y < h - 1 and 1 <= gx < w - 1:
                if abs(dx) > half:
                    if grid[y, gx] == SOLID:
                        grid[y, gx] = FREEZE
                else:
                    grid[y, gx] = AIR
