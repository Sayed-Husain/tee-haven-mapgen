"""LangGraph pipeline for AI-driven Gores map generation.

Orchestrates the full generation flow:
    LLM plans → walker generates → post-process → validate → assemble → export

The LLM makes ONE creative decision at the start (challenge sequence,
difficulty curve, theme). The walker then executes each segment
deterministically. Retry logic handles validation failures.

Usage:
    from mapgen.graph import run_pipeline
    result = run_pipeline(difficulty="medium", n_segments=5, theme="grass")
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any, Optional, TypedDict

import numpy as np
from langgraph.graph import StateGraph, END

from .assemble import stitch_segments, write_map, SPAWN_ID, FINISH_ID
from .builder import _carve_opening
from .config_mapping import get_walker_config, get_segment_dimensions, DEFAULT_CONFIG
from .extract import AIR, SOLID, FREEZE
from .postprocess import widen_narrow_passages, fix_edge_bugs, remove_freeze_blobs
from .schema import Opening, CheckpointSpec, Blueprint
from .validate import validate_segment
from .walker import generate_segment_grid, WalkerConfig


# ── State ────────────────────────────────────────────────────────

class PipelineState(TypedDict, total=False):
    """Full state flowing through the LangGraph pipeline."""

    # User inputs
    user_difficulty: str
    user_n_segments: int
    user_theme: Optional[str]

    # LLM plan (set by plan_node, or hardcoded for testing)
    challenge_sequence: list[str]
    difficulty_progression: list[str]
    visual_theme: str

    # Segment processing
    segments: list[dict]
    current_segment_index: int

    # Assembly + export
    assembled_grid: Any  # np.ndarray
    entities: list[tuple[int, int, int]]
    output_path: Optional[str]

    # Retry limits
    max_seed_retries: int
    max_param_retries: int


# ── Node functions ───────────────────────────────────────────────

def plan_node(state: PipelineState) -> dict:
    """Initialize segments from the challenge sequence.

    For now this uses a hardcoded or pre-set challenge sequence.
    The LLM integration (step 4) will replace the sequence generation.
    """
    sequence = state["challenge_sequence"]
    difficulties = state["difficulty_progression"]

    segments = []
    for i, (cluster_label, diff) in enumerate(zip(sequence, difficulties)):
        w, h = get_segment_dimensions(cluster_label)
        cfg = get_walker_config(cluster_label, diff)

        segments.append({
            "index": i,
            "cluster_type": cluster_label,
            "difficulty": diff,
            "walker_config": {
                "inner_size": cfg.inner_size,
                "outer_margin": cfg.outer_margin,
                "inner_circularity": cfg.inner_circularity,
                "outer_circularity": cfg.outer_circularity,
                "momentum_prob": cfg.momentum_prob,
                "shift_weights": cfg.shift_weights,
                "size_mutate_prob": cfg.size_mutate_prob,
                "size_range": cfg.size_range,
                "circ_mutate_prob": cfg.circ_mutate_prob,
                "waypoint_reached_dist": cfg.waypoint_reached_dist,
                "fade_steps": cfg.fade_steps,
                "fade_min_size": cfg.fade_min_size,
            },
            "width": w,
            "height": h,
            "seed": (i + 1) * 42,  # deterministic initial seeds
            "entry_x": w // 2,
            "exit_x": w // 2,
            "grid": None,
            "blueprint": None,
            "validation": None,
            "status": "pending",
            "retry_count": 0,
            "param_retry_count": 0,
        })

    return {"segments": segments, "current_segment_index": 0}


def walker_node(state: PipelineState) -> dict:
    """Generate the game layer grid for the current segment."""
    idx = state["current_segment_index"]
    seg = state["segments"][idx]

    cfg = WalkerConfig(**seg["walker_config"])
    grid, waypoints = generate_segment_grid(
        width=seg["width"],
        height=seg["height"],
        config=cfg,
        seed=seg["seed"],
        entry_x=seg["entry_x"],
        exit_x=seg["exit_x"],
    )

    # Build entry/exit openings
    entry = Opening(side="top", x=seg["entry_x"] - 2, y=0, width=5)
    exit_ = Opening(side="bottom", x=seg["exit_x"] - 2, y=0, width=5)
    _carve_opening(grid, entry)
    _carve_opening(grid, exit_)

    # Store blueprint info for assembly
    bp_dict = {
        "width": seg["width"],
        "height": seg["height"],
        "entry": {"side": "top", "x": seg["entry_x"] - 2, "y": 0, "width": 5},
        "exit": {"side": "bottom", "x": seg["exit_x"] - 2, "y": 0, "width": 5},
    }

    # Update segment record
    segments = list(state["segments"])
    segments[idx] = {**seg, "grid": grid, "blueprint": bp_dict, "status": "generated"}
    return {"segments": segments}


def postprocess_node(state: PipelineState) -> dict:
    """Apply post-processing passes to the current segment."""
    idx = state["current_segment_index"]
    seg = state["segments"][idx]
    grid = seg["grid"]

    bp = seg["blueprint"]
    entry = Opening(**bp["entry"])
    exit_ = Opening(**bp["exit"])

    widen_narrow_passages(grid, min_width=4)
    grid = fix_edge_bugs(grid)
    grid = remove_freeze_blobs(grid, entry, exit_)

    segments = list(state["segments"])
    segments[idx] = {**seg, "grid": grid, "status": "postprocessed"}
    return {"segments": segments}


def validate_node(state: PipelineState) -> dict:
    """BFS connectivity check on the current segment."""
    idx = state["current_segment_index"]
    seg = state["segments"][idx]
    grid = seg["grid"]

    bp = seg["blueprint"]
    entry = Opening(**bp["entry"])
    exit_ = Opening(**bp["exit"])

    result = validate_segment(grid, entry, exit_)

    status = "valid" if result.playable and result.reachable_pct >= 50.0 else "failed"

    segments = list(state["segments"])
    segments[idx] = {
        **seg,
        "validation": {
            "playable": result.playable,
            "reachable_pct": result.reachable_pct,
            "island_count": result.island_count,
        },
        "status": status,
    }

    label = seg["cluster_type"][:30]
    print(f"  Seg {idx + 1} ({label}): reach={result.reachable_pct:.0f}% "
          f"{'OK' if status == 'valid' else 'FAILED'}")

    return {"segments": segments}


def retry_decision(state: PipelineState) -> str:
    """Route based on validation result and retry counts."""
    idx = state["current_segment_index"]
    seg = state["segments"][idx]

    if seg["status"] == "valid":
        return "valid"

    max_seed = state.get("max_seed_retries", 3)
    max_param = state.get("max_param_retries", 2)

    if seg["retry_count"] < max_seed:
        return "retry_seed"
    if seg["param_retry_count"] < max_param:
        return "retry_params"
    return "failed"


def retry_seed_node(state: PipelineState) -> dict:
    """Bump seed and retry count for current segment."""
    idx = state["current_segment_index"]
    seg = state["segments"][idx]

    segments = list(state["segments"])
    segments[idx] = {
        **seg,
        "seed": seg["seed"] + 1000,
        "retry_count": seg["retry_count"] + 1,
        "status": "pending",
    }
    return {"segments": segments}


def retry_params_node(state: PipelineState) -> dict:
    """Widen walker params and reset seed retries."""
    idx = state["current_segment_index"]
    seg = state["segments"][idx]

    cfg = dict(seg["walker_config"])
    cfg["inner_size"] = min(cfg["inner_size"] + 1, 9)
    sr = cfg["size_range"]
    cfg["size_range"] = (min(sr[0] + 1, sr[1]), sr[1] + 1)

    segments = list(state["segments"])
    segments[idx] = {
        **seg,
        "walker_config": cfg,
        "seed": seg["seed"] + 5000,
        "retry_count": 0,
        "param_retry_count": seg["param_retry_count"] + 1,
        "status": "pending",
    }
    print(f"  Widening params: inner_size={cfg['inner_size']}")
    return {"segments": segments}


def fallback_bridge_node(state: PipelineState) -> dict:
    """Last resort: bridge gaps in the current segment."""
    from .bfs import opening_tiles, bridge_gaps

    idx = state["current_segment_index"]
    seg = state["segments"][idx]
    grid = seg["grid"]

    bp = seg["blueprint"]
    entry = Opening(**bp["entry"])
    exit_ = Opening(**bp["exit"])

    entry_tiles = opening_tiles(entry, grid.shape[0], grid.shape[1])
    exit_tiles = opening_tiles(exit_, grid.shape[0], grid.shape[1])
    bridge_gaps(grid, entry_tiles, exit_tiles)

    fix_edge_bugs(grid)

    segments = list(state["segments"])
    segments[idx] = {**seg, "grid": grid, "status": "valid"}
    print(f"  Seg {idx + 1}: bridged gaps (fallback)")
    return {"segments": segments}


def advance_segment_node(state: PipelineState) -> dict:
    """Move to next segment. Propagate exit_x → next entry_x."""
    idx = state["current_segment_index"]
    next_idx = idx + 1
    segments = list(state["segments"])

    # Propagate exit position to next segment's entry
    if next_idx < len(segments):
        current_exit_x = segments[idx]["exit_x"]
        segments[next_idx] = {**segments[next_idx], "entry_x": current_exit_x}

    return {"segments": segments, "current_segment_index": next_idx}


def check_segments_done(state: PipelineState) -> str:
    """Check if all segments are processed."""
    if state["current_segment_index"] >= len(state["segments"]):
        return "done"
    return "continue"


def assemble_node(state: PipelineState) -> dict:
    """Stitch all segments into a full map."""
    bp_grid_pairs = []
    for seg in state["segments"]:
        bp_data = seg["blueprint"]
        entry = Opening(**bp_data["entry"])
        exit_ = Opening(**bp_data["exit"])
        cp = CheckpointSpec(x=seg["width"] // 2 - 3, y=seg["height"] // 2, width=6)

        bp = Blueprint(
            width=seg["width"],
            height=seg["height"],
            difficulty=seg["difficulty"],
            entry=entry,
            exit=exit_,
            checkpoint=cp,
            obstacles=[],
        )
        bp_grid_pairs.append((bp, seg["grid"]))

    full_grid, entities = stitch_segments(bp_grid_pairs)
    return {"assembled_grid": full_grid, "entities": entities}


def export_node(state: PipelineState) -> dict:
    """Write the final .map file."""
    output_path = state.get("output_path") or "maps/output/generated.map"
    path = write_map(
        state["assembled_grid"],
        state["entities"],
        output_path,
    )
    return {"output_path": str(path)}


# ── Graph wiring ─────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """Build and compile the LangGraph pipeline."""
    graph = StateGraph(PipelineState)

    # Add nodes
    graph.add_node("plan", plan_node)
    graph.add_node("walker", walker_node)
    graph.add_node("postprocess", postprocess_node)
    graph.add_node("validate", validate_node)
    graph.add_node("retry_seed", retry_seed_node)
    graph.add_node("retry_params", retry_params_node)
    graph.add_node("fallback_bridge", fallback_bridge_node)
    graph.add_node("advance_segment", advance_segment_node)
    graph.add_node("assemble", assemble_node)
    graph.add_node("export", export_node)

    # Linear edges
    graph.add_edge("plan", "walker")
    graph.add_edge("walker", "postprocess")
    graph.add_edge("postprocess", "validate")

    # Retry decision after validate
    graph.add_conditional_edges(
        "validate",
        retry_decision,
        {
            "valid": "advance_segment",
            "retry_seed": "retry_seed",
            "retry_params": "retry_params",
            "failed": "fallback_bridge",
        },
    )

    # Retry loops back to walker
    graph.add_edge("retry_seed", "walker")
    graph.add_edge("retry_params", "walker")

    # Fallback → advance (segment is "good enough")
    graph.add_edge("fallback_bridge", "advance_segment")

    # Segment loop
    graph.add_conditional_edges(
        "advance_segment",
        check_segments_done,
        {
            "continue": "walker",
            "done": "assemble",
        },
    )

    # Post-assembly
    graph.add_edge("assemble", "export")
    graph.add_edge("export", END)

    # Entry point
    graph.set_entry_point("plan")

    return graph


# ── Public API ───────────────────────────────────────────────────

def run_pipeline(
    difficulty: str = "medium",
    n_segments: int = 5,
    theme: str = "grass",
    challenge_sequence: list[str] | None = None,
    output_path: str = "maps/output/generated.map",
    copy_to_teeworlds: bool = True,
) -> dict:
    """Run the full generation pipeline.

    Args:
        difficulty: overall difficulty ("easy", "medium", "hard")
        n_segments: number of segments
        theme: visual theme (for automapper, not yet implemented)
        challenge_sequence: explicit cluster labels per segment.
            If None, uses a default mixed sequence.
        output_path: where to save the .map file
        copy_to_teeworlds: copy to Teeworlds maps folder

    Returns:
        Final pipeline state dict.
    """
    # Default challenge sequence if none provided
    if challenge_sequence is None:
        default_types = [
            "Winding freeze corridor ascent",
            "high air zigzag traverse",
            "tight solid corridor traversal",
            "Winding freeze corridor descent",
            "open air freeze zigzag descent",
        ]
        challenge_sequence = [
            default_types[i % len(default_types)]
            for i in range(n_segments)
        ]

    # Build difficulty progression
    if difficulty == "easy":
        progression = ["easy"] * n_segments
    elif difficulty == "hard":
        progression = ["medium"] * (n_segments // 2) + ["hard"] * (n_segments - n_segments // 2)
    else:
        # Medium: start easy, end medium
        progression = ["easy"] * (n_segments // 3 + 1) + ["medium"] * (n_segments - n_segments // 3 - 1)
    progression = progression[:n_segments]

    initial_state: PipelineState = {
        "user_difficulty": difficulty,
        "user_n_segments": n_segments,
        "user_theme": theme,
        "challenge_sequence": challenge_sequence,
        "difficulty_progression": progression,
        "visual_theme": theme,
        "segments": [],
        "current_segment_index": 0,
        "output_path": output_path,
        "max_seed_retries": 3,
        "max_param_retries": 2,
    }

    print(f"Generating {n_segments}-segment {difficulty} map...")
    print(f"Sequence: {[s[:25] + '...' if len(s) > 25 else s for s in challenge_sequence]}")
    t0 = time.time()

    graph = build_graph()
    compiled = graph.compile()
    result = compiled.invoke(initial_state)

    elapsed = time.time() - t0
    print(f"\nPipeline complete in {elapsed:.1f}s")

    # Copy to Teeworlds maps folder
    if copy_to_teeworlds:
        tw_maps = Path(r"C:\Users\sh121\AppData\Roaming\Teeworlds\maps")
        if tw_maps.exists():
            dest = tw_maps / "TeeHaven_Generated.map"
            shutil.copy(result["output_path"], str(dest))
            print(f"Copied to: {dest}")

    return result
