# Failed Approaches

This project went through 6 major architecture iterations before arriving at the current walker-based approach. Each failed attempt taught something that shaped the next one. This document records what was tried, why it didn't work, and what was learned.

## Context: What Makes Gores Hard to Generate

Gores is a DDNet/Teeworlds game mode with strict constraints:
- Maps are mostly **solid** (hookable) and **freeze** (dangerous) tiles with narrow **air** passages
- Players navigate using hooking (grappling hook, ~12 tile range), jumping (~5 tile height), and momentum
- Touching freeze tiles freezes the player (effectively death in race context)
- The structure follows strict layering: SOLID -> FREEZE -> AIR
- Maps must be traversable from spawn to finish with these movement mechanics
- Real Gores maps have organic, winding corridors -- not rectangular rooms

## V1: LLM Outputs Obstacle Coordinates

**Approach**: Ask GPT-4o to output a JSON blueprint with positioned obstacles: `{type: "platform", x: 10, y: 20, width: 8, height: 3}`. A builder stamps each obstacle onto a solid grid.

**Result**: 0% first-attempt pass rate. Every generated map was unplayable.

**Why it failed**: LLMs cannot do spatial math. They don't understand that obstacle A at (10, 20) blocks the path from (5, 15) to (15, 25). Coordinates were essentially random -- obstacles overlapped, gaps were either trivially small or impossibly large, and there was no guarantee of a connected path from entry to exit.

The LLM would output coordinates that looked reasonable as numbers but were spatially nonsensical. A "platform at x=30" means nothing to a model that can't visualize what x=30 looks like relative to the rest of the map.

**Lesson**: Don't ask LLMs to reason about spatial coordinates. Their strength is creative and semantic reasoning, not geometry.

## V2: LLM Outputs Sequence, Chain Engine Positions

**Approach**: Remove coordinate responsibility from the LLM entirely. The LLM outputs an ordered sequence of obstacle types with parameters (no coordinates). A chain layout engine computes positions by distributing obstacles along the vertical flow axis with proportional sizing.

**Result**: Maps were playable -- first-attempt pass rate improved dramatically. But every obstacle was a clean rectangular bounding box. Maps looked like stacked rectangles, nothing like real Gores.

**Why it failed**: The chain engine solved the spatial problem correctly: obstacles were properly sized, non-overlapping, and connected. But the geometry was entirely rectangular. Real Gores maps have organic, winding corridors with irregular edges and natural-looking freeze transitions. Rectangles don't feel authentic.

**Lesson**: Correct positioning is necessary but not sufficient. The geometry itself needs to be organic.

## V3: Walker Carves Corridors

**Approach**: Replace bounding-box placement with a probabilistic walker algorithm. The walker carves air passages through solid terrain by stepping between waypoints, applying a kernel that converts solid to air. The LLM still picks the challenge sequence, but the walker handles all spatial work.

**Result**: Dense, organic-looking maps with winding corridors and natural edges. Visually much closer to real Gores maps. But no actual Gores challenges -- just walkable corridors.

**Why it failed**: The walker creates connected air passages by construction. But "connected" is the opposite of "challenging" in Gores. Real Gores challenges exist precisely because the path is **broken** -- there's a gap you must hook across, a ceiling you must swing from, a drop where you must land on a platform. If you can walk through the entire map without hooking, there's no gameplay.

**Lesson**: Connectivity is necessary for playability but antithetical to challenge. Challenges come from intentional discontinuities in the walkable path.

## V4: Challenge Rooms with Discontinuities

**Approach**: Combine the walker (for corridors between rooms) with dedicated room builders that create intentional discontinuities. Four room types were implemented:
- **hook_swing**: air gap wider than jump range (~6 tiles) with hookable ceiling
- **platform_drop**: vertical descent with staggered platforms requiring precise landing
- **freeze_gauntlet**: narrow safe path (1-2 tiles) through freeze walls
- **hook_chain**: sequential hook points in zigzag pattern

**Result**: Mechanically correct -- the rooms did force specific movement skills. But the geometry was clean rectangles inside bounding boxes. Massive air spaces with tiny features floating in them. Didn't look or feel like real Gores.

**Why it failed**: Hand-coded builders filled their bounding boxes with air and placed a few features. The result was big empty rooms connected by corridors. In real Gores maps, solid and freeze dominate (60-70% of tiles) with air as the minority (~30%). Our rooms had the inverse ratio.

The deeper issue: we were trying to hand-code what "hook_swing" looks like in tiles. But we don't have the aesthetic intuition that human map makers have. Our hook swings were geometrically correct but visually wrong.

**Lesson**: Hand-coding tile geometry for challenge types doesn't scale. The builders don't understand Gores aesthetics -- only the data from real maps does.

### Template Stamping (sub-experiment)

As part of V4, we tried cropping real challenge patterns from analyzed maps and stamping them as templates. Built a pattern library with ~15 examples per challenge type.

**Result**: Individual templates looked authentic. But when placed inside rectangular bounding boxes, they appeared as isolated islands with visible seams. The surrounding context (how a pattern transitions to solid/freeze terrain) was lost.

**Lesson**: Isolated patterns without context don't compose well. The transition between challenge and terrain is as important as the challenge itself.

### Real Segment Stitching (exploration)

Also explored: skip generation entirely, pick 4-5 real segments from analyzed maps, standardize their width, force entry/exit openings, and assemble into a map.

**Result**: Looked authentic -- because it WAS real map data. But this isn't generation; it's compilation. No creative value, no novelty.

**Why it mattered**: This experiment proved that the quality standard exists in our data. The 47K analyzed segments contain everything needed -- the gap was in our generation approach, not our data pipeline.

**Lesson**: The data IS the quality standard. Any generation approach must learn from it.

## Conditional VAE (ML Experiment)

**Approach**: Train a Conditional Variational Autoencoder on the segment data. Architecture:
- 3M parameters
- 36,867 training segments resized to 64x64
- 6 tile categories as one-hot channels (6, 64, 64)
- Conditioned on: openness score, flow direction, has_freeze/death/nohook flags
- Encoder: 4 conv layers (6->32->64->128->256), batch norm, ReLU
- Latent dim: 128
- Decoder: mirror of encoder with transposed convolutions
- Loss: cross-entropy reconstruction + beta-KL (beta=0.1)
- Training: 200 epochs, Adam lr=1e-3, batch 64, GTX 1080

**Result**: 78.7% per-tile accuracy. Output had correct tile statistics (right freeze-to-solid ratio, freeze neighboring solid) but was structurally random. No connected corridors, no coherent passages. The model learned local patterns but not global structure.

**Why it failed**:

1. **22% error rate is catastrophic for spatial coherence.** 78% accuracy sounds decent, but it means ~900 wrong tiles per 64x64 grid. A single misplaced solid tile can block an entire corridor. Spatial tasks have no tolerance for random errors.

2. **Resizing destroyed structural information.** Real segments vary wildly in size (20x15 to 200x200+). Squeezing them all to 64x64 turned corridors into noise. A 200x30 horizontal corridor becomes a smeared 64x64 patch where the corridor structure is unrecognizable.

3. **VAEs learn distributions, not structure.** The model learned "freeze tiles tend to neighbor solid tiles" (a local pattern) but not "corridors wind continuously from entry to exit" (a global structural property). VAEs are fundamentally distribution-matching models; spatial coherence requires something more.

4. **Insufficient capacity for the diversity.** 36K segments spanning dozens of difficulty levels, flow patterns, and challenge types. A 3M parameter model can't capture that diversity -- it learns the average, which is nothing recognizable.

**Lesson**: Pure ML tile generation for this problem needs either much more capacity (large diffusion models, transformers) or a fundamentally different formulation. The spatial coherence problem is harder than the statistical matching problem. Generating tiles that look like Gores on average is easy; generating tiles that form playable corridors is hard.

The VAE experiment is preserved on the `experiment/vae` branch with full training code and checkpoints.

## V5: Walker + Clusters + LLM Planning (Current)

**What changed**: Instead of trying to generate tiles from scratch (whether by hand-coded builders, templates, or ML), use a probabilistic walker algorithm that guarantees connectivity by construction. The walker's parameters (passage width, freeze thickness, path straightness) are calibrated from cluster statistics of real maps. The LLM selects from this data-driven vocabulary.

**Why it works**:
1. **Walker guarantees connected passages** -- no broken paths, no unplayable maps
2. **Two-kernel system guarantees freeze borders** -- SOLID->FREEZE->AIR layering by construction
3. **Parameters calibrated from real data** -- authentic proportions and density
4. **LLM stays in its strength zone** -- creative sequencing, not spatial math
5. **Post-processing handles edge cases** -- narrow passages, edge bugs, freeze blobs

**What's still missing**: Passages don't force specific Gores mechanics (hooking, swinging). Maps are playable and authentic-looking, but lack intentional challenge design. This is the next development frontier.

## Key Takeaways

1. **Separate creative decisions from spatial execution.** LLMs design the experience; algorithms build the geometry. Every attempt to mix these responsibilities failed.

2. **Real map data is the quality standard.** Approaches not grounded in analyzed data produce unrealistic output. The 47K segment corpus is the project's most valuable asset.

3. **Connectivity by construction beats validation by checking.** The walker's two-kernel guarantee eliminates an entire class of bugs that plagued V1-V4.

4. **Fail forward.** Each iteration taught something specific:
   - V1: LLMs can't do spatial math
   - V2: correct positioning doesn't mean organic geometry
   - V3: connectivity alone isn't gameplay
   - V4: hand-coded builders can't match real aesthetics
   - VAE: pure ML generation needs fundamentally more capacity
   - V5: let algorithms handle spatial work, let data handle aesthetics
