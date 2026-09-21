# REVIEW.md — Pre-Merge Review Passes

Run each pass independently against the diff before closing a wave. Rank every finding **Important** (blocks merge) or **nit** (non-blocking, max 5).

## Pass 1: Gate Verification
Quote the wave's gate sentence verbatim. Point to the exact test name, log line, or benchmark measurement for each numbered clause and score `GREEN` or `RED`. Any partial clause is `RED`.

## Pass 2: Observation Masking & Trajectory Splicing Integrity
Verify that:
1. Environment `Observation` tokens receive `0.0` loss weight in all SFT data loaders.
2. Teacher corrections $\sigma'_k$ replace only step $k$ while preserving student prefix $(\sigma_1, \dots, \sigma_{k-1})$ byte-for-byte.
3. Short-horizon RL rollouts start strictly at step $k$ after $P_{k-1}$.

## Pass 3: TPU v6e Shape & Recompilation Hygiene
Verify that sequence padding lengths and batch dimensions are multiples of `256` (matching the TPU `v6e` MXU systolic array) and that variable-length prefixes in short-horizon GRPO are bucketed to fixed sequence lengths (`2048`, `4096`, `8192`, `16384`, `32768`) to prevent XLA JIT thrashing.

## Pass 4: Sandbox Isolation & Negative Controls
Verify that `CodeAct` tool execution enforces timeout and resource bounds, and that the wave's negative control fixture fails as expected when the target mechanism (`R_key` bonus or teacher step splicing) is disabled.

## Pass 5: Evergreen Prose & Clean Tree
Verify that `git status` shows zero untracked working logs (`.agents/`, `intent/`, `specs/`, `plans/` remain git-ignored) and that tracked code/docs contain no wave numbers or investigation notes.
