# AGENTS.md — Codebase Invariants & Engineering Guardrails

1. **Evergreen Tracked Code**: Source files (`src/`, `tests/`, `docs/`) must never contain wave numbers, temporary hypothesis labels, or single-host narrative logs. Operational wave records live exclusively in `.agents/`, `intent/`, `specs/`, and `plans/` (all git-ignored).
2. **Cross-Tokenizer Safety**: Because teachers (`DeepSeek-V3.1-Terminus`, `GLM-5`, `Kimi-K3`) and the student (`Qwen3-30B-A3B-Instruct-2507` / `Qwen3`) use different tokenizers, `MPS` step-splicing must always operate on structured `Step(thought, action, observation)` objects and serialize to `Qwen3` token IDs only inside the student rollout / MaxText / Tunix boundary.
3. **Observation Loss Masking**: Every SFT dataset builder (`BC-Init` and `SCoRe-SFT`) must assert that environment observation tokens (`observation`) have loss weight `0.0`. Training on tool stdout/stderr degrades reasoning stability.
4. **TPU Dimension Alignment**: Tensor batch sizes, sequence padding lengths, and LoRA/projection ranks on TPU `v6e` must be multiples of `256` to match the 256x256 MXU systolic array and avoid padding waste or XLA recompilation.
5. **Cloud Resource Naming**: Any cloud VM, TPU slice, or GKE resource created by scripts must follow `<resource>-jawadamin-<region>-MMYY` (e.g., `tpu-v6e-jawadamin-us-east5-0926`).
6. **Per-Wave `concepts.md` (Git-Ignored)**: Every wave `N` must publish `.agents/wave-N/concepts.md` (kept untracked by git) explaining the core mathematical, algorithmic, and hardware concepts leveraged in that wave.

