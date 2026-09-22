# TPU-Distil

**Simple, reusable on-policy agentic distillation (`SCoRe-RL`) on Cloud TPU `v6e`.**

`TPU-Distil` transfers multi-turn execution and error-recovery capabilities from frontier large MoE teachers (**`DeepSeek-V4.1-Flash`**, **`Kimi-K3`**, or **`GLM-5.3`**) into a single-host TPU student (**`Qwen3-Next-80B-A3B-Instruct`**, `80B` total / `3B` active parameters) using **Self-Correction via Reinforcement Learning** ([arXiv:2509.14257v3](https://arxiv.org/abs/2509.14257)).

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full design, Teacher–Student `Terminal-Bench` delta table, and 4-wave pass gates.

## Core 4-File Pipeline (`src/tpu_distil/`)

1. `src/tpu_distil/trajectory.py`: Cross-tokenizer safe `Step(thought, action, observation)` schema with `0.0` loss masking on `observation` tokens and `256`-aligned TPU `v6e` padding.
2. `src/tpu_distil/sandbox.py`: Container sandbox worker pool (`Terminal-Bench`, `R2E-Gym`, `SWE-Gym`, `Tool-Star`).
3. `src/tpu_distil/splicer.py`: Mixed-Policy Splicing (`MPS`) — runs student on TPU `v6e` to first execution failure $s_m$, then splices in the Teacher (`DeepSeek-V4.1-Flash` / `Kimi-K3` / `GLM-5.3`) recovery suffix.
4. `src/tpu_distil/score_reward.py`: Process-reward shaper and `MaxText` LoRA SFT + `Tunix` short-horizon (`H_rem=4` steps from $s_m$) LoRA `GRPO` driver on TPU `v6e`.
