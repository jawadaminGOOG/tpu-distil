# TPU-Distil: On-Policy Agentic Self-Correction (`SCoRe-RL`) on Cloud TPU `v6e`

![TPU-Distil Visual Research Poster](docs/tpu_distil_poster.png)

**Simple, reusable on-policy agentic distillation (`SCoRe-RL`) on Google Cloud TPU `v6e` (Trillium).**

`TPU-Distil` transfers multi-turn terminal execution and error-recovery capabilities from a frontier Mixture-of-Experts (MoE) teacher (**`deepseek-ai/DeepSeek-V4.1-Flash`**, `552B` total / `8B` prefill, `16B` decode active parameters served with `mxfp4` fused Megablox GMM `EP8/TP2` on `TPU v6e-16`) into three compact Cloud TPU `v6e` student architectures using **Self-Correction via Reinforcement Learning** ([arXiv:2509.14257v3](https://arxiv.org/abs/2509.14257)):
- **`Qwen/Qwen3-30B-A3B-Instruct-2507`** (`48`-layer Sparse MoE, `30.5B` total / `3.3B` active parameters, `128` experts/layer, top-8 routing)
- **`Qwen/Qwen3.6-27B-FP8`** (`64`-layer Dense Hybrid `27B`, `48` Gated-DeltaNet + `16` GQA layers, exact `256x256` TPU MXU alignment: `d_model=5120=20x256`, `d_ff=17408=68x256`, `head_dim=256`, `vocab=248320=970x256`)
- **`Qwen/Qwen3.8-27B-FP8`** (`64`-layer Dense Hybrid `27B`, `48` Gated-DeltaNet + `16` GQA layers, exact `256x256` TPU MXU alignment: `d_model=5120=20x256`, `d_ff=17408=68x256`, `head_dim=256`, `vocab=248320=970x256`)

- **[Practitioner Technical Note (`docs/TECHNICAL_NOTE.md`)](docs/TECHNICAL_NOTE.md):** Full methodology, mathematical formulation, `Terminus-2` Teacher parity analysis (`80.0%`), `Terminal-Gym-4K` zero-leakage decontamination, `200`-step `SCoRe-RL` 6-metric convergence telemetry, and 3-student TPU `v6e-8` hardware/accuracy comparison.
- **[System Architecture (`docs/ARCHITECTURE.md`)](docs/ARCHITECTURE.md):** 5-module engine design, hardware topology, and engineering guardrails.
- **[Visual Research Poster (`docs/tpu_distil_poster.svg` / `.png`)](docs/tpu_distil_poster.svg):** High-resolution vector and raster summary poster.

---

## Headline Results on Held-Out `Terminal-Bench` (`50` Tasks, Zero Oracle Feedback)

### 1. Public-Parity `Terminus-2` Teacher & 3-Student `200`-Step `SCoRe-RL` Comparison (`Terminal-Gym-4K` Regime)

Trained on the zero-leakage **`Terminal-Gym-4K`** corpus (`4,096` container-verified multi-domain trajectories across 6 terminal engineering domains with `0%` `Terminal-Bench` overlap; `max_observed_ngram_jaccard = 0.003597 < 0.25`) and **`200` online `GRPO` steps (`4,096` rollouts)** on an 8-chip **Cloud TPU `v6e-8`** slice:

| Model / Student Architecture | Active / Total Params | TPU `v6e` MXU Alignment | `Zero-Shot` Pass@1 | `BC-Control` Pass@1 | `SCoRe-SFT` Pass@1 | `200`-Step `SCoRe-RL` Pass@1 | Gain vs. `Zero-Shot` (`95% CI`) | TPU `v6e` MXU Util | Peak HBM / Chip |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Qwen/Qwen3-30B-A3B-Instruct-2507`** *(Sparse MoE)* | `3.3B` / `30.5B` | Grouped Ragged Dot (`d_model=2048`, `d_moe=768`) | `16.0%` (`8/50`) | `28.0%` (`14/50`) | `40.0%` (`20/50`) | **`54.0%` (`27/50`)** | **`+38.0%`** (`[+24%, +52%]`, $p<10^{-4}$) | `21.8%` | `14.51 GB` |
| **`Qwen/Qwen3.6-27B-FP8`** *(Dense Hybrid 27B)* | `27.0B` / `27.0B` | Exact `256x256` Systolic (`5120`, `17408`, `head=256`) | `16.0%` (`8/50`) | `38.0%` (`19/50`) | `48.0%` (`24/50`) | **`60.0%` (`30/50`)** | **`+44.0%`** (`[+30%, +58%]`, $p<10^{-4}$) | **`49.4%`** | `5.15 GB` |
| **`Qwen/Qwen3.8-27B-FP8`** *(Dense Hybrid 27B)* | `27.0B` / `27.0B` | Exact `256x256` Systolic (`5120`, `17408`, `head=256`) | `16.0%` (`8/50`) | `38.0%` (`19/50`) | `50.0%` (`25/50`) | **`62.0%` (`31/50`)** | **`+46.0%`** (`[+32%, +60%]`, $p<10^{-4}$) | **`51.2%`** | `5.16 GB` |
| *Teacher (`deepseek-ai/DeepSeek-V4.1-Flash`)* | *`8B/16B` / `552B`* | *`mxfp4` Megablox GMM (`EP8/TP2` on `v6e-16`)* | ***`80.0%` (`40/50`)*** | — | — | — | *`+5.5%` vs `74.5%` public `Terminus-2`* | — | `17.50 GB` |

### 2. Controlled 2-Turn `MPS` + Short-Horizon `GRPO` Ablation (`Qwen/Qwen3-30B-A3B-Instruct-2507`)

Under the strict 2-turn single-correction harness (`max_turns=2`, `2,048` `MBPP`/`HumanEval`/`KodCode-V1` `MPS` trajectories + `36` short-horizon `GRPO` steps):

| Evaluation Arm | Model Checkpoint | Cloud TPU Slice | Overall `Pass@1` (`50` Tasks) | Gain vs. `Zero-Shot` | Wrong $\to$ Right $P(\text{pass}_1 \mid \text{fail}_0)$ | Peak HBM / Chip |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`Zero-Shot` (Base Student)** | `Qwen/Qwen3-30B-A3B-Instruct-2507` | `v6e-4` | **`16.0%`** (`8 / 50`) | *Baseline (`0.0%`)* | **`2.33%`** (`1 / 43`) | `11.13 GB` |
| **`BC-Control` (Stage 1 Pure Teacher SFT)** | `Qwen3-30B-A3B` + `Rank-64 LoRA` | `v6e-8` | **`18.0%`** (`9 / 50`) | `+2.0%` (`+1` task) | **`4.65%`** (`2 / 43`) | `14.51 GB` |
| **`SCoRe-SFT` (Stage 1 First-Error Spliced `MPS`)** | `Qwen3-30B-A3B` + `Rank-64 LoRA` | `v6e-8` | **`24.0%`** (`12 / 50`) | `+8.0%` (`+4` tasks) | **`11.63%`** (`5 / 43`) | `14.51 GB` |
| **`SCoRe-RL` (Stage 1 `SCoRe-SFT` + Stage 2 `GRPO`)** | `Qwen3-30B-A3B` + `Rank-64 LoRA` | `v6e-8` | **`34.0%`** (`17 / 50`) | **`+18.0%` (`p = 0.0001`)** | **`23.26%` (`+18.61%` vs. `BC`, `p = 0.0002`)** | **`14.51 GB` (`0` OOMs)** |
| *2-Turn Teacher Baseline* | `deepseek-ai/DeepSeek-V4.1-Flash` | `v6e-16` | *`64.0%` (`32 / 50`)* | *`+48.0%` headroom* | *`43.75%` (`14 / 32`)* | — |

---

## Core Pipeline Modules (`src/tpu_distil/`)

1. **[`src/tpu_distil/trajectory.py`](src/tpu_distil/trajectory.py):** Cross-tokenizer safe `Step(thought, action, observation)` and `Trajectory` schema with strict `0.0` loss masking on `observation` (`stdout`/`stderr`), prompt, and pre-splice prefix tokens, plus `256`-multiple sequence padding for the TPU `v6e` `256x256` MXU systolic array.
2. **[`src/tpu_distil/sandbox.py`](src/tpu_distil/sandbox.py):** Rootless container sandbox worker (`bwrap` / `podman` / subprocess isolation) with `git diff` error-state snapshotting (`state_s_m_diff`), oracle-free execution (`include_test_feedback=False`), and held-out `pytest` verification.
3. **[`src/tpu_distil/splicer.py`](src/tpu_distil/splicer.py):** Mixed-Policy Splicing (`MPS`) — runs the student on TPU `v6e-4` up to its first execution failure $s_m$ (`exit_code != 0`), then prompts the teacher (`deepseek-ai/DeepSeek-V4.1-Flash`) on TPU `v6e-16` to generate a container-verified recovery suffix $(a_m^*, o_m^*, \dots, a_T^*)$.
4. **[`src/tpu_distil/score_reward.py`](src/tpu_distil/score_reward.py):** Hybrid SCoRe process-reward shaper, group-relative advantage normalizer (`compute_grpo_advantages`), and TPU `v6e` `LoRAConfig` (`rank=64/256, freeze_moe_router=True`).
5. **[`src/tpu_distil/rl_telemetry.py`](src/tpu_distil/rl_telemetry.py):** Six-metric RL convergence telemetry and benchmark decontamination (`compute_k3_kl_divergence`, `filter_informative_grpo_groups`, `compute_reward_smoothness_metrics`, `evaluate_rl_health_corridor`, and 3-layer `verify_zero_benchmark_leakage`).

---

## Quickstart & Verification

Run the complete unit and dataset/checkpoint invariant test suite (`22` tests covering sandbox execution, observation loss masking, 256-token MXU alignment, cross-tokenizer splicing, RL telemetry corridors, zero-leakage `Terminal-Gym-4K` decontamination, and 3-student `Terminal-Bench` evaluation gates):

```bash
PYTHONPATH=src pytest tests/ -v
```
