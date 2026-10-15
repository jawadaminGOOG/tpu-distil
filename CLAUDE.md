# CLAUDE.md — TPU-Distil

Read `README.md` first: it states the project thesis and the current result. `AGENTS.md` holds the codebase invariants.

## Goal (the only thing this repo has to prove)
Distilling a strong teacher into a small student that already runs well on TPU v6e (1) raises the student's quality significantly in a domain where it was weak and (2) serves at a small fraction of the teacher's cost. Teacher, benchmark and student are interchangeable. Every claim needs a same-run paired comparison with a 95% CI, a self-distillation control, a measured noise floor, measured serving cost, and a script that regenerates the number from raw logs. Prefer the smallest experiment that can show this; add method complexity only after plain distillation is measured.

## Commands
- Tests (≈2 min, CPU): `cd <repo> && /tmp/venv_jax011/bin/python -m pytest tests -q` → green = all pass. Tests that need the Qwen3 tokenizer in the local Hugging Face cache, or the reference checkout for `tests/test_trainer_equivalence.py` (`$TPU_DISTIL_REFERENCE_REPO`), skip without them. The `distil/` tests need only pytest.
- Serve a small model on one v6e chip (TPU VM): `distil/serve_small.sh <chip> <model_dir> <name>` → port 8100+chip. Gemma 4: `MODEL_IMPL_TYPE=flax_nnx EXTRA_ARGS="--chat-template-content-format string"` (first start compiles ~17 min per model directory).
- Generate / grade / compare (harness VM, stdlib only): `distil/{generate,grade,compare}.py`. Task-specific scripts take `--task bird|finqa|math` (default `bird`); BIRD data at `~/kd/bird`, FinQA at `~/kd/finqa`, MATH (`train.jsonl`, `test.jsonl`) at `~/kd/math` on the harness VM.
- Build training data: `distil/build_sft.py` (SFT records), `distil/build_grpo_rollouts.py` (GRPO records; `--keep-truncated` keeps rollouts cut off at the length limit as reward 0); audit with `distil/leak_audit.py`.
- Train SFT (TPU VM, all 8 chips, replicas stopped; the student family comes from the checkpoint's `config.json`): `MODEL=Qwen3-1.7B EVAL_TASKS=<dev ids json> trainer/train_tpu_vm.sh sft <run> <sft.jsonl> --global_batch 256 --lr 1e-4 --warmup_frac 0.05 --epochs 3 --len_buckets 2048,4096,8192` (≈20 min for 10k BIRD examples on Qwen3-1.7B). Defaults: FSDP 8, TP 1, MICRO_BATCH 8, MAX_SEQ_LEN 8192, RANK 256 (alpha 512).
- One GRPO iteration: `MODEL=Qwen3-1.7B REF=<KL reference HF dir> EVAL_TASKS=<dev ids json> trainer/train_tpu_vm.sh grpo <run> <policy HF dir> <rollouts.jsonl> <k> --overlong drop --grpo_len_buckets 2048,4096,8192 --mini_batch 64 --beta 0.02 --lr 2e-6 --allow_unaligned_rollouts`, with `trainer/kl_guard.sh $KD_DATA_DIR/train/<run>/metrics/grpo_iter_metrics.jsonl kd-train` running beside it (stops the run if a step's KL exceeds 0.2). Rollouts must be sampled at temperature 1.0 and top-p 1.0 from the policy being trained: the trainer recomputes log-probabilities at temperature 1 without importance correction.
- Report: `distil/report.py` (see its docstring) regenerates accuracy, paired CIs and cost per correct answer.

## Hardware
TPU VM `jawadamin-pd-v6e8` (v6e-8, us-east5-a). Scripts read the data disk from `KD_DATA_DIR` (default `/mnt/data`; it holds `models/`, `train/`, `gcs/`, the compilation caches and the repository checkout at `code/`) and the docker daemon from `DOCKER_HOST`; the VM runs an isolated daemon for training and serving, so export `DOCKER_HOST` with its socket path. Harness VM `vm-jawadamin-us-east5-0926`. Teacher: Vertex endpoint (8×H200, us-south1) behind the proxy on the harness VM at `:18201`; it scales to zero (≈25 min cold start).

## Vocabulary
- **EX**: BIRD execution accuracy — predicted and gold SQL return the same set of rows.
- **MATH accuracy**: the last `\boxed{}` equals the gold boxed answer after the reference `strip_string` normalisation (`distil/tasks/hendrycks_math.py`); strict exact match.
- **FinQA accuracy**: the number after the last `Answer:` matches the gold answer within 1% relative (×100 and ÷100 percentage forms accepted; absolute 0.005 when |gold| < 0.5; yes/no exact; `distil/tasks/finqa.py`). More lenient than the official FinQA metric.
- **Group**: what the leak filters key on — the BIRD database or the FinQA report page.
- **KD arm / SELF arm**: student trained on verified teacher samples / on the same number of its own verified samples (self-distillation; no teacher involved). **CORR**: teacher rewrites of the student's failed answers.
- **Two recipes**: KD + GRPO for a student weak in the domain; self-distillation + GRPO with `--keep-truncated` for a student already strong in it (MATH, Gemma 4 E2B).
- **Noise floor**: base vs base on a second replica, same run.
- **train-holdout**: 4 BIRD train databases never trained on, for sanity checks that do not touch dev. FinQA's holdout is its dev split.
- **Record**: one training example, `{"id", "messages", "response", ...}` (`trainer/data.py`); only the response and its `<|im_end|>` carry loss.

## Traps
- `prompt_logprobs` on a vLLM-TPU replica at 0.85 HBM OOMs and kills the engine; use a dedicated replica with `GPU_MEM=0.6`.
- Greedy decoding is not bit-identical across vLLM-TPU replicas (BIRD ≈1 pt; FinQA up to 1.9 pts, identical text on about 55% of questions); always compare arms from the same run.
- Micro batch 16 at 8k tokens OOMs on Qwen3-1.7B; use 8.
- Gemma 4 on vLLM-TPU: without `--chat-template-content-format string` the system message renders as a Python list (check with `/tokenize` against `apply_chat_template`); `prompt_logprobs` kills the engine (multimodal architecture), so run `parity_check.py --positions generated`.
- A rollout driver must stop when any generation request failed: a TPU outage once left half the GRPO rollouts as errors and the iteration trained on the rest.
- TPU host maintenance reboots the VM about daily; restart replicas afterwards (generation clients retry and resume).
- BIRD dev, FinQA test and MATH test are the test sets: never train, tune or select on them.
- The GRPO launcher refuses `REF` equal to `BASE` (the KL reference is normally the SFT-merged policy); set `ALLOW_BASE_REF=1` only on purpose.
