# Compute, data and cost of the distillation runs

This note answers four questions about each run in the README (BIRD first, then [FinQA](#finqa)): how many training FLOPs distillation and RL took, how much accelerator time, how much data, and what it cost. Every number below is produced by `distil/compute_report.py` from the run's raw logs (trainer statistics and timing files, generation logs with per-request token usage, serving benchmarks, the teacher keep-warm log and the Cloud Billing Catalog SKU pages).

## Answers

Student Qwen3-1.7B (1.41B decoder matmul parameters plus a 0.31B tied LM head; LoRA rank 256 on all seven projections adds 0.28B trainable parameters). Teacher DeepSeek-V4.1-Flash on 8×H200. Hardware: one TPU v6e-8 host.

| Question | Recipe (teacher data → KD SFT → 6 GRPO iterations) |
|---|---|
| **Training FLOPs** | **4.2 × 10¹⁸ model FLOPs** (5.5 × 10¹⁸ executed, including rematerialisation). KD SFT 0.55 × 10¹⁸ (13%); the six GRPO iterations 3.7 × 10¹⁸ (87%). Student rollouts add 0.64 × 10¹⁸ inference FLOPs. |
| **Accelerator time** | **41.9 TPU v6e chip-hours**: KD SFT 3.6, GRPO updates 25.8 (about 4.3 per iteration), GRPO rollouts 12.5 (3.5 of which were lost to a replica fault that stalled one round). **5.1 teacher H200-hours** of active generation (38 minutes on 8 GPUs). |
| **Data** | 8,432 BIRD training questions on 64 databases. Teacher: 33,728 answers (15.0M prompt and 6.7M completion tokens), of which 20,133 were correct and 10,112 distinct ones were kept for SFT (61.4M tokens processed over 3 epochs, 5.3M of them trained; the trainer drops 128 trailing examples per epoch, so these counts are about 1.3% high). RL: 6 rounds × 67,456 student rollouts (98M prompt and 84M completion tokens in total), giving 6 × ~16,300 training sequences (231M tokens processed, 21.9M trained). The gold SQL is used only to verify answers (SFT filtering and the RL reward), never as a training target. |
| **Cost** | **$136** at active-compute prices: TPU $113.02 (41.9 chip-hours × $2.70) + teacher $23.44 (5.1 GPU-hours × $4.58); $146 with the teacher at the reservation rate. |

What it took to *prove* the result is reported separately: 12.6 TPU chip-hours ($34) and 9.2 active teacher H200-hours ($42) for the screen, the control and correction arms, the evaluations, benchmarks and the trainer reproduction. A dedicated teacher endpoint bills while idle: over the whole experiment it was up for at least 5.3 hours (42.5 GPU-hours, $194), much of it waiting between phases. A batch job sized to the 38 minutes of teacher generation the recipe needs avoids most of that.

**Utilisation.** Measured model-FLOP utilisation was 3–5% per recipe training run (up to 6% for other runs with a warm compile cache). Wall time includes compilation, data loading and checkpointing, and a 1.7B-parameter model trained with LoRA on long, mostly masked prompts leaves most of a v6e-8 idle. At 40% utilisation the same training would take about 3.2 chip-hours, so the accelerator figures above are an upper bound for this recipe, not its floor.

Per-run detail (generated):

| Run | Role | Tokens processed | Model FLOPs | Wall (s) | Accelerators | Accelerator-hours | MFU |
|---|---|---|---|---|---|---|---|
| kd-q17-kd | recipe | 61,400,622 | 5.54e+17 | 1615 | 8 v6e chip | 3.59 | 4.7% |
| kd-q17-rl-it1 | recipe | 36,980,084 | 4.72e+17 | 2028 | 8 v6e chip | 4.51 | 3.2% |
| kd-q17-rl-it2 | recipe | 37,645,405 | 6.21e+17 | 1935 | 8 v6e chip | 4.30 | 4.4% |
| kd-q17-rl-it3 | recipe | 38,815,499 | 6.42e+17 | 1922 | 8 v6e chip | 4.27 | 4.5% |
| kd-q17-rl-it4 | recipe | 38,626,310 | 6.38e+17 | 1916 | 8 v6e chip | 4.26 | 4.5% |
| kd-q17-rl-it5 | recipe | 39,541,140 | 6.54e+17 | 1905 | 8 v6e chip | 4.23 | 4.7% |
| kd-q17-rl-it6 | recipe | 39,159,313 | 6.48e+17 | 1894 | 8 v6e chip | 4.21 | 4.7% |
| teacher_train_samples | recipe | 21,748,771 | — | 2305 | 8 H200 | 5.12 | — |
| rl_it1_rollouts | recipe | 30,154,618 | 1.05e+17 | 665 | 8 v6e chip | 1.48 | — |
| rl_it2_rollouts | recipe | 30,541,712 | 1.07e+17 | 685 | 8 v6e chip | 1.52 | — |
| rl_it3_rollouts | recipe | 30,597,451 | 1.07e+17 | 784 | 7 v6e chip | 1.52 | — |
| rl_it4_rollouts | recipe | 30,390,360 | 1.06e+17 | 767 | 7 v6e chip | 1.49 | — |
| rl_it5_rollouts | recipe | 30,320,822 | 1.06e+17 | 2578 | 7 v6e chip | 5.01 | — |
| rl_it6_rollouts | recipe | 30,168,212 | 1.05e+17 | 753 | 7 v6e chip | 1.46 | — |


## FinQA

The same recipe on FinQA, from the FinQA run's raw logs with the same script and prices.

| Question | Recipe (teacher data → KD SFT → 6 GRPO iterations) |
|---|---|
| **Training FLOPs** | **2.0 × 10¹⁸ model FLOPs** (2.6 × 10¹⁸ executed, including rematerialisation). KD SFT 0.20 × 10¹⁸ (10%); the six GRPO iterations 1.78 × 10¹⁸ (90%). Student rollouts add 0.29 × 10¹⁸ inference FLOPs. |
| **Accelerator time** | **23.1 TPU v6e chip-hours**: KD SFT 1.7, GRPO updates 16.3 (about 2.7 per iteration), GRPO rollouts 5.1 (1.3 of which were lost to a replica fault that stalled the last round). **3.5 teacher H200-hours** of active generation (26 minutes on 8 GPUs). |
| **Data** | 6,109 FinQA training questions. Teacher: 24,436 answers (6.2M prompt and 2.4M completion tokens), of which 18,487 were correct and 6,016 distinct ones were kept for SFT (22.7M tokens processed over 3 epochs, 2.0M of them trained; the trainer drops 128 trailing examples per epoch, so these counts are about 2.1% high). RL: 6 rounds × 48,872 student rollouts (41.8M prompt and 41.1M completion tokens in total), giving 89,620 training sequences (115.8M tokens processed, 13.7M trained). The gold answers are used only to verify answers, never as a training target. |
| **Cost** | **$79** at active-compute prices: TPU $62.38 (23.1 chip-hours × $2.70) + teacher $16.15 (3.5 GPU-hours × $4.58); $85 with the teacher at the reservation rate. |

FinQA needs about half the BIRD budget: its prompts and answers are shorter, it has fewer training questions, and from the fourth GRPO iteration fewer questions had mixed outcomes, so the iterations shrank. Proving the result took 1.3 TPU chip-hours ($4) and 1.0 active teacher H200-hours ($5) for the self-distillation arm, the teacher's test evaluation and the serving benchmarks; the student's self-samples, the evaluations and the hold-out screens ran on student replicas and are not counted here. The teacher endpoint was up for at least 4.9 hours during the FinQA work (39.5 GPU-hours, $181), mostly idle while the student trained.

Per-run detail (generated):

| Run | Role | Tokens processed | Model FLOPs | Wall (s) | Accelerators | Accelerator-hours | MFU |
|---|---|---|---|---|---|---|---|
| fq-kd | recipe | 22,656,249 | 2e+17 | 764 | 8 v6e chip | 1.70 | 3.6% |
| fq-self | overhead | 23,209,458 | 2.05e+17 | 516 | 8 v6e chip | 1.15 | 5.4% |
| fq-rl-it1 | recipe | 20,842,445 | 2.59e+17 | 1142 | 8 v6e chip | 2.54 | 3.1% |
| fq-rl-it2 | recipe | 20,788,114 | 3.33e+17 | 1438 | 8 v6e chip | 3.20 | 3.2% |
| fq-rl-it3 | recipe | 21,073,556 | 3.38e+17 | 1320 | 8 v6e chip | 2.93 | 3.5% |
| fq-rl-it4 | recipe | 19,422,165 | 3.11e+17 | 1251 | 8 v6e chip | 2.78 | 3.4% |
| fq-rl-it5 | recipe | 17,429,794 | 2.79e+17 | 1133 | 8 v6e chip | 2.52 | 3.4% |
| fq-rl-it6 | recipe | 16,220,137 | 2.6e+17 | 1035 | 8 v6e chip | 2.30 | 3.4% |
| teacher_train_samples | recipe | 8,612,539 | — | 1588 | 8 H200 | 3.53 | — |
| rl_it1_rollouts | recipe | 13,361,497 | 4.64e+16 | 311 | 7 v6e chip | 0.60 | — |
| rl_it2_rollouts | recipe | 13,673,018 | 4.75e+16 | 329 | 7 v6e chip | 0.64 | — |
| rl_it3_rollouts | recipe | 13,985,581 | 4.86e+16 | 349 | 7 v6e chip | 0.68 | — |
| rl_it4_rollouts | recipe | 13,941,219 | 4.84e+16 | 340 | 7 v6e chip | 0.66 | — |
| rl_it5_rollouts | recipe | 13,839,731 | 4.81e+16 | 331 | 7 v6e chip | 0.64 | — |
| rl_it6_rollouts | recipe | 14,091,480 | 4.9e+16 | 986 | 7 v6e chip | 1.92 | — |
| teacher_test_eval | overhead | 1,279,023 | — | 163 | 8 H200 | 0.36 | — |
| bench_teacher_finqa | overhead | 3,342,413 | — | 287 | 8 H200 | 0.64 | — |
| bench_rl_final_finqa | overhead | 6,545,753 | — | 307 | 1 v6e chip | 0.09 | — |
| bench_base_finqa | overhead | 6,671,899 | — | 344 | 1 v6e chip | 0.10 | — |

## What is counted

**Recipe** — what it takes to produce the final student:
1. Teacher data: the teacher answers the training questions (4 samples each).
2. KD SFT: LoRA SFT of Qwen3-1.7B on the verified teacher answers.
3. RL: for each GRPO iteration, rollouts from the current student (8 samples per training question) and the GRPO update.

**Experiment overhead** — what it took to *prove* the result, not to produce it: the student/teacher screen, the self-distillation and teacher-correction arms, hold-out screens, the evaluations, serving benchmarks and the trainer reproduction run.

## Method

**Training FLOPs.** Per token, a forward pass costs 2 × (decoder matmul parameters + LM-head parameters) plus attention, 4 × layers × attention width × s / 2 for mean sequence length s. LoRA adapters add 2 × adapter parameters per token. A LoRA update costs 2 forward passes (the forward pass and the activation gradients; frozen weights need no weight gradients) plus 3 adapter forward passes; rematerialisation adds one more forward pass on the hardware. A GRPO iteration also runs one forward pass per token for the old-policy log-probabilities, and one more when the KL reference differs from the policy. Tokens processed are the real (unpadded) tokens of every training sequence, including the masked prompt tokens. Norms, softmax and element-wise operations are not counted.

**Accelerator time.** Logged wall time × accelerators in use: trainer wall time (including compilation, data loading and checkpointing) × 8 TPU v6e chips; generation wall time × the chips serving it; teacher generation wall time × 8 H200 GPUs. Model-FLOP utilisation (MFU) is model FLOPs ÷ (accelerator time × 918 TFLOP/s, the v6e bf16 peak per chip).

**Teacher endpoint time.** A dedicated endpoint bills while it is up, busy or not. Up-time comes from a keep-warm log that pinged the endpoint every 5 minutes during the teacher phases; each run of successful pings counts for its span plus one interval.

**Cost.** Accelerator-hours × Cloud Billing Catalog list prices: TPU v6e on-demand in us-east5 for the student; H200 at the DWS flex-start rate (GPU only) for the teacher, with the calendar-reservation rate as an upper reference.

## Not counted
- Serving replica start-up and idle serving between steps, and the time a cold teacher endpoint takes to scale up.
- The endpoint's scale-down tail after the last ping (it stays up for up to about 75 minutes of idleness), so teacher up-time is a lower bound.
- One training launch that ran out of memory after three steps and was restarted with a smaller micro-batch.
- The CPU VM that ran grading and orchestration, storage and network egress.

## Reproduce
`python3 distil/compute_report.py --manifest manifest.json --out compute.json --md compute.md` — the manifest lists the raw files and hardware facts (see the script's docstring).
