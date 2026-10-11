# How the recipe ran

The machines, libraries, settings and checks of the README's runs, stage by stage. Sections 1–7 describe the BIRD run. Section 8 lists what changed for FinQA.

## Machines

| Machine | Role |
|---|---|
| Cloud TPU v6e-8 VM, us-east5-a (8 v6e chips) | Serves the student, trains it, merges adapters, benchmarks throughput |
| CPU VM n4-standard-64, us-east5-a | Holds the BIRD databases, sends prompts, grades answers, builds data sets, runs long jobs |
| Vertex AI dedicated endpoint, a3-ultragpu-8g (8 H200), us-south1 | Serves the teacher, DeepSeek-V4.1-Flash |

A Cloud Storage bucket holds checkpoints, data and evidence.

## 1. Task data

Evaluation set: BIRD dev, 1,534 questions on 11 databases that training never sees. `distil/prompts.py` builds a system message and a user message per question. The user message holds:

1. The CREATE TABLE statements of the database.
2. Three example rows per table, each cell cut to 40 characters.
3. The BIRD external-knowledge hint.
4. The question.
5. An instruction to reason briefly and write one SQLite query in a `sql` code block.

The template is fixed, and the run records its SHA-256. The stage uses only `sqlite3` from the standard library. Training uses 64 of the 69 training databases. We removed `works_cycles` (64,016-character schema); app_store, disney, legislator and movie_platform form a hold-out set.

## 2. Serving

| | Student (Qwen3-1.7B) | Teacher (DeepSeek-V4.1-Flash) |
|---|---|---|
| Software | vLLM 0.26.1rc1 TPU build, tpu-inference backend, JAX 0.11.0, libtpu 0.0.44 | Vertex AI Model Garden vLLM container `pytorch-vllm-serve`, release 20260922 |
| Layout | one replica per v6e chip (tensor parallel 1), up to 8 replicas | tensor parallel 8 with expert parallelism |
| Flags | `--max-model-len 16384 --max-num-batched-tokens 4096 --max-num-seqs 128 --gpu-memory-utilization 0.85`, bf16 KV cache (`distil/serve_small.sh`) | `--max-model-len 65536 --max-num-seqs 128`, fp8 KV cache |
| Mode | non-thinking: `chat_template_kwargs {"enable_thinking": false}` | non-thinking: `chat_template_kwargs {"thinking": false}` |

`distil/vertex_openai_proxy.py` adds an OAuth token to each teacher request. The endpoint scales to zero when idle, so a keep-warm loop sends a one-token request every 5 minutes during teacher stages.

`distil/generate.py` (standard library only) spreads requests over the replicas and retries for about 55 minutes. It appends each response and its token usage to a JSONL file, and resumes after an interruption.

## 3. Grading

`distil/grade.py` measures execution accuracy. It runs the last `sql` code block on a read-only database copy with a 30-second timeout. It compares the returned set of rows with the gold set. Controls on BIRD dev: the gold query scored 100% of the 1,530 executable gold queries; another question's gold query from the same database scored 0.2%.

## 4. Training data

1. The teacher answered each of the 8,432 training questions 4 times at temperature 0.7.
2. `distil/build_sft.py` kept answers whose SQL returns the gold rows, at most 4 distinct queries per question.
3. It drew exactly 10,112 rows with a fixed seed.
4. The student's own correct answers (8 samples per question) gave a self-distillation control set of the same size.
5. `distil/leak_audit.py` checked every row again. It refuses evaluation databases, exact and 13-gram overlap with dev questions, and SQL that no longer returns the gold rows.

## 5. Supervised fine-tuning (SFT)

Training uses all 8 chips; the launcher is `trainer/train_tpu_vm.sh`.

| Library | Version | Use |
|---|---|---|
| google-tunix | 0.1.8 | Qwen3 model, trainer |
| qwix | 0.1.8 | low-rank adaptation (LoRA) adapters |
| Flax (NNX), Optax, Orbax | 0.12.8, 0.2.6, 0.12.2 | modules, AdamW, checkpoints |
| Transformers, safetensors | 5.12.1, 0.7.0 | tokenizer and chat template, checkpoint files |
| JAX, libtpu | 0.11.0, 0.0.44 | compilation and TPU runtime |

| Setting | Value |
|---|---|
| Student | Qwen3-1.7B, bf16 weights, frozen |
| LoRA | rank 256, alpha 512, on q, k, v, o, gate, up, down of every layer (196 modules); float32 adapter weights |
| Batch | 256 sequences per step, micro-batch 8, gradient accumulation 32 |
| Optimiser | AdamW, lr 1e-4, 5% warm-up, cosine decay to 0, weight decay 0, gradient clip at norm 1.0 |
| Length | at most 8,192 tokens; length buckets 2,048, 4,096, 8,192 |
| Epochs | 3 (117 optimiser steps for 10,112 examples) |
| Mesh | fully sharded data parallel 8, tensor parallel 1; decoder rematerialisation; splash attention |
| Loss | weight 1 on the response and its end-of-turn token; weight 0 on system, prompt and template tokens |

Checks before any training result counted:

1. `trainer/parity_check.py`: training model vs vLLM replica, next-token log-probabilities on 16 fixed prompts. Top-1 agreement 0.998, mean Kullback–Leibler (KL) divergence 1e-4 (limits 0.99 and 1e-3).
2. A test on the real tokenizer: the training context equals the prompt that vLLM serves.

The trainer merges the adapters into the base weights and streams the Hugging Face checkpoint to Cloud Storage. Merged checkpoints serve with the base model's flags, so arms differ only in weights.

## 6. Reinforcement learning (GRPO)

Group relative policy optimisation (GRPO) starts from the knowledge distillation (KD) checkpoint, with the same trainer. Per iteration:

1. Seven or eight replicas of the current student answer each training question 8 times at temperature 1.0, top-p 1.0 (with seven, the eighth chip serves the KD student for the hold-out screen). The trainer recomputes log-probabilities at temperature 1 without correction, so other settings would make the data off-policy.
2. Reward is 1 if the SQL returns the gold rows, else 0. `distil/build_grpo_rollouts.py` keeps at most 2,048 questions with mixed rewards for one pass.

| Setting | Value |
|---|---|
| Advantage | group-relative, divided by the group standard deviation |
| Policy loss | PPO clip 0.2, dual clip 3 |
| KL | k3 estimate against the KD checkpoint, clamped per token at 10, beta 0.02 |
| Optimiser | AdamW, constant lr 2e-6 in iterations 1–2, 5e-6 in iterations 3–6 |
| Batch | 64 sequences per step, 1 epoch, 256 steps per iteration |
| Length | at most 8,192 tokens; longer sequences dropped |
| Guard | `trainer/kl_guard.sh` stops the run if a step's KL is above 0.2 |

A driver on the CPU VM ran six iterations: serve, screen against KD on the hold-out set, collect and grade rollouts, update, fetch the merged checkpoint. The highest step KL rose from 0.0045 (iteration 1) to 0.0328 (iteration 6). The guard never stopped a run.

## 7. Evaluation

- Compared arms ran in one evaluation: greedy decoding, 2,048-token output limit.
- Each run included two base replicas, to measure the serving noise floor.
- `distil/compare.py`: paired per-question differences, bootstrap 95% confidence interval (10,000 resamples, fixed seed).
- Pass conditions were written before each stage. BIRD dev never chose a checkpoint or a hyperparameter.
- `distil/report.py` and `distil/compute_report.py` regenerate every result and cost figure from raw files and Cloud Billing Catalog prices.

## 8. Second task: FinQA

The same machines, libraries, trainer and settings ran FinQA. Each pipeline script takes `--task finqa`; the task module `distil/tasks/finqa.py` supplies the prompt, the answer extraction and the grader.

| Stage | FinQA |
|---|---|
| Data | FinQA train 6,251, dev 883, test 1,147 questions. Test is the evaluation set. Dev is the holdout for screens. |
| Leak filter | `distil/prompts.py` removed 142 training questions that match a test or dev question exactly or share a 13-word sequence with one. No training question shares a report page with test or dev. 6,109 training questions remain. |
| Prompt | System: "You are an expert financial analyst." User: the report text before the table, the table as pipe-separated rows, the text after the table, the question, and an instruction to reason briefly and end with `Answer: <number>`. Yes/no is allowed only for yes/no questions. Version `finqa-2`; the run records its SHA-256. |
| Grading | The number after the last `Answer:` line. Correct if it is within 1% of the gold answer, also in the ×100 and ÷100 percentage forms. If the gold answer is below 0.5 in magnitude, an absolute error of 0.005 is also correct, but 0 never matches a gold answer that is not 0. Yes/no answers must match exactly. Controls on test: gold 100%, shuffled 0.6%. |
| Teacher data | 4 samples per question at temperature 0.7, top-p 0.95: 24,436 answers in 26 minutes, 18,487 correct. At most 4 distinct answers per question, then exactly 6,016 rows (4,775 questions). |
| Control | 8 student samples per question: 6,016 rows from 3,558 questions. |
| Leak audit | Both sets: 0 rows from non-training questions, 0 evaluation report pages, 0 exact or 13-gram overlaps, 0 answers that fail re-grading. |
| SFT | Same settings as BIRD. 6,016 rows give 69 optimiser steps over 3 epochs (23 full batches per epoch; the trainer drops the last 128 rows of each epoch). |
| GRPO | Same settings as BIRD: 6 iterations, lr 2e-6 then 5e-6. Rollouts cover all 6,109 questions × 8 samples. From iteration 4, fewer than 2,048 questions had mixed outcomes, so the iterations used all of them: 255, 255, 254, 236, 210 and 193 steps. The highest step KL rose from 0.0061 to 0.0492. The guard never stopped a run. |
| Evaluation | One run with base (two replicas), KD, GRPO iteration 2 and GRPO iteration 6, on test and dev. Iteration 6 was fixed as the final arm before the run. |
| Serving benchmark | FinQA test prompts, same ladder and limits as BIRD. Student: best rate 24.1 requests/s at 256 concurrent requests. Teacher: 17.9 requests/s at 128. |

The first FinQA prompt (`finqa-1`) offered `Answer: yes` / `Answer: no` on every question. The base student then answered yes or no on 363 numeric test questions. We changed the prompt before any teacher output or training. We chose the new prompt on dev: the not-a-number failures of the base student fell from 262 to 65.

## Problems and fixes

| Problem | Fix |
|---|---|
| Out of memory with two 8,192-token sequences per chip | One sequence per chip; batch of 256 kept |
| A prompt log-probability request (about 4.6 GB extra memory) stopped a serving replica | Parity uses a separate replica with less memory for weights and cache |
| One chip's runtime stopped with `SLICE_FAILURE_WORKER_UNAVAILABLE` | A watchdog restarts replicas; the client retries |
| Host maintenance rebooted the TPU VM | Clients waited and resumed |
| Two replicas of the same weights gave different greedy text on 45% of FinQA test questions (different final answers on 23%) | Every comparison uses arms from one run; two base replicas measure the noise floor in each run |
| The teacher takes about 25 minutes to start from zero | Keep-warm loop during teacher stages |

## Reproduce

Commands: `CLAUDE.md` and each script's docstring. A second KD run with this code gave identical losses at all 117 steps and a byte-identical merged checkpoint.
