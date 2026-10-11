# TPU-Distil: distil a strong teacher into a small student that runs well on TPU

## Thesis

Distilling a strong teacher model into a small student that already runs efficiently on TPU v6e gives two things at once:

1. **Quality:** a significant accuracy gain in a domain where the student was weak.
2. **Efficiency:** inference at a small fraction of the teacher's cost, on hardware the student is already good at.

The teacher, the benchmark and the student are interchangeable. What matters is defensible evidence: same-run paired comparisons, a control that separates the teacher's contribution from "any in-domain fine-tuning", a measured noise floor, measured serving cost, and every number regenerated from raw logs by one script.

## Results at a glance

Teacher DeepSeek-V4.1-Flash (non-thinking mode, Vertex AI endpoint on 8×H200). Student Qwen3-1.7B (non-thinking mode), served by vLLM-TPU on one TPU v6e chip. One recipe, two task families, the same code path; only the task module changes.

| Task (evaluation set) | Base | KD | KD + GRPO | Teacher | GRPO − KD [95% CI] | Teacher ÷ student cost per correct answer |
|---|---|---|---|---|---|---|
| BIRD text-to-SQL (dev, 1,534 questions) | 30.6% | 48.2% | **52.2%** | 60.7% | +4.0 [+2.2, +5.9] | **15.8×** |
| FinQA financial question answering (test, 1,147 questions) | 44.0% | 66.6% | **72.5%** | 75.7% | +5.8 [+3.9, +7.8] | **17.5×** |

Each row comes from one same-run evaluation of the student arms. The distilled and reinforced student closes 72% (BIRD) and 90% (FinQA) of the teacher–base gap. Producing it cost $136 (BIRD) and $79 (FinQA) of accelerator time at list prices ([`docs/COMPUTE.md`](docs/COMPUTE.md)).

## BIRD text-to-SQL

Task: BIRD text-to-SQL, execution accuracy (EX) on all 1,534 BIRD dev questions; the dev databases are never seen in training. Greedy decoding, one fixed prompt. Within each accuracy table below, every student arm was served and evaluated in the same run, and every Δ is a paired comparison over the same questions with a bootstrap 95% CI.

| Model | How it was trained | BIRD dev EX |
|---|---|---|
| Teacher, DeepSeek-V4.1-Flash | — | 60.7% |
| **Qwen3-1.7B, KD + GRPO** | KD student, then 6 GRPO iterations with the execution reward | **52.2%** |
| **Qwen3-1.7B, KD** | LoRA SFT on 10,112 verified teacher answers | **48.2%** |
| Qwen3-1.7B base | — | 30.6% |

| Paired comparison (same run, 1,534 questions) | Δ EX [95% CI] | Better / worse |
|---|---|---|
| KD + GRPO − base | **+21.6 [+19.2, +24.1]** | 387 / 55 |
| KD − base | **+17.6 [+15.2, +20.1]** | 334 / 64 |
| KD + GRPO − KD | **+4.0 [+2.2, +5.9]** | 136 / 74 |
| Base − base, second replica (noise floor) | +0.3 [−0.8, +1.4] | 39 / 34 |
| Teacher − KD + GRPO (teacher from an earlier run) | +8.5 [+6.0, +10.9] | 261 / 131 |

- **Distillation is the large effect, and it is stable.** Three separate same-run evaluations of the same KD checkpoint measured KD − base at +17.8, +18.1 and +17.6 (one training run; training-seed variance was not measured). On four held-out BIRD *train* databases (522 questions), KD − base is +10.0 [+6.7, +13.2].
- **RL adds a smaller, significant gain on top.** Six GRPO iterations (lr 5e-6 for the last four) raise the distilled student by +4.0 on dev and close 72% of the teacher–student gap (KD alone: 58% in the same run). After two iterations at lr 2e-6 the gain was +1.7 to +2.1. On the 522 held-out train-database questions the RL gain is not detectable (+0.4 [−2.3, +2.9]).
- **The teacher's own answers are the right training data.** A separate same-run evaluation compared training data sources for the base student:

| Arm (separate same-run evaluation) | Training data | EX | Δ vs base [95% CI] |
|---|---|---|---|
| KD | 10,112 verified teacher answers | 48.7% | +18.1 [+15.6, +20.6] |
| CORR | 5,883 verified teacher corrections of the student's own failed answers | 37.1% | +6.5 [+4.0, +8.9] |
| Self-distillation (control) | 10,112 of the student's own verified answers | 35.6% | +5.0 [+3.0, +7.0] |
| Base | — | 30.6% | |

KD − self-distillation is +13.1 [+10.7, +15.5]: most of the gain (13.1 of 18.1 points) comes from the teacher's answers rather than from fine-tuning on in-domain data, which alone gives +5.0. Teacher corrections help about as much as the student's own answers (CORR − self +1.5 [−1.0, +3.9]).

| Serving (BIRD dev prompts, best error-free rate over a concurrency ladder; each model benchmarked in its own session on the same hardware and build) | $/hour | Requests/s | $ per 1,000 correct answers |
|---|---|---|---|
| KD + GRPO student, 1× TPU v6e chip | 2.70 (on-demand) | 15.8 | 0.091 |
| Base student, 1× TPU v6e chip | 2.70 (on-demand) | 13.8 | 0.178 |
| Teacher, 8× H200 | 36.60 (GPU-only, cheapest listed rate) | 11.7 | 1.438 |

The distilled student costs **15.8× less per correct answer** than the teacher: 13.6× from hardware price, 1.35× from throughput, and 0.86× from accuracy (the teacher is more accurate). The pricing is deliberately unfavourable to the student: the TPU uses its on-demand rate, and the teacher uses the cheapest listed H200 rate in its region (DWS flex-start, us-south1) with CPU and memory excluded; at the same purchase tier for both the ratio roughly doubles. Prices come from the Cloud Billing Catalog (effective 2026-10-09).

Both servers ran with untuned defaults and a cap of 128 concurrent sequences; the benchmarks capped answers at 1,024 tokens (accuracy runs used 2,048; 17 of 1,024 benchmark requests reached the cap). Teacher: Vertex Model Garden vLLM, TP8 with expert parallelism, fp8 KV cache. Student: vLLM-TPU, one chip, bf16 KV cache. The teacher's throughput flattened from 128 to 512 concurrent requests, so its server, not the network, was the limit.

## FinQA financial question answering

Task: [FinQA](https://github.com/czyssrs/FinQA), numerical reasoning over a page of an earnings report (text and a table). The answer is a number (or yes/no), graded correct within 1% relative error (absolute 0.005 when the gold answer is below 0.5 in magnitude), also accepting the ×100 and ÷100 percentage forms. Evaluation set: all 1,147 test questions, never trained on or used to choose a checkpoint or a hyperparameter; dev (883 questions) is the holdout for screens. Training uses 6,109 of the 6,251 training questions: the 142 that match a test or dev question exactly or share a 13-word sequence with one are left out, and no training question comes from a test or dev report page. Evaluator controls on test: gold answers score 100%, answers shuffled between questions 0.6%.

Same recipe and settings as BIRD: 4 teacher samples per training question, verified answers only (at most 4 distinct per question), 6,016 rows, LoRA SFT, then 6 GRPO iterations with the answer-match reward.

| Model (same-run evaluation) | FinQA test | FinQA dev |
|---|---|---|
| Teacher, DeepSeek-V4.1-Flash (earlier run) | 75.7% | — |
| **Qwen3-1.7B, KD + GRPO** | **72.5%** | 74.1% |
| **Qwen3-1.7B, KD** | **66.6%** | 70.4% |
| Qwen3-1.7B base (two replicas) | 44.0% / 42.1% | 43.9% / 44.4% |

| Paired comparison (same run, FinQA test, 1,147 questions) | Δ [95% CI] | Better / worse |
|---|---|---|
| KD + GRPO − base | **+28.4 [+25.5, +31.5]** | 367 / 41 |
| KD − base | **+22.6 [+19.4, +25.7]** | 321 / 62 |
| KD + GRPO − KD | **+5.8 [+3.9, +7.8]** | 100 / 33 |
| Base − base, second replica (noise floor) | −1.9 [−3.3, −0.5] | 23 / 45 |
| Teacher − KD + GRPO (teacher from an earlier run) | +3.2 [+1.0, +5.4] | 105 / 68 |

- **Distillation replicates, with a larger gain.** KD − base is +22.6 here and +24.1 [+21.0, +27.2] in the earlier run that included the control arm.
- **The teacher's answers again beat in-domain fine-tuning.** In that earlier same-run evaluation the self-distillation control (6,016 of the student's own verified answers) scored 57.0% against KD's 66.6%: KD − self is +9.6 [+6.9, +12.4], and self − base is +14.6 [+11.8, +17.4]. The teacher's share of the gain (9.6 of 24.1 points) is smaller than on BIRD, because part of FinQA's self-distillation gain is answer-format learning: the base student gives no number on more than 100 test questions.
- **RL adds a significant gain on FinQA too.** Six GRPO iterations add +5.8 on test and +3.6 [+1.5, +5.9] on dev; the first two iterations alone add +3.1 [+1.5, +4.8] on test. The final student is 3.2 points behind the teacher and closes 90% of the gap (KD alone: 71% in the same run). Answers grow by about 19% under RL (mean 133 → 159 tokens); 23 of 1,147 reach the 2,048-token limit (KD: 10).

| Serving (FinQA test prompts, as for BIRD) | $/hour | Requests/s | $ per 1,000 correct answers |
|---|---|---|---|
| KD + GRPO student, 1× TPU v6e chip | 2.70 (on-demand) | 24.1 | 0.043 |
| Base student, 1× TPU v6e chip | 2.70 (on-demand) | 22.0 | 0.077 |
| Teacher, 8× H200 | 36.60 (GPU-only, cheapest listed rate) | 17.9 | 0.750 |

The final FinQA student costs **17.5× less per correct answer** than the teacher: 13.6× from hardware price, 1.35× from throughput, and 0.96× from accuracy. The student's rate was still rising at the top of its ladder (256 concurrent requests); the teacher's peaked at 128 and fell at 256 and 512.

### What this does not show
- One student, one teacher, and one training seed per arm. The noise floor measures serving non-determinism (two replicas of the same weights), not training-seed variance.
- **Serving noise can exceed one point.** Greedy decoding on vLLM-TPU is not batch-invariant: the two FinQA base replicas gave identical text on only 630 of 1,147 test questions, and their scores differed by −1.9 [−3.3, −0.5] in the final run (+0.5 and +1.0 in two earlier runs), outside the 1.5-point replica tolerance set before the earlier FinQA runs. Each arm is served on its own replicas, so replica effects of this size are part of every arm comparison. The headline FinQA gains (+5.8 from GRPO, +22.6 and +28.4 over base) are at least three times that difference; smaller effects, such as the FinQA dev RL gain (+3.6) and the two-iteration gains, are not. Every comparison uses arms from the same run.
- On BIRD, the RL gain is significant on dev (11 databases, 1,534 questions) but not detectable on the 522 held-out train-database questions. The decision to run four more BIRD RL iterations was taken after seeing the first RL result on dev; to limit tuning on the test set, dev was then used exactly once more, on an iterate fixed in advance, and early stopping used the held-out questions only. FinQA then ran the same six-iteration schedule unchanged, with its last iterate fixed in advance and its test set used once for RL.
- The FinQA prompt was revised once before any teacher output or training. The first version offered "Answer: yes / no" on every question; reading the base student's test answers showed it answered yes or no on 363 numeric test questions. The replacement prompt was then chosen on dev, and every reported FinQA result uses it.
- FinQA grading accepts a 1% relative error and the ×100 and ÷100 percentage forms, which is more lenient than the official FinQA execution-accuracy metric; the scores here are not comparable to published FinQA results. Answers shuffled between questions score 0.6% under this grader.
- **The efficiency margin depends on the teacher deployment.** With the pricing above, the BIRD ratio drops to 10× if the teacher served 1.6× faster, and to parity at 15.8× faster (FinQA: 1.75× and 17.5×).
- The teacher's accuracy was measured in an earlier run than the student arms, with a 4,096-token output limit vs 2,048 for the students. Only 5 of its 1,534 BIRD answers and 7 of its 1,147 FinQA answers exceeded 2,048 tokens, so this moves its scores by at most 0.3 and 0.6 points.
- Qwen3-4B-Instruct-2507 already matches the teacher on BIRD (61.7%) without any training; its serving cost was not measured. Picking the right student can matter as much as distilling it. The 1.7B was chosen because it had the largest gap.
- BIRD and FinQA are public, so the teacher and the base student may have seen the evaluation questions in pre-training. This applies to every arm; no evaluation question, database or report page was used in training.

## Recipe

1. **Measure the gap.** Serve candidate students on single v6e chips and the teacher on its endpoint; score all on the benchmark with one prompt. Keep a student with a large gap.
2. **Generate verified teacher data.** Sample the teacher on training questions (4 samples, T = 0.7). Keep only samples the task verifies (BIRD: the SQL returns the gold rows; FinQA: the number matches the gold answer), with at most 4 distinct answers per question. Audit for leakage against the evaluation set (evaluation databases or report pages, exact and 13-gram question overlap) and re-verify every kept answer.
3. **Train.** LoRA SFT (rank 256, alpha 512, attention and MLP projections) on the TPU v6e-8, global batch 256, lr 1e-4 with cosine decay, 3 epochs. Loss is on the assistant tokens only. Merge the adapter and export a Hugging Face checkpoint.
4. **Control.** Train the same recipe on the same number of the student's own verified samples (self-distillation).
5. **Reinforce.** GRPO: sample the current student 8 times per training question at temperature 1.0, reward 1 when the answer verifies, train on up to 2,048 questions with mixed outcomes (PPO clip 0.2, KL to the KD student with β 0.02, one pass in steps of 64 sequences); six iterations, lr 2e-6 for the first two and 5e-6 after. On BIRD only, CORR: show the teacher the student's first failed answer and its execution result (never the gold query), ask for a minimal correction, keep verified corrections, same SFT recipe on the base student.
6. **Evaluate and price.** Serve every arm and a second base replica together; compare paired; benchmark serving throughput; compute cost per correct answer from catalog prices.

How the run was executed, stage by stage: [`docs/RECIPE.md`](docs/RECIPE.md). Components and data flow: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). One-page summary: [`docs/tpu_distil_poster.png`](docs/tpu_distil_poster.png).

## Layout

| Path | What |
|---|---|
| `distil/tasks/` | Task modules selected with `--task`: `bird.py` (schema prompts, SQL extraction, execution accuracy) and `finqa.py` (report-page prompts, answer extraction, numeric grading) |
| `distil/prompts.py`, `generate.py`, `grade.py`, `compare.py` | Prompt files, multi-endpoint generation, grading with gold and shuffled controls, paired bootstrap |
| `distil/build_sft.py`, `leak_audit.py`, `overlap.py` | Verified SFT sets, the independent leakage and re-verification audit, and the question-overlap index both use |
| `distil/correct.py`, `build_grpo_rollouts.py`, `passk.py` | Teacher corrections of failed student answers (BIRD), GRPO records from graded samples, pass@k |
| `distil/serve_small.sh`, `bench_serving.py`, `report.py` | Single-chip vLLM-TPU serving, throughput benchmark, report generator |
| `distil/compute_report.py`, `docs/COMPUTE.md` | Training FLOPs, accelerator time, data and cost from raw logs, and the note it produced |
| `docs/RECIPE.md` | How the run was executed: machines, libraries and versions, serving, SFT and GRPO settings, evaluation design |
| `docs/ARCHITECTURE.md` | Components, data flow, record formats and the rules the code enforces |
| `docs/poster/` | Poster generator (`make_poster.py`) and the values it draws (`results.json`) |
| `distil/teacher_client.py`, `vertex_openai_proxy.py` | Retrying chat client, OpenAI-compatible proxy in front of the Vertex teacher endpoint |
| `trainer/` | Tunix LoRA trainer for dense Qwen3 (`train_tpu_vm.sh`: SFT, GRPO, merge), logit-parity check, merged export |
| `tests/` | Unit and integration tests for all of the above |

Training data is JSONL, one single-turn record per line: `{"id", "messages": [system, user], "response"}` plus the task's group key (`db_id` for BIRD, `report` for FinQA); GRPO records add `"group_id"` and `"reward"` (`trainer/data.py`).

Tests: `python -m pytest tests -q` (needs JAX, Tunix, qwix and transformers for the trainer tests).
