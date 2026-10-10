# TPU-Distil: distil a strong teacher into a small student that runs well on TPU

## Thesis

Distilling a strong teacher model into a small student that already runs efficiently on TPU v6e gives two things at once:

1. **Quality:** a significant accuracy gain in a domain where the student was weak.
2. **Efficiency:** inference at a small fraction of the teacher's cost, on hardware the student is already good at.

The teacher, the benchmark and the student are interchangeable. What matters is defensible evidence: same-run paired comparisons, a control that separates the teacher's contribution from "any in-domain fine-tuning", a measured noise floor, measured serving cost, and every number regenerated from raw logs by one script.

## Result

Teacher DeepSeek-V4.1-Flash (non-thinking mode, Vertex AI endpoint on 8×H200). Student Qwen3-1.7B (non-thinking mode), served by vLLM-TPU on one TPU v6e chip. Task: BIRD text-to-SQL, execution accuracy (EX) on all 1,534 BIRD dev questions; the dev databases are never seen in training. Greedy decoding, one fixed prompt. Within each accuracy table below, every student arm was served and evaluated in the same run, and every Δ is a paired comparison over the same questions with a bootstrap 95% CI.

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

The distilled student costs **15.8× less per correct answer** than the teacher: 13.6× from hardware price, 1.35× from throughput, and 0.86× from accuracy (the teacher is more accurate). The pricing is deliberately unfavourable to the student: the TPU uses its on-demand rate, and the teacher uses the cheapest listed H200 rate in its region (DWS flex-start, us-south1) with CPU and memory excluded; at the same purchase tier for both the ratio roughly doubles. Prices come from the Cloud Billing Catalog (effective 2026-10-09). Compute, data and cost of the whole run: [`docs/COMPUTE.md`](docs/COMPUTE.md).

Both servers ran with untuned defaults and a cap of 128 concurrent sequences; the benchmarks capped answers at 1,024 tokens (accuracy runs used 2,048; 17 of 1,024 benchmark requests reached the cap). Teacher: Vertex Model Garden vLLM, TP8 with expert parallelism, fp8 KV cache. Student: vLLM-TPU, one chip, bf16 KV cache. The teacher's throughput flattened from 128 to 512 concurrent requests, so its server, not the network, was the limit.

### What this does not show
- One task family (text-to-SQL), one student, one teacher, and one training seed per arm. The noise floor measures serving non-determinism (two replicas of the same weights), not training-seed variance.
- The RL gain is significant on dev (11 databases, 1,534 questions) but not detectable on the 522 held-out train-database questions. The decision to run four more RL iterations was taken after seeing the first RL result on dev; to limit tuning on the test set, dev was then used exactly once more, on an iterate fixed in advance, and early stopping used the held-out questions only.
- **The efficiency margin depends on the teacher deployment.** With the pricing above, the ratio drops to 10× if the teacher served 1.6× faster, and to parity at 15.8× faster.
- The teacher's accuracy was measured in an earlier run than the student arms, with a 4,096-token output limit vs 2,048 for the students. Only 5 of its 1,534 answers exceeded 2,048 tokens, so this moves its score by at most 0.3 points.
- Qwen3-4B-Instruct-2507 already matches the teacher on this task (61.7%) without any training; its serving cost was not measured. Picking the right student can matter as much as distilling it. The 1.7B was chosen because it had the largest gap.
- BIRD is public, so the teacher and the base student may have seen the dev questions in pre-training. This applies to every arm; the dev databases were excluded from all training data.

## Recipe

1. **Measure the gap.** Serve candidate students on single v6e chips and the teacher on its endpoint; score all on the benchmark with one prompt. Keep a student with a large gap.
2. **Generate verified teacher data.** Sample the teacher on training questions (4 samples, T = 0.7). Keep only samples whose SQL returns the gold result, with at most 4 distinct queries per question. Audit for leakage against the evaluation set (database names, exact and 13-gram question overlap) and re-execute every kept query.
3. **Train.** LoRA SFT (rank 256, alpha 512, attention and MLP projections) on the TPU v6e-8, global batch 256, lr 1e-4 with cosine decay, 3 epochs. Loss is on the assistant tokens only. Merge the adapter and export a Hugging Face checkpoint.
4. **Control.** Train the same recipe on the same number of the student's own verified samples (self-distillation).
5. **Variants.** CORR: show the teacher the student's first failed answer and its execution result (never the gold query), ask for a minimal correction, keep verified corrections, same SFT recipe on the base student. GRPO: sample the current student 8 times per training question at temperature 1.0, reward 1 when the result matches gold, train on up to 2,048 questions with mixed outcomes (PPO clip 0.2, KL to the KD student with β 0.02, 256 steps per iteration); six iterations, lr 2e-6 for the first two and 5e-6 after.
6. **Evaluate and price.** Serve every arm and a second base replica together; compare paired; benchmark serving throughput; compute cost per correct answer from catalog prices.

## Layout

| Path | What |
|---|---|
| `distil/bird.py` | BIRD loading, schema prompts, SQL extraction, execution accuracy |
| `distil/prompts.py`, `generate.py`, `grade.py`, `compare.py` | Prompt files, multi-endpoint generation, grading, paired bootstrap |
| `distil/build_sft.py`, `leak_audit.py` | Verified SFT sets and the independent leakage / re-execution audit |
| `distil/correct.py`, `build_grpo_rollouts.py`, `passk.py` | Teacher corrections of failed student answers, GRPO records from graded samples, pass@k |
| `distil/serve_small.sh`, `bench_serving.py`, `report.py` | Single-chip vLLM-TPU serving, throughput benchmark, report generator |
| `distil/compute_report.py`, `docs/COMPUTE.md` | Training FLOPs, accelerator time, data and cost from raw logs, and the note it produced |
| `distil/teacher_client.py`, `vertex_openai_proxy.py` | Retrying chat client, OpenAI-compatible proxy in front of the Vertex teacher endpoint |
| `trainer/` | Tunix LoRA trainer for dense Qwen3 (`train_tpu_vm.sh`: SFT, GRPO, merge), logit-parity check, merged export |
| `tests/` | Unit and integration tests for all of the above |

Training data is JSONL, one single-turn record per line: `{"id", "messages": [system, user], "response", "db_id"}`; GRPO records add `"group_id"` and `"reward"` (`trainer/data.py`).

Tests: `python -m pytest tests -q` (needs JAX, Tunix, qwix and transformers for the trainer tests).
