"""End-to-end verification tests for Terminus-2 parity, Terminal-Gym-4K zero-leakage,
200-step SCoRe-RL telemetry, and 3-student TPU v6e distillation evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path

from tpu_distil.rl_telemetry import (
    RLStepTelemetry,
    evaluate_rl_health_corridor,
    verify_zero_benchmark_leakage,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _find_latest_log(filename: str) -> Path:
    matches = sorted((REPO_ROOT / ".agents").glob(f"*/logs/{filename}"))
    assert matches, f"Missing expected log artifact: {filename}"
    return matches[-1]


def test_terminus2_teacher_baseline_parity() -> None:
    probe_path = _find_latest_log("terminus_baseline_probe.json")
    probe = json.loads(probe_path.read_text(encoding="utf-8"))

    teacher_res = probe["teacher_terminus2_zero_shot"]
    assert teacher_res["model"] == "deepseek-ai/DeepSeek-V4.1-Flash"
    assert teacher_res["num_tasks"] == 50
    assert teacher_res["passed_count"] >= 39
    assert teacher_res["pass_at_1"] >= 0.78
    assert probe["public_benchmark_reference"]["within_10pct_of_public_benchmark"] is True


def test_terminal_gym_4k_zero_benchmark_leakage() -> None:
    summary_path = _find_latest_log("terminal_gym_summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["train_trajectories"] == 4096
    assert summary["bc_trajectories"] == 4096
    assert summary["val_trajectories"] == 128
    assert summary["zero_benchmark_leakage"] is True
    assert summary["id_collisions_count"] == 0
    assert summary["hash_collisions_count"] == 0
    assert summary["ngram_violations_count"] == 0
    assert summary["max_observed_ngram_jaccard"] < 0.25
    assert summary["observation_loss_mask_zero_verified"] is True
    assert summary["pad_multiple_256_verified"] is True

    # Independently audit a sample of training trajectories against all 50 held-out Terminal-Bench tasks
    tbench_dir = REPO_ROOT / "data/terminal-bench/original-tasks"
    eval_ids: set[str] = set()
    eval_instructions: list[str] = []
    for tdir in sorted(tbench_dir.iterdir()):
        tyaml = tdir / "task.yaml"
        if tyaml.exists():
            eval_ids.add(tdir.name)
            eval_instructions.append(tyaml.read_text(encoding="utf-8"))

    train_lines = (REPO_ROOT / "data/terminal_gym_train.jsonl").read_text(encoding="utf-8").splitlines()[:256]
    train_sample = [json.loads(ln) for ln in train_lines if ln.strip()]
    audit = verify_zero_benchmark_leakage(
        train_sample, eval_ids, eval_instructions, max_ngram_jaccard=0.25
    )
    assert audit["zero_benchmark_leakage"] is True


def test_200_step_rl_telemetry_health_corridor() -> None:
    telemetry_path = _find_latest_log("rl_telemetry.json")
    telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))

    assert telemetry["all_students_healthy"] is True
    expected_models = (
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "Qwen/Qwen3.6-27B-FP8",
        "Qwen/Qwen3.8-27B-FP8",
    )
    for model_id in expected_models:
        assert model_id in telemetry["students"]
        s_data = telemetry["students"][model_id]
        steps = [RLStepTelemetry(**d) for d in s_data["steps"]]
        assert len(steps) == 200
        corridor = evaluate_rl_health_corridor(steps)
        assert corridor["all_healthy"] is True
        assert 0.015 <= corridor["mean_k3_kl"] <= 0.150
        assert corridor["mean_dynamic_group_rate"] >= 0.65
        assert corridor["smoothness"]["reward_snr"] >= 2.0
        assert corridor["min_policy_entropy"] >= 0.35
        val_curve = s_data["validation_pass_at_1_curve"]
        assert val_curve[-1]["val_pass_at_1"] - val_curve[0]["val_pass_at_1"] >= 0.12


def test_three_student_terminal_bench_evaluation_exceeds_50pct() -> None:
    eval_matches = sorted((REPO_ROOT / ".agents").glob("*/logs/terminal_bench_*eval.json"))
    multi_eval_path = next(
        p for p in reversed(eval_matches) if "students" in json.loads(p.read_text(encoding="utf-8"))
    )
    report = json.loads(multi_eval_path.read_text(encoding="utf-8"))

    assert report["gate_verdict"] == "GREEN"
    assert all(report["clauses"].values())

    for student_key in ("qwen3_30b_a3b_moe", "qwen36_27b_dense", "qwen38_27b_dense"):
        s_eval = report["students"][student_key]
        summary = s_eval["summary"]
        p_zero = summary["zero_shot_pass_at_1"]
        p_bc = summary["bc_control_pass_at_1"]
        p_sft = summary["score_sft_pass_at_1"]
        p_rl = summary["score_rl_pass_at_1"]

        assert p_rl > 0.50, f"{student_key} SCoRe-RL Pass@1 {p_rl} did not exceed 50%"
        assert p_rl > p_sft > p_bc > p_zero, f"{student_key} violated strict ablation ordering"
        assert summary["wrong_to_right_transition_gain_over_bc"] >= 0.15

        # Verify zero teacher trajectory leakage across all student evaluation arms
        for arm_name, arm_data in s_eval["arms"].items():
            assert len(arm_data["task_results"]) == 50
            for task_res in arm_data["task_results"]:
                for step in task_res["trajectory"]["steps"]:
                    assert step.get("teacher_generated", False) is False, (
                        f"Teacher step leaked into {student_key} arm {arm_name} task {task_res['task_id']}"
                    )
