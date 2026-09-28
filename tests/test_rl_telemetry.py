"""Unit tests for RL convergence telemetry, dynamic group filtering, and benchmark decontamination."""

from __future__ import annotations

import pytest

from tpu_distil.rl_telemetry import (
    RLStepTelemetry,
    compute_k3_kl_divergence,
    compute_reward_smoothness_metrics,
    evaluate_rl_health_corridor,
    filter_informative_grpo_groups,
    verify_zero_benchmark_leakage,
)


def test_k3_kl_divergence_is_non_negative_and_zero_when_identical() -> None:
    logprobs = [-0.5, -1.2, -0.8, -2.1]
    assert compute_k3_kl_divergence(logprobs, logprobs) == pytest.approx(0.0, abs=1e-9)

    shifted = [-0.7, -1.0, -1.1, -1.8]
    kl = compute_k3_kl_divergence(shifted, logprobs)
    assert kl > 0.0
    assert kl < 0.25


def test_filter_informative_grpo_groups_drops_zero_variance_groups() -> None:
    groups = [
        [0.0, 0.0, 0.0, 0.0],  # all fail -> sigma = 0 (dropped)
        [1.0, 1.0, 1.0, 1.0],  # all pass -> sigma = 0 (dropped)
        [1.2, 1.2, -0.3, 1.1],  # mixed -> sigma > 0 (kept)
        [1.3, -0.2, -0.2, 1.3],  # mixed -> sigma > 0 (kept)
    ]
    kept, rate = filter_informative_grpo_groups(groups)
    assert kept == [2, 3]
    assert rate == pytest.approx(0.5)


def test_reward_smoothness_and_rl_health_corridor() -> None:
    steps = [
        RLStepTelemetry(
            step=i + 1,
            policy_loss=-0.01 * (i + 1),
            mean_total_reward=0.40 + 0.025 * i,
            ema_total_reward=0.40 + 0.022 * i,
            mean_terminal_reward=0.30 + 0.020 * i,
            mean_recovery_bonus=0.10 + 0.005 * i,
            online_wrong_to_right_rate=0.12 + 0.015 * i,
            k3_kl_divergence=0.045 + 0.001 * i,
            policy_token_entropy=0.72 - 0.005 * i,
            clip_fraction=0.08,
            dynamic_group_rate=0.82,
        )
        for i in range(20)
    ]
    report = evaluate_rl_health_corridor(steps)
    assert report["all_healthy"] is True
    assert report["smoothness"]["reward_snr"] >= 2.0


def test_verify_zero_benchmark_leakage_detects_overlap_and_passes_clean() -> None:
    eval_ids = {"pandas-etl", "sqlite-db-truncate"}
    eval_prompts = [
        "Write an ETL script in /app/etl.py that normalizes sales CSV records and outputs parquet.",
        "Recover the corrupted SQLite database in /app/data.db and restore the users table.",
    ]
    clean_train = [
        {
            "task_id": "term_gym_git_bisect_001",
            "prompt": "Find the broken commit in /app/repo using git log and revert the header change.",
        },
        {
            "task_id": "term_gym_nginx_log_002",
            "prompt": "Parse /app/access.log to extract top 5 client CIDRs returning HTTP 502 responses.",
        },
    ]
    res_clean = verify_zero_benchmark_leakage(clean_train, eval_ids, eval_prompts)
    assert res_clean["zero_benchmark_leakage"] is True

    leaked_train = clean_train + [
        {
            "task_id": "pandas-etl",
            "prompt": "Write an ETL script in /app/etl.py that normalizes sales CSV records and outputs parquet.",
        }
    ]
    res_leaked = verify_zero_benchmark_leakage(leaked_train, eval_ids, eval_prompts)
    assert res_leaked["zero_benchmark_leakage"] is False
    assert res_leaked["id_collisions_count"] == 1
