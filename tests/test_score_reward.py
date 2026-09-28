"""Unit tests for `tpu_distil.score_reward`."""

from __future__ import annotations

import pytest

from tpu_distil.score_reward import (
    LoRAConfig,
    compute_grpo_advantages,
    compute_score_reward,
)
from tpu_distil.trajectory import Step, Trajectory


def test_score_reward_orders_self_correction_above_failure() -> None:
    recovered_traj = Trajectory(
        task_id="tb-201",
        prompt="Fix broken SQL query script",
        steps=[
            Step(
                thought="Initial failing run",
                action="python3 query.py",
                observation="SyntaxError",
                exit_code=1,
                unit_test_pass_rate=0.0,
            ),
            Step(
                thought="Fix SQL syntax and rerun pytest",
                action="sed -i 's/SELEC /SELECT /' query.py && pytest",
                observation="4 passed",
                exit_code=0,
                unit_test_pass_rate=1.0,
                teacher_generated=True,
            ),
        ],
        terminal_reward=1.0,
        splice_step=1,
    )
    failed_traj = Trajectory(
        task_id="tb-201",
        prompt="Fix broken SQL query script",
        steps=[
            Step(
                thought="Initial failing run",
                action="python3 query.py",
                observation="SyntaxError",
                exit_code=1,
                unit_test_pass_rate=0.0,
            ),
            Step(
                thought="Still broken",
                action="python3 query.py",
                observation="SyntaxError",
                exit_code=1,
                unit_test_pass_rate=0.0,
            ),
        ],
        terminal_reward=0.0,
        splice_step=1,
    )

    r_recovered = compute_score_reward(recovered_traj, splice_step=1, kl_per_step=[0.05])
    r_failed = compute_score_reward(failed_traj, splice_step=1, kl_per_step=[0.05])

    assert r_recovered > 1.25
    assert r_failed < 0.0
    assert r_recovered > r_failed


def test_lora_config_freezes_moe_router_and_enforces_256_alignment() -> None:
    cfg = LoRAConfig()
    assert cfg.freeze_moe_router is True
    assert "gate_proj" not in cfg.target_modules
    assert cfg.pad_multiple % 256 == 0
    assert cfg.max_seq_len % 256 == 0
    assert cfg.global_batch_size % 256 == 0

    with pytest.raises(ValueError, match="gate_proj"):
        LoRAConfig(target_modules=("q_proj", "gate_proj"))

    with pytest.raises(ValueError, match="256"):
        LoRAConfig(global_batch_size=128)


def test_compute_grpo_advantages_zero_mean_unit_variance() -> None:
    rewards = [0.0, 0.2, 0.5, 1.4, 1.5, 0.0, 1.4, 0.0]
    adv = compute_grpo_advantages(rewards)
    assert len(adv) == 8
    assert abs(sum(adv)) < 1e-6
    assert adv[4] > adv[0]
