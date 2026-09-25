"""SCoRe dense process-reward shaper, GRPO advantage estimator, and TPU v6e LoRA config.

Implements the SCoRe-RL reward formulation (arXiv:2509.14257v3) adapted for
multi-turn executable environments (`Terminal-Bench`, `R2E-Gym`, `SWE-Gym`):
  R(tau) = R_terminal(tau)
           + sum_{t=m}^T gamma^{t-m} * alpha * (Phi(s_t, a_t) - Phi(s_{t-1}, a_{t-1}))
           + r_recovery(m)
           - beta * sum_{t=m}^T D_KL(pi_theta || pi_ref)

Also defines `LoRAConfig` for `Qwen3-Next-80B-A3B-Instruct` on Cloud TPU `v6e`,
enforcing `freeze_moe_router=True` (`gate_proj` excluded) and 256-aligned dimensions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from tpu_distil.trajectory import Trajectory


@dataclass(frozen=True)
class LoRAConfig:
    """TPU v6e LoRA configuration for Qwen3-Next-80B-A3B-Instruct."""

    rank: int = 64
    alpha: int = 128
    dropout: float = 0.0
    freeze_moe_router: bool = True
    pad_multiple: int = 256
    max_seq_len: int = 8192
    global_batch_size: int = 256
    short_horizon_rem_steps: int = 4
    grpo_group_size: int = 8
    target_modules: tuple[str, ...] = field(
        default_factory=lambda: (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "down_proj",
        )
    )

    def __post_init__(self) -> None:
        if self.pad_multiple <= 0 or self.pad_multiple % 256 != 0:
            raise ValueError(f"pad_multiple ({self.pad_multiple}) must be a multiple of 256")
        if self.max_seq_len <= 0 or self.max_seq_len % 256 != 0:
            raise ValueError(f"max_seq_len ({self.max_seq_len}) must be a multiple of 256")
        if self.global_batch_size <= 0 or self.global_batch_size % 256 != 0:
            raise ValueError(
                f"global_batch_size ({self.global_batch_size}) must be a multiple of 256"
            )
        if self.freeze_moe_router and "gate_proj" in self.target_modules:
            raise ValueError(
                "MoE router 'gate_proj' must be excluded from target_modules when freeze_moe_router=True"
            )


def compute_score_reward(
    traj: Trajectory,
    splice_step: int | None = None,
    alpha: float = 0.2,
    beta: float = 0.02,
    gamma: float = 0.99,
    recovery_bonus: float = 0.30,
    kl_per_step: list[float] | None = None,
) -> float:
    """Compute the SCoRe dense process reward + self-correction transition bonus.

    Args:
        traj: Multi-turn trajectory evaluated in the container sandbox.
        splice_step: Index `m` where recovery begins (defaults to `traj.splice_step`).
        alpha: Weight on step-wise unit-test progress potential delta.
        beta: KL divergence penalty coefficient against `SCoRe-SFT` reference policy.
        gamma: Per-turn discount factor.
        recovery_bonus: Bonus awarded when recovery turns resolve an earlier error
            and achieve `terminal_reward >= 1.0`.
        kl_per_step: Optional per-step KL divergence estimates `D_KL(pi_theta || pi_ref)`.

    Returns:
        Scalar shaped SCoRe reward for the trajectory.
    """
    m = traj.splice_step if splice_step is None else splice_step
    total_reward = float(traj.terminal_reward)

    if not traj.steps:
        return total_reward

    prev_potential = traj.steps[m - 1].unit_test_pass_rate if m > 0 else 0.0
    had_prior_error = any(s.exit_code != 0 for s in traj.steps[: max(m, 1)])

    for idx in range(m, len(traj.steps)):
        step = traj.steps[idx]
        discount = gamma ** (idx - m)
        delta_potential = step.unit_test_pass_rate - prev_potential
        total_reward += discount * alpha * delta_potential
        prev_potential = step.unit_test_pass_rate

    # Explicit SCoRe wrong -> right transition bonus
    if (had_prior_error or m > 0) and traj.terminal_reward >= 1.0 and traj.steps[-1].exit_code == 0:
        total_reward += recovery_bonus

    if kl_per_step is not None:
        total_reward -= beta * sum(kl_per_step)

    return total_reward


def compute_grpo_advantages(rewards: list[float], eps: float = 1e-8) -> list[float]:
    """Compute group-normalized GRPO advantages across `G` short-horizon rollouts."""
    if not rewards:
        return []
    mean_r = sum(rewards) / len(rewards)
    var_r = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
    std_r = math.sqrt(var_r)
    if std_r < eps:
        return [0.0 for _ in rewards]
    return [(r - mean_r) / (std_r + eps) for r in rewards]
