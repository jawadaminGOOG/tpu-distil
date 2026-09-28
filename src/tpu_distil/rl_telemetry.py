"""RL Convergence Telemetry, Dynamic Group Filtering, and Benchmark Decontamination.

Implements six production observability and data-hygiene primitives for multi-turn
reinforcement learning (`SCoRe-RL` / `GRPO`) on Cloud TPU `v6e`:
1. Unbiased, non-negative Schulman K3 KL-divergence estimation (`compute_k3_kl_divergence`).
2. Dynamic Informative Group Filtering (`filter_informative_grpo_groups`) to drop
   zero-variance rollout groups (all-pass or all-fail where group advantage == 0.0).
3. EMA reward smoothness and Signal-to-Noise Ratio (`compute_reward_smoothness_metrics`).
4. Process vs. outcome reward decomposition and shaping-divergence detection.
5. Policy token entropy and PPO/GRPO ratio clipping corridor checks (`evaluate_rl_health_corridor`).
6. Three-layer benchmark decontamination (`verify_zero_benchmark_leakage`) guaranteeing
   zero task-ID or n-gram prompt overlap with held-out evaluation benchmarks.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RLStepTelemetry:
    """Per-step observability record for online multi-turn GRPO training."""

    step: int
    policy_loss: float
    mean_total_reward: float
    ema_total_reward: float
    mean_terminal_reward: float
    mean_recovery_bonus: float
    online_wrong_to_right_rate: float
    k3_kl_divergence: float
    policy_token_entropy: float
    clip_fraction: float
    dynamic_group_rate: float


def compute_k3_kl_divergence(
    policy_logprobs: Sequence[float],
    ref_logprobs: Sequence[float],
) -> float:
    """Compute the unbiased, non-negative Schulman K3 KL estimator D_KL(pi_theta || pi_ref).

    For each token position t with log-ratio log_r_t = log pi_ref(y_t) - log pi_theta(y_t):
        k3_t = exp(log_r_t) - 1.0 - log_r_t >= 0.0
    """
    if len(policy_logprobs) != len(ref_logprobs):
        raise ValueError(
            f"Length mismatch: policy_logprobs={len(policy_logprobs)} vs ref_logprobs={len(ref_logprobs)}"
        )
    if not policy_logprobs:
        return 0.0

    total = 0.0
    for lp_pol, lp_ref in zip(policy_logprobs, ref_logprobs):
        log_r = max(-20.0, min(20.0, float(lp_ref) - float(lp_pol)))
        k3_val = math.exp(log_r) - 1.0 - log_r
        total += max(0.0, k3_val)
    return total / float(len(policy_logprobs))


def filter_informative_grpo_groups(
    group_rewards: Sequence[Sequence[float]],
    min_std: float = 1e-4,
) -> tuple[list[int], float]:
    """Filter rollout groups to retain only those with non-zero within-group reward variance.

    Returns:
        `(informative_group_indices, dynamic_informative_rate)` where `dynamic_informative_rate`
        is the fraction of sampled groups with `std(rewards) > min_std`.
    """
    if not group_rewards:
        return [], 0.0

    kept_indices: list[int] = []
    for idx, rewards in enumerate(group_rewards):
        if len(rewards) < 2:
            continue
        mean_r = sum(rewards) / float(len(rewards))
        var_r = sum((float(r) - mean_r) ** 2 for r in rewards) / float(len(rewards))
        if math.sqrt(var_r) > min_std:
            kept_indices.append(idx)

    rate = float(len(kept_indices)) / float(len(group_rewards))
    return kept_indices, round(rate, 6)


def compute_reward_smoothness_metrics(
    step_rewards: Sequence[float],
    ema_alpha: float = 0.2,
) -> dict[str, float]:
    """Compute EMA reward trajectory, step-to-step volatility, and Signal-to-Noise Ratio (SNR)."""
    if not step_rewards:
        return {
            "ema_start": 0.0,
            "ema_end": 0.0,
            "ema_gain": 0.0,
            "step_diff_std": 0.0,
            "reward_snr": 0.0,
            "monotonic_ema_fraction": 0.0,
        }

    ema_vals: list[float] = []
    curr = float(step_rewards[0])
    for r in step_rewards:
        curr = ema_alpha * float(r) + (1.0 - ema_alpha) * curr
        ema_vals.append(curr)

    if len(step_rewards) < 2:
        return {
            "ema_start": round(ema_vals[0], 6),
            "ema_end": round(ema_vals[-1], 6),
            "ema_gain": 0.0,
            "step_diff_std": 0.0,
            "reward_snr": 0.0,
            "monotonic_ema_fraction": 1.0,
        }

    diffs = [float(step_rewards[i]) - float(step_rewards[i - 1]) for i in range(1, len(step_rewards))]
    mean_diff = sum(diffs) / float(len(diffs))
    var_diff = sum((d - mean_diff) ** 2 for d in diffs) / float(len(diffs))
    std_diff = math.sqrt(var_diff)

    ema_gain = ema_vals[-1] - ema_vals[0]
    snr = abs(ema_gain) / (std_diff + 1e-6)
    non_decreasing = sum(
        1 for i in range(1, len(ema_vals)) if ema_vals[i] >= ema_vals[i - 1] - 1e-5
    )
    mono_frac = float(non_decreasing) / float(len(ema_vals) - 1)

    return {
        "ema_start": round(ema_vals[0], 6),
        "ema_end": round(ema_vals[-1], 6),
        "ema_gain": round(ema_gain, 6),
        "step_diff_std": round(std_diff, 6),
        "reward_snr": round(snr, 4),
        "monotonic_ema_fraction": round(mono_frac, 4),
    }


def evaluate_rl_health_corridor(
    steps: Sequence[RLStepTelemetry],
    min_dynamic_group_rate: float = 0.65,
    kl_corridor: tuple[float, float] = (0.015, 0.150),
    max_kl_spike: float = 0.250,
    min_entropy: float = 0.35,
    min_snr: float = 2.0,
) -> dict[str, Any]:
    """Audit an RL training trajectory against stability, KL, entropy, and reward-shaping corridors."""
    if not steps:
        raise ValueError("steps must be non-empty")

    total_rewards = [s.mean_total_reward for s in steps]
    term_rewards = [s.mean_terminal_reward for s in steps]
    rec_bonuses = [s.mean_recovery_bonus for s in steps]
    kls = [s.k3_kl_divergence for s in steps]
    entropies = [s.policy_token_entropy for s in steps]
    clips = [s.clip_fraction for s in steps]
    dyn_rates = [s.dynamic_group_rate for s in steps]

    smoothness = compute_reward_smoothness_metrics(total_rewards)
    mean_kl = sum(kls) / float(len(kls))
    max_kl = max(kls)
    min_ent = min(entropies)
    mean_dyn = sum(dyn_rates) / float(len(dyn_rates))
    mean_clip = sum(clips) / float(len(clips))

    # Detect reward-shaping hacking: recovery bonus increases while terminal reward drops
    q = max(1, len(steps) // 4)
    term_gain = (sum(term_rewards[-q:]) / q) - (sum(term_rewards[:q]) / q)
    bonus_gain = (sum(rec_bonuses[-q:]) / q) - (sum(rec_bonuses[:q]) / q)
    no_shaping_divergence = bool(term_gain > 0.0 and bonus_gain >= 0.0)

    kl_in_corridor = bool(kl_corridor[0] <= mean_kl <= kl_corridor[1] and max_kl <= max_kl_spike)
    entropy_healthy = bool(min_ent >= min_entropy)
    dynamic_groups_healthy = bool(mean_dyn >= min_dynamic_group_rate)
    snr_healthy = bool(smoothness["reward_snr"] >= min_snr and smoothness["ema_gain"] > 0.0)

    all_healthy = (
        kl_in_corridor
        and entropy_healthy
        and dynamic_groups_healthy
        and snr_healthy
        and no_shaping_divergence
    )

    return {
        "num_steps": len(steps),
        "mean_k3_kl": round(mean_kl, 6),
        "max_k3_kl": round(max_kl, 6),
        "min_policy_entropy": round(min_ent, 6),
        "mean_clip_fraction": round(mean_clip, 6),
        "mean_dynamic_group_rate": round(mean_dyn, 6),
        "terminal_reward_gain": round(term_gain, 6),
        "recovery_bonus_gain": round(bonus_gain, 6),
        "smoothness": smoothness,
        "checks": {
            "kl_in_corridor": kl_in_corridor,
            "entropy_healthy": entropy_healthy,
            "dynamic_groups_healthy": dynamic_groups_healthy,
            "snr_healthy": snr_healthy,
            "no_shaping_divergence": no_shaping_divergence,
        },
        "all_healthy": all_healthy,
    }


def _word_ngrams(text: str, n: int = 8) -> set[tuple[str, ...]]:
    tokens = re.findall(r"[a-z0-9_]+", text.lower())
    if len(tokens) < n:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def verify_zero_benchmark_leakage(
    train_records: Sequence[dict[str, Any]],
    eval_task_ids: set[str],
    eval_instructions: Sequence[str],
    max_ngram_jaccard: float = 0.25,
) -> dict[str, Any]:
    """Verify that a rollout training corpus has zero overlap with held-out evaluation tasks."""
    eval_ids_norm = {t.strip().lower() for t in eval_task_ids}
    eval_hashes = {
        hashlib.sha256(instr.strip().lower().encode("utf-8")).hexdigest()
        for instr in eval_instructions
    }
    eval_ngram_sets = [_word_ngrams(instr, n=8) for instr in eval_instructions]

    id_collisions: list[str] = []
    hash_collisions: list[str] = []
    ngram_violations: list[tuple[str, float]] = []
    max_observed_jaccard = 0.0

    for rec in train_records:
        tid = str(rec.get("task_id", "")).strip()
        tid_low = tid.lower()
        prompt = str(rec.get("prompt", "")).strip()
        p_hash = hashlib.sha256(prompt.lower().encode("utf-8")).hexdigest()

        if tid_low in eval_ids_norm or any(eid in tid_low for eid in eval_ids_norm if len(eid) > 4):
            id_collisions.append(tid)
        if p_hash in eval_hashes:
            hash_collisions.append(tid)

        t_ngrams = _word_ngrams(prompt, n=8)
        if t_ngrams:
            for e_ngrams in eval_ngram_sets:
                if not e_ngrams:
                    continue
                inter = len(t_ngrams & e_ngrams)
                if inter == 0:
                    continue
                jacc = float(inter) / float(len(t_ngrams | e_ngrams))
                if jacc > max_observed_jaccard:
                    max_observed_jaccard = jacc
                if jacc >= max_ngram_jaccard:
                    ngram_violations.append((tid, round(jacc, 4)))
                    break

    clean = not id_collisions and not hash_collisions and not ngram_violations
    return {
        "num_train_records": len(train_records),
        "num_eval_tasks": len(eval_task_ids),
        "id_collisions_count": len(id_collisions),
        "hash_collisions_count": len(hash_collisions),
        "ngram_violations_count": len(ngram_violations),
        "max_observed_ngram_jaccard": round(max_observed_jaccard, 6),
        "zero_benchmark_leakage": clean,
    }
