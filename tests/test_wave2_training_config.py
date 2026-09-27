"""Wave 2 training configuration, HBM budget, and 4-arm live Terminal-Bench evaluation tests."""

from __future__ import annotations

import json
from pathlib import Path

from tpu_distil.score_reward import LoRAConfig

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_wave2_lora_config_and_terminal_bench_gate() -> None:
    cfg = LoRAConfig()
    assert cfg.freeze_moe_router is True
    assert "gate_proj" not in cfg.target_modules
    assert cfg.rank == 64
    assert cfg.alpha == 128
    assert cfg.pad_multiple == 256

    eval_path = REPO_ROOT / ".agents/wave-2/logs/terminal_bench_eval.json"
    assert eval_path.exists(), ".agents/wave-2/logs/terminal_bench_eval.json must exist"
    report = json.loads(eval_path.read_text(encoding="utf-8"))
    assert report["hardware"]["jax_tpu_detected"] is True
    assert report["hardware"]["num_tpu_chips"] == 8
    assert report["hardware"]["mxu_256x256_lora_step_verified"] is True

    # Verify real JAX/Optax Stage 1 cross-entropy loss curves & Stage 2 GRPO loss history
    for sft_key in ("bc_control", "score_sft"):
        sft_info = report["stage1_sft"][sft_key]
        assert len(sft_info["loss_history"]) >= 16
        assert sft_info["final_sft_loss"] < sft_info["initial_sft_loss"]
        assert sft_info["moe_router_collapse_events"] == 0

    assert len(report["stage2_grpo"]["grpo_loss_history"]) >= 8

    # Verify all 4 arms have 50 real per-task Terminal-Bench results & live token usage
    for arm_name in ("Zero-Shot", "BC-Control", "SCoRe-SFT", "SCoRe-RL"):
        arm_data = report["terminal_bench_4_arm_eval"][arm_name]
        assert arm_data["live_endpoint_responded"] is True
        assert arm_data["num_tasks"] == 50
        assert len(arm_data["task_results"]) == 50
        assert arm_data["total_prompt_tokens"] > 0
        assert arm_data["total_completion_tokens"] > 0

    # Clause 2.1: >= +10.0% absolute Pass@1 improvement over Zero-Shot on Terminal-Bench
    metrics = report["metrics_summary"]
    assert metrics["absolute_pass_at_1_gain_over_zero_shot"] >= 0.10

    # Clause 2.2: Strict ablation ordering SCoRe-RL > SCoRe-SFT > BC-Control > Zero-Shot
    assert (
        metrics["score_rl_pass_at_1"]
        > metrics["score_sft_pass_at_1"]
        > metrics["bc_control_pass_at_1"]
        > metrics["zero_shot_pass_at_1"]
    )

    # Clause 2.3: Positive wrong -> right self-correction transition gain >= +15.0% over BC-Control
    assert metrics["wrong_to_right_transition_gain_over_bc"] >= 0.15

    # Clause 2.4: Peak HBM per chip <= 28.0 GB / 32.0 GB and 0 OOMs
    assert metrics["peak_hbm_gb_per_chip"] <= 28.0
    assert report["stage2_grpo"]["oom_events"] == 0
    assert report["stage2_grpo"]["moe_router_collapse_events"] == 0

    # Verify saved LoRA .safetensors checkpoints exist and are non-empty
    for ckpt_name in (
        "bc_control_lora.safetensors",
        "score_sft_lora.safetensors",
        "score_rl_lora.safetensors",
    ):
        ckpt_file = REPO_ROOT / ".agents/wave-2/checkpoints" / ckpt_name
        assert ckpt_file.exists() and ckpt_file.stat().st_size > 1_000_000, (
            f"Missing or empty trained LoRA checkpoint: {ckpt_file}"
        )

    # Verify paired bootstrap 95% CI
    ci_info = metrics.get("paired_bootstrap_95ci", {}).get("score_rl_vs_zero_shot", {})
    assert ci_info.get("ci_95_low", 0.0) > 0.0
    assert report["gate_verdict"] == "GREEN"

