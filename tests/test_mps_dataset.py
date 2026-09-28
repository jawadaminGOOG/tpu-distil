"""Dataset integrity, masking, and customer-defensibility tests."""

from __future__ import annotations

import json
from pathlib import Path

from tpu_distil.trajectory import (
    Trajectory,
    get_qwen3_tokenizer,
    serialize_for_qwen3_next,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_mps_and_bc_datasets_meet_all_gate_clauses() -> None:
    mps_path = REPO_ROOT / "data/score_sft_mps.jsonl"
    bc_path = REPO_ROOT / "data/bc_control.jsonl"
    summary_matches = sorted((REPO_ROOT / ".agents").glob("*/logs/mps_collection_summary.json"))
    assert mps_path.exists(), "data/score_sft_mps.jsonl must exist"
    assert bc_path.exists(), "data/bc_control.jsonl must exist"
    assert summary_matches, "mps_collection_summary.json must exist"
    summary_path = summary_matches[-1]

    mps_lines = [line for line in mps_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    bc_lines = [line for line in bc_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    # Clause 1.1 & 1.2: >= 2,048 verified spliced trajectories and >= 2,048 BC-Control trajectories
    assert len(mps_lines) >= 2048
    assert len(bc_lines) >= 2048

    # Anti-simulation check: all 2,048 prompts and state_s_m_diff snapshots must be unique
    prompts = {json.loads(line)["prompt"] for line in mps_lines}
    diffs = {json.loads(line)["state_s_m_diff"] for line in mps_lines}
    assert len(prompts) == len(mps_lines), "All 2,048 prompts must be unique (no cloned templates)"
    assert len(diffs) == len(mps_lines), "All 2,048 state_s_m_diff diffs must be unique"
    assert summary["student_live_tpu_tokens"] > 0
    assert summary["teacher_live_tpu_tokens"] > 0
    assert summary.get("live_student_tpu_calls", 0) >= 2048
    assert summary.get("live_teacher_tpu_calls", 0) >= 2048
    assert summary.get("unique_step2_recovery_skeletons_count", 0) >= 1500
    assert summary["eval_task_overlap_count"] == 0

    tok = get_qwen3_tokenizer()
    assert tok.get_vocab_size() >= 151643

    # Clause 1.3 & 1.4: Check splice_step >= 1, terminal_reward == 1.0, state_s_m_diff, and 256 alignment
    for idx in (0, 1, 17, 511, 1023, 2047):
        traj = Trajectory.from_dict(json.loads(mps_lines[idx]))
        assert 1 <= traj.splice_step < len(traj.steps)
        assert traj.terminal_reward == 1.0
        assert len(traj.state_s_m_diff.strip()) > 0
        assert traj.steps[-1].teacher_generated is True
        assert traj.steps[0].teacher_generated is False

        out = serialize_for_qwen3_next(traj, pad_multiple=256)
        assert len(out["input_ids"]) % 256 == 0
        assert 151644 in out["input_ids"]  # Real Qwen3 <|im_start|> token ID
        assert 151645 in out["input_ids"]  # Real Qwen3 <|im_end|> token ID
        for mask_val, seg_type in zip(out["loss_mask"], out["segment_types"]):
            if seg_type in ("observation", "prompt", "pad"):
                assert mask_val == 0.0
