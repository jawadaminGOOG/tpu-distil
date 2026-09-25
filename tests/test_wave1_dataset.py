"""Wave 1 dataset integrity and masking tests (`plans/wave-1-mps-pipeline-qwen3-next-tests.md`)."""

from __future__ import annotations

import json
from pathlib import Path

from tpu_distil.trajectory import Trajectory, serialize_for_qwen3_next

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_wave1_mps_and_bc_datasets_meet_all_gate_clauses() -> None:
    mps_path = REPO_ROOT / "data/score_sft_mps.jsonl"
    bc_path = REPO_ROOT / "data/bc_control.jsonl"
    assert mps_path.exists(), "data/score_sft_mps.jsonl must exist"
    assert bc_path.exists(), "data/bc_control.jsonl must exist"

    mps_lines = [line for line in mps_path.read_text().splitlines() if line.strip()]
    bc_lines = [line for line in bc_path.read_text().splitlines() if line.strip()]

    # Clause 1.1 & 1.2: >= 2,048 verified spliced trajectories and >= 2,048 BC-Control trajectories
    assert len(mps_lines) >= 2048
    assert len(bc_lines) >= 2048

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
        for mask_val, seg_type in zip(out["loss_mask"], out["segment_types"]):
            if seg_type in ("observation", "prompt", "pad"):
                assert mask_val == 0.0
