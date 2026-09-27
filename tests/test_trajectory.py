"""Unit tests for `tpu_distil.trajectory` (Wave 0 Clause 0.1)."""

from __future__ import annotations

import pytest

from tpu_distil.trajectory import Step, Trajectory, serialize_for_qwen3_next


def test_observation_loss_mask_is_strictly_zero() -> None:
    traj = Trajectory(
        task_id="tb-001",
        prompt="Fix broken nginx config and verify port 8080 responds.",
        steps=[
            Step(
                thought="Check current nginx config.",
                action="nginx -t",
                observation="nginx: [emerg] unexpected '}' in /etc/nginx/nginx.conf:42",
                exit_code=1,
                unit_test_pass_rate=0.0,
                teacher_generated=False,
            ),
            Step(
                thought="Remove stray brace on line 42 and reload nginx.",
                action="sed -i '42s/}//' /etc/nginx/nginx.conf && nginx -t",
                observation="nginx: configuration file /etc/nginx/nginx.conf test is successful",
                exit_code=0,
                unit_test_pass_rate=1.0,
                teacher_generated=True,
            ),
        ],
        terminal_reward=1.0,
        splice_step=1,
    )

    out = serialize_for_qwen3_next(traj, pad_multiple=256)
    input_ids = out["input_ids"]
    loss_mask = out["loss_mask"]
    segment_types = out["segment_types"]

    assert len(input_ids) == len(loss_mask) == len(segment_types)
    assert len(input_ids) % 256 == 0

    for mask_val, seg in zip(loss_mask, segment_types):
        if seg in ("observation", "prompt", "pad"):
            assert mask_val == 0.0, f"Segment {seg} leaked non-zero loss weight {mask_val}"


def test_recovery_suffix_loss_mask_is_one_and_prefix_is_zero() -> None:
    traj = Trajectory(
        task_id="tb-002",
        prompt="Create a systemd unit check script.",
        steps=[
            Step(
                thought="Student turn 0 fails.",
                action="bad_cmd --invalid",
                observation="bash: bad_cmd: command not found",
                exit_code=127,
                teacher_generated=False,
            ),
            Step(
                thought="Teacher turn 1 fixes the command.",
                action="systemctl list-units --type=service",
                observation="UNIT LOAD ACTIVE SUB DESCRIPTION",
                exit_code=0,
                teacher_generated=True,
            ),
        ],
        terminal_reward=1.0,
        splice_step=1,
    )

    # Use a word-level tokenizer that lets us inspect spans
    out = serialize_for_qwen3_next(traj, pad_multiple=256)
    loss_mask = out["loss_mask"]
    segment_types = out["segment_types"]

    assistant_weights = [m for m, s in zip(loss_mask, segment_types) if s == "assistant_action"]
    # First assistant turn (idx=0 < splice_step=1) must be 0.0; second turn (idx=1 >= 1) must be 1.0
    assert 0.0 in assistant_weights
    assert 1.0 in assistant_weights


def test_tpu_v6e_256_sequence_alignment_enforced() -> None:
    traj = Trajectory(task_id="tb-003", prompt="Test alignment", steps=[])
    with pytest.raises(ValueError, match="256"):
        serialize_for_qwen3_next(traj, pad_multiple=128)

    out_256 = serialize_for_qwen3_next(traj, pad_multiple=256, pad_token_id=151643)
    out_512 = serialize_for_qwen3_next(traj, pad_multiple=512, pad_token_id=151643)
    assert len(out_256["input_ids"]) == 256
    assert len(out_512["input_ids"]) == 512

    non_pad_count = sum(1 for s in out_256["segment_types"] if s != "pad")
    assert 0 < non_pad_count < 256
    # Verify exact prefix preservation and right-padding boundary
    assert out_256["input_ids"][:non_pad_count] == out_512["input_ids"][:non_pad_count]
    assert out_256["input_ids"][non_pad_count:] == [151643] * (256 - non_pad_count)
    assert out_512["input_ids"][non_pad_count:] == [151643] * (512 - non_pad_count)
    assert out_512["loss_mask"][non_pad_count:] == [0.0] * (512 - non_pad_count)
    assert out_512["segment_types"][non_pad_count:] == ["pad"] * (512 - non_pad_count)

