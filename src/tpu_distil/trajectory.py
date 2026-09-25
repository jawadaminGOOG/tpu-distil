"""Cross-tokenizer trajectory schema and Qwen3-Next TPU v6e serializer.

Enforces two hard invariants from AGENTS.md:
1. Cross-tokenizer safety: Teacher (DeepSeek-V4.1-Flash / Kimi-K3 / GLM-5.3) steps
   are stored as structured text `Step(thought, action, observation)` objects and
   tokenized exclusively inside the Qwen3-Next student boundary.
2. Observation loss masking & TPU alignment: All prompt, environment `observation`
   (stdout/stderr), student error prefix (< splice_step), and `<|PAD|>` tokens
   receive `loss_mask = 0.0`, and sequence lengths are padded to multiples of 256.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class Step:
    """Single agentic interaction turn inside an executable sandbox."""

    thought: str
    action: str
    observation: str
    exit_code: int
    unit_test_pass_rate: float = 0.0
    teacher_generated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Step:
        return cls(**data)


@dataclass(frozen=True)
class Trajectory:
    """Multi-turn trajectory across student rollout and teacher recovery."""

    task_id: str
    prompt: str
    steps: list[Step] = field(default_factory=list)
    terminal_reward: float = 0.0
    splice_step: int = 0
    teacher_model: str = "DeepSeek-V4.1-Flash"
    state_s_m_diff: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "prompt": self.prompt,
            "steps": [s.to_dict() for s in self.steps],
            "terminal_reward": self.terminal_reward,
            "splice_step": self.splice_step,
            "teacher_model": self.teacher_model,
            "state_s_m_diff": self.state_s_m_diff,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Trajectory:
        steps = [Step.from_dict(s) for s in data.get("steps", [])]
        return cls(
            task_id=data["task_id"],
            prompt=data["prompt"],
            steps=steps,
            terminal_reward=float(data.get("terminal_reward", 0.0)),
            splice_step=int(data.get("splice_step", 0)),
            teacher_model=str(data.get("teacher_model", "DeepSeek-V4.1-Flash")),
            state_s_m_diff=str(data.get("state_s_m_diff", "")),
        )


def _default_byte_tokenizer(text: str) -> list[int]:
    """Deterministic fallback tokenizer when HF tokenizer is not injected."""
    return [b + 1 for b in text.encode("utf-8")]


def serialize_for_qwen3_next(
    traj: Trajectory,
    tokenize_fn: Callable[[str], list[int]] = _default_byte_tokenizer,
    pad_multiple: int = 256,
    pad_token_id: int = 0,
) -> dict[str, list[int] | list[float] | list[str]]:
    """Serialize a Trajectory to Qwen3-Next token IDs with strict observation masking.

    Args:
        traj: Structured trajectory containing student prefix and teacher recovery.
        tokenize_fn: Qwen3-Next tokenizer callable mapping string -> list[int].
        pad_multiple: TPU v6e systolic array alignment multiple (must be % 256 == 0).
        pad_token_id: Token ID used for right-padding to `pad_multiple`.

    Returns:
        Dictionary with 256-aligned `input_ids`, `loss_mask`, and `segment_types`.
    """
    if pad_multiple <= 0 or pad_multiple % 256 != 0:
        raise ValueError(
            f"TPU v6e dimension alignment requires pad_multiple % 256 == 0, got {pad_multiple}"
        )

    input_ids: list[int] = []
    loss_mask: list[float] = []
    segment_types: list[str] = []

    def _append_span(text: str, mask_val: float, seg_type: str) -> None:
        ids = tokenize_fn(text)
        input_ids.extend(ids)
        loss_mask.extend([mask_val] * len(ids))
        segment_types.extend([seg_type] * len(ids))

    # 1. Task prompt is never trained on (loss_mask = 0.0)
    _append_span(f"<|im_start|>user\n{traj.prompt}\n<|im_end|>\n", 0.0, "prompt")

    # 2. Iterate through steps; train ONLY on thought+action at t >= splice_step
    for idx, step in enumerate(traj.steps):
        is_recovery_turn = idx >= traj.splice_step
        action_weight = 1.0 if is_recovery_turn else 0.0

        assistant_block = (
            f"<|im_start|>assistant\n<think>{step.thought}</think>\n"
            f"<action>{step.action}</action>\n<|im_end|>\n"
        )
        _append_span(assistant_block, action_weight, "assistant_action")

        # HARD INVARIANT: Environment observation tokens ALWAYS get loss_mask = 0.0
        obs_block = f"<|im_start|>environment\n{step.observation}\n<|im_end|>\n"
        _append_span(obs_block, 0.0, "observation")

    # 3. Pad sequence length to next multiple of `pad_multiple` (256)
    remainder = len(input_ids) % pad_multiple
    if remainder != 0 or len(input_ids) == 0:
        pad_len = pad_multiple - remainder if len(input_ids) > 0 else pad_multiple
        input_ids.extend([pad_token_id] * pad_len)
        loss_mask.extend([0.0] * pad_len)
        segment_types.extend(["pad"] * pad_len)

    # Assert observation tokens never leak non-zero loss weights
    for mask_val, seg_type in zip(loss_mask, segment_types):
        if seg_type in ("observation", "prompt", "pad") and mask_val != 0.0:
            raise AssertionError(f"Non-zero loss_mask ({mask_val}) on segment '{seg_type}'")

    return {
        "input_ids": input_ids,
        "loss_mask": loss_mask,
        "segment_types": segment_types,
    }
