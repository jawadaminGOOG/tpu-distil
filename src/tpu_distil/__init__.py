"""TPU-Distil: 4-File On-Policy SCoRe-RL Distillation Engine for Cloud TPU v6e."""

from tpu_distil.sandbox import ContainerSandbox, SandboxResult
from tpu_distil.score_reward import (
    LoRAConfig,
    compute_grpo_advantages,
    compute_score_reward,
)
from tpu_distil.splicer import (
    build_teacher_recovery_prompt,
    find_first_error_step,
    splice_trajectory,
)
from tpu_distil.trajectory import Step, Trajectory, serialize_for_qwen3_next

__all__ = [
    "ContainerSandbox",
    "LoRAConfig",
    "SandboxResult",
    "Step",
    "Trajectory",
    "build_teacher_recovery_prompt",
    "compute_grpo_advantages",
    "compute_score_reward",
    "find_first_error_step",
    "serialize_for_qwen3_next",
    "splice_trajectory",
]
