"""Mixed-Policy Splicing (MPS) first-error detector and Teacher recovery splicer.

Implements the core SCoRe-RL data collection mechanism:
1. Roll out the student (`Qwen3-Next-80B-A3B-Instruct` on TPU v6e) until the first
   step `m` where `exit_code != 0` or action syntax fails.
2. Prompt the Teacher (`DeepSeek-V4.1-Flash` on TPU v6e-16) with the task context,
   student prefix `0..m-1`, the student's failing action `a_m^-`, and the error
   observation `o_m^-`.
3. Splice the Teacher's verified recovery suffix `(a_m^*, o_m^*, ..., a_T^*)` onto
   the student prefix `0..m-1` so `SCoRe-SFT` and `SCoRe-RL` learn how to recover
   from authentic student errors.
"""

from __future__ import annotations

from tpu_distil.trajectory import Step, Trajectory


def find_first_error_step(steps: list[Step]) -> int | None:
    """Return the 0-based index `m` of the first failing step, or None if all pass."""
    for idx, step in enumerate(steps):
        if step.exit_code != 0 or not step.action.strip():
            return idx
    return None


def splice_trajectory(
    student_traj: Trajectory,
    teacher_suffix: list[Step],
    terminal_reward: float,
    teacher_model: str = "DeepSeek-V4.1-Flash",
    state_s_m_diff: str = "",
    include_failed_step_in_prefix: bool = False,
) -> Trajectory:
    """Splice a Teacher recovery suffix at the student's first error step `m`.

    When `include_failed_step_in_prefix=False` (default), preserves `student_traj.steps[:m]`
    and replaces step `m` with `teacher_suffix`. When `include_failed_step_in_prefix=True`,
    preserves `student_traj.steps[:m+1]` (including the student's failing step `(a_m^-, o_m^-)`
    with `teacher_generated=False` / `loss_mask=0.0`) and appends `teacher_suffix` at `m+1`.
    """
    if not teacher_suffix:
        raise ValueError("teacher_suffix must contain at least one recovery Step")

    first_err = find_first_error_step(student_traj.steps)
    if first_err is None:
        splice_idx = len(student_traj.steps)
    else:
        splice_idx = first_err + 1 if include_failed_step_in_prefix else first_err

    student_prefix = [
        Step(
            thought=s.thought,
            action=s.action,
            observation=s.observation,
            exit_code=s.exit_code,
            unit_test_pass_rate=s.unit_test_pass_rate,
            teacher_generated=False,
        )
        for s in student_traj.steps[:splice_idx]
    ]
    marked_teacher_suffix = [
        Step(
            thought=s.thought,
            action=s.action,
            observation=s.observation,
            exit_code=s.exit_code,
            unit_test_pass_rate=s.unit_test_pass_rate,
            teacher_generated=True,
        )
        for s in teacher_suffix
    ]

    return Trajectory(
        task_id=student_traj.task_id,
        prompt=student_traj.prompt,
        steps=student_prefix + marked_teacher_suffix,
        terminal_reward=terminal_reward,
        splice_step=splice_idx,
        teacher_model=teacher_model,
        state_s_m_diff=state_s_m_diff or student_traj.state_s_m_diff,
    )


def build_teacher_recovery_prompt(
    task_prompt: str,
    student_prefix: list[Step],
    failed_step: Step,
) -> str:
    """Build the Teacher recovery prompt conditioned on the student's error at `s_m`."""
    history_lines: list[str] = []
    for i, step in enumerate(student_prefix):
        history_lines.append(
            f"[Turn {i}]\n<think>{step.thought}</think>\n"
            f"<action>{step.action}</action>\n"
            f"<observation exit_code={step.exit_code}>{step.observation}</observation>"
        )
    prefix_block = "\n\n".join(history_lines) if history_lines else "(No prior successful turns)"

    return (
        "You are an expert Linux terminal and software engineering teacher recovering "
        "a student agent that just made an execution mistake.\n\n"
        f"### Task Objective\n{task_prompt}\n\n"
        f"### Prior Student Prefix (Turns 0..{len(student_prefix) - 1})\n{prefix_block}\n\n"
        f"### Student's Failing Attempt at Turn {len(student_prefix)}\n"
        f"Action: `{failed_step.action}`\n"
        f"Exit Code: `{failed_step.exit_code}`\n"
        f"Error Output:\n```\n{failed_step.observation}\n```\n\n"
        "Provide the explicit root-cause diagnosis inside `<think>...</think>` and the "
        "exact corrected bash command inside `<action>...</action>` to fix the error and "
        "pass the task verification."
    )
