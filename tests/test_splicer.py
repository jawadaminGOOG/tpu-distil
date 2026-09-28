"""Unit tests for `tpu_distil.splicer`."""

from __future__ import annotations

from tpu_distil.splicer import (
    build_teacher_recovery_prompt,
    find_first_error_step,
    splice_trajectory,
)
from tpu_distil.trajectory import Step, Trajectory


def test_find_first_error_step() -> None:
    steps = [
        Step(thought="List files", action="ls -la", observation="main.py", exit_code=0),
        Step(
            thought="Run broken pytest",
            action="pytest",
            observation="1 failed",
            exit_code=1,
        ),
        Step(thought="Later step", action="echo hi", observation="hi", exit_code=0),
    ]
    assert find_first_error_step(steps) == 1
    assert find_first_error_step(steps[:1]) is None


def test_splice_trajectory_preserves_student_prefix_and_marks_teacher() -> None:
    student_traj = Trajectory(
        task_id="r2e-101",
        prompt="Fix division by zero in calc.py",
        steps=[
            Step(thought="Inspect calc.py", action="cat calc.py", observation="def div(a,b): return a/b", exit_code=0),
            Step(thought="Bad patch", action="sed -i 's/b/0/' calc.py && pytest", observation="ZeroDivisionError", exit_code=1),
        ],
        terminal_reward=0.0,
    )
    teacher_suffix = [
        Step(
            thought="Restore b and guard b == 0",
            action="python3 -c \"open('calc.py','w').write('def div(a,b): return 0 if b==0 else a/b\\n')\" && pytest",
            observation="2 passed in 0.03s",
            exit_code=0,
            unit_test_pass_rate=1.0,
        )
    ]

    spliced = splice_trajectory(
        student_traj=student_traj,
        teacher_suffix=teacher_suffix,
        terminal_reward=1.0,
        teacher_model="DeepSeek-V4.1-Flash",
        state_s_m_diff="diff --git a/calc.py b/calc.py",
    )

    assert spliced.splice_step == 1
    assert len(spliced.steps) == 2
    assert spliced.steps[0].action == "cat calc.py"
    assert spliced.steps[0].teacher_generated is False
    assert spliced.steps[1].teacher_generated is True
    assert spliced.terminal_reward == 1.0
    assert spliced.teacher_model == "DeepSeek-V4.1-Flash"


def test_build_teacher_recovery_prompt_includes_failing_turn() -> None:
    prefix = [Step(thought="Check dir", action="pwd", observation="/workspace", exit_code=0)]
    failed = Step(thought="Run test", action="pytest test_app.py", observation="AssertionError: 4 != 5", exit_code=1)
    prompt = build_teacher_recovery_prompt("Fix test_app.py", prefix, failed)
    assert "AssertionError: 4 != 5" in prompt
    assert "pytest test_app.py" in prompt
    assert "<think>" in prompt


def test_splice_trajectory_includes_failed_step_when_requested() -> None:
    student_traj = Trajectory(
        task_id="r2e-102",
        prompt="Fix division by zero in calc.py",
        steps=[
            Step(thought="Inspect calc.py", action="cat calc.py", observation="def div(a,b): return a/b", exit_code=0),
            Step(thought="Bad patch", action="sed -i 's/b/0/' calc.py && pytest", observation="ZeroDivisionError", exit_code=1),
        ],
        terminal_reward=0.0,
    )
    teacher_suffix = [
        Step(
            thought="Restore b and guard b == 0",
            action="python3 -c \"open('calc.py','w').write('def div(a,b): return 0 if b==0 else a/b\\n')\" && pytest",
            observation="2 passed in 0.03s",
            exit_code=0,
            unit_test_pass_rate=1.0,
        )
    ]

    spliced = splice_trajectory(
        student_traj=student_traj,
        teacher_suffix=teacher_suffix,
        terminal_reward=1.0,
        teacher_model="DeepSeek-V4.1-Flash",
        state_s_m_diff="diff --git a/calc.py b/calc.py",
        include_failed_step_in_prefix=True,
    )

    assert spliced.splice_step == 2
    assert len(spliced.steps) == 3
    assert spliced.steps[0].teacher_generated is False
    assert spliced.steps[1].teacher_generated is False
    assert spliced.steps[1].exit_code == 1
    assert spliced.steps[2].teacher_generated is True
    assert spliced.terminal_reward == 1.0

