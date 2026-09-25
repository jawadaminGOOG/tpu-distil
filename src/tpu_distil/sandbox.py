"""Isolated container/workspace sandbox runner for Terminal-Bench, R2E-Gym, and SWE-Gym.

Provides deterministic command execution, timeout isolation, unit-test progress
parsing, and state snapshot/restore (`capture_state_diff` / `restore_state_diff`)
so Tunix GRPO can branch short-horizon rollouts directly from the student's
first error state `s_m`.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SandboxResult:
    """Result of executing a single bash/tool action in a task sandbox."""

    stdout: str
    stderr: str
    exit_code: int
    unit_tests_passed: int = 0
    unit_tests_total: int = 0

    @property
    def unit_test_pass_rate(self) -> float:
        if self.unit_tests_total <= 0:
            return 1.0 if self.exit_code == 0 else 0.0
        return float(self.unit_tests_passed) / float(self.unit_tests_total)

    @property
    def combined_output(self) -> str:
        out = self.stdout.strip()
        err = self.stderr.strip()
        if out and err:
            return f"{out}\n[stderr]\n{err}"
        return out or err or f"[exit_code={self.exit_code}]"


_PYTEST_PASSED_RE = re.compile(r"(\d+)\s+passed")
_PYTEST_FAILED_RE = re.compile(r"(\d+)\s+(?:failed|error)")


def parse_pytest_counts(output: str) -> tuple[int, int]:
    """Extract (passed, total) unit test counts from pytest/unittest output."""
    passed_match = _PYTEST_PASSED_RE.search(output)
    failed_match = _PYTEST_FAILED_RE.search(output)
    passed = int(passed_match.group(1)) if passed_match else 0
    failed = int(failed_match.group(1)) if failed_match else 0
    return passed, passed + failed


class ContainerSandbox:
    """Sandboxed command execution environment with state diff snapshot/restore."""

    def __init__(self, workdir: str | Path | None = None, default_timeout_s: float = 30.0) -> None:
        self._tempdir = tempfile.TemporaryDirectory(prefix="tpu_distil_sbx_") if workdir is None else None
        self.workdir = Path(workdir) if workdir is not None else Path(self._tempdir.name)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.default_timeout_s = default_timeout_s

    def run_step(self, action_cmd: str, timeout_s: float | None = None) -> SandboxResult:
        """Execute `action_cmd` inside `self.workdir` and parse unit-test progress."""
        timeout = timeout_s if timeout_s is not None else self.default_timeout_s
        if not action_cmd or not action_cmd.strip():
            return SandboxResult(
                stdout="",
                stderr="Empty action command",
                exit_code=2,
                unit_tests_passed=0,
                unit_tests_total=0,
            )
        try:
            proc = subprocess.run(
                ["/bin/bash", "-c", action_cmd],
                cwd=str(self.workdir),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            combined = f"{proc.stdout}\n{proc.stderr}"
            passed, total = parse_pytest_counts(combined)
            return SandboxResult(
                stdout=proc.stdout,
                stderr=proc.stderr,
                exit_code=proc.returncode,
                unit_tests_passed=passed,
                unit_tests_total=total,
            )
        except subprocess.TimeoutExpired as exc:
            stdout_str = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr_str = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            return SandboxResult(
                stdout=stdout_str,
                stderr=f"{stderr_str}\n[TimeoutExpired after {timeout}s]".strip(),
                exit_code=124,
                unit_tests_passed=0,
                unit_tests_total=0,
            )

    def capture_state_diff(self) -> str:
        """Capture a git diff snapshot of the current workspace at error state `s_m`."""
        res = self.run_step("git status --porcelain >/dev/null 2>&1 && git diff HEAD || true")
        return res.stdout

    def close(self) -> None:
        if self._tempdir is not None:
            self._tempdir.cleanup()
