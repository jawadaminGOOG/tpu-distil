"""Isolated container/workspace sandbox runner for Terminal-Bench, R2E-Gym, and SWE-Gym.

Provides deterministic command execution, timeout isolation, unit-test progress
parsing, and state snapshot/restore (`capture_state_diff`) on official
`Terminal-Bench` tasks so Tunix GRPO can branch short-horizon rollouts directly
from the student's first error state `s_m`.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
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

    def __init__(
        self,
        workdir: str | Path | None = None,
        default_timeout_s: float = 30.0,
        app_dir: Path | None = None,
        test_dir: Path | None = None,
    ) -> None:
        self._tempdir = (
            tempfile.TemporaryDirectory(prefix="tpu_distil_sbx_") if workdir is None else None
        )
        self.root_dir = Path(workdir) if workdir is not None else Path(self._tempdir.name)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.app_dir = app_dir if app_dir is not None else self.root_dir
        self.workdir = self.app_dir
        self.test_dir = test_dir
        self.default_timeout_s = default_timeout_s

    @classmethod
    def from_terminal_bench_task(
        cls,
        task_dir: str | Path,
        default_timeout_s: float = 15.0,
    ) -> ContainerSandbox:
        """Initialize an isolated sandbox from an official Terminal-Bench task directory."""
        tdir = Path(task_dir)
        sbx = cls(default_timeout_s=default_timeout_s)
        app_dir = sbx.root_dir / "app"
        test_dir = sbx.root_dir / "tests"
        app_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)
        sbx.app_dir = app_dir
        sbx.workdir = app_dir
        sbx.test_dir = test_dir

        if (tdir / "tests").exists():
            shutil.copytree(tdir / "tests", test_dir, dirs_exist_ok=True)

        cache_root = Path("/tmp/tpu_distil_image_cache_v2") / tdir.name
        cache_app = cache_root / "app"
        cache_wd_file = cache_root / "workdir_rel.txt"

        def _rewrite_paths(text: str, target_app: Path) -> str:
            t = (
                text.replace(str(target_app), "/app")
                .replace(str(cache_app), "/app")
                .replace("-C / ", "-C /app ")
                .replace("/app/protected", "/protected")
                .replace("/app/opt/", "/opt/")
                .replace("/app/etc/", "/etc/")
                .replace("/app/tmp/", "/tmp/")
                .replace("/protected", "/app/protected")
                .replace("/opt/", "/app/opt/")
                .replace("/etc/", "/app/etc/")
            )
            t = re.sub(r"/tmp/(?!tpu_distil_)", "/app/tmp/", t)
            return t.replace("/app", str(target_app))

        if not (cache_root / ".built").exists():
            shutil.rmtree(cache_root, ignore_errors=True)
            cache_app.mkdir(parents=True, exist_ok=True)
            for item in tdir.iterdir():
                if item.name in (
                    "solution.sh",
                    "tests",
                    "run-tests.sh",
                    "docker-compose.yaml",
                    "Dockerfile",
                    "task.yaml",
                ):
                    continue
                if item.name == "task-deps" and item.is_dir():
                    shutil.copytree(item, cache_app, dirs_exist_ok=True)
                elif item.is_dir():
                    shutil.copytree(item, cache_app / item.name, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, cache_app / item.name)

            (cache_app / "tmp").mkdir(parents=True, exist_ok=True)
            (cache_app / "etc").mkdir(parents=True, exist_ok=True)
            (cache_app / "opt").mkdir(parents=True, exist_ok=True)

            df_path = tdir / "Dockerfile"
            df_txt = df_path.read_text(errors="ignore") if df_path.exists() else ""
            joined = re.sub(r"\\\s*\n", " ", df_txt)
            cur_wd = cache_app
            sbx.app_dir = cache_app
            env_build = sbx._build_env()

            for raw_line in joined.splitlines():
                s = raw_line.strip()
                if not s or s.startswith("#"):
                    continue
                upper_s = s.upper()
                if upper_s.startswith("WORKDIR "):
                    wd_raw = s[8:].strip()
                    if wd_raw in ("/app", "/app/"):
                        cur_wd = cache_app
                    elif wd_raw.startswith("/app/"):
                        cur_wd = cache_app / wd_raw[5:].rstrip("/")
                    elif wd_raw.startswith("/"):
                        cur_wd = cache_app / wd_raw.lstrip("/").rstrip("/")
                    else:
                        cur_wd = (cur_wd / wd_raw).resolve()
                    cur_wd.mkdir(parents=True, exist_ok=True)
                elif upper_s.startswith(("COPY ", "ADD ")):
                    parts = s.split()
                    if len(parts) >= 3:
                        srcs, dst = parts[1:-1], parts[-1]
                        if dst in (".", "./"):
                            dst_p = cur_wd
                        elif dst in ("/app", "/app/"):
                            dst_p = cache_app
                        elif dst.startswith("/app/"):
                            dst_p = cache_app / dst[5:]
                        elif dst.startswith("/"):
                            dst_p = cache_app / dst.lstrip("/")
                        else:
                            dst_p = cur_wd / dst
                        for src_tok in srcs:
                            if src_tok.startswith("--"):
                                continue
                            src_p = tdir / src_tok.rstrip("/.")
                            if src_p.exists():
                                if src_p.is_dir():
                                    shutil.copytree(src_p, dst_p, dirs_exist_ok=True)
                                elif dst.endswith("/") or dst in (".", "./", "/app", "/app/"):
                                    dst_p.mkdir(parents=True, exist_ok=True)
                                    shutil.copy2(src_p, dst_p / src_p.name)
                                else:
                                    dst_p.parent.mkdir(parents=True, exist_ok=True)
                                    shutil.copy2(src_p, dst_p)
                elif upper_s.startswith("RUN "):
                    cmd = s[4:].strip()
                    if any(
                        k in cmd
                        for k in (
                            "apt-get",
                            "apt ",
                            "apk ",
                            "yum ",
                            "dnf ",
                            "pip install",
                            "pip3 install",
                            "conda",
                            "cargo",
                            "rustup",
                            "npm ",
                            "sudo",
                            "nohup",
                        )
                    ):
                        continue
                    for sf in list(cache_app.glob("*.sh")) + list(cache_app.glob("*.py")):
                        if sf.is_file():
                            try:
                                sf.write_text(_rewrite_paths(sf.read_text(errors="ignore"), cache_app))
                            except Exception:
                                pass
                    run_cmd = _rewrite_paths(cmd, cache_app)
                    if run_cmd.endswith("&& make") or run_cmd == "make":
                        run_cmd = run_cmd + " -j16"
                    sbx._exec_safe(
                        ["/bin/bash", "-c", run_cmd],
                        cwd=cur_wd,
                        env=env_build,
                        timeout_s=90.0,
                    )

            for sub in ("john/src", "john/doc", "john/.git", "frotz/src", "frotz/.git"):
                shutil.rmtree(cache_app / sub, ignore_errors=True)
            shutil.rmtree(cache_app / "tests", ignore_errors=True)
            for forbidden in (
                "solution.sh",
                "solution.yaml",
                "run-tests.sh",
                "docker-compose.yaml",
                "Dockerfile",
                "task.yaml",
            ):
                (cache_app / forbidden).unlink(missing_ok=True)

            try:
                rel_wd = str(cur_wd.relative_to(cache_app))
            except ValueError:
                rel_wd = "."
            cache_wd_file.write_text(rel_wd, encoding="utf-8")
            (cache_root / ".built").write_text("ok", encoding="utf-8")

        for item in cache_app.iterdir():
            dst_item = app_dir / item.name
            if item.name in ("john", "frotz") and item.is_dir():
                if not dst_item.exists():
                    dst_item.symlink_to(item)
            elif item.is_dir():
                shutil.copytree(item, dst_item, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dst_item)
        sbx.app_dir = app_dir
        rel_wd = cache_wd_file.read_text(encoding="utf-8").strip() if cache_wd_file.exists() else "."
        if rel_wd.startswith(("john", "frotz")):
            rel_wd = "."
        sbx.workdir = app_dir if rel_wd in ("", ".") else (app_dir / rel_wd)
        sbx.workdir.mkdir(parents=True, exist_ok=True)

        for sf in list(app_dir.glob("*.sh")) + list(app_dir.glob("*.py")):
            if sf.is_file():
                try:
                    sf.write_text(_rewrite_paths(sf.read_text(errors="ignore"), app_dir))
                except Exception:
                    pass
        test_out_py = test_dir / "test_outputs.py"
        if test_out_py.is_file():
            try:
                test_out_py.write_text(_rewrite_paths(test_out_py.read_text(errors="ignore"), app_dir))
            except Exception:
                pass

        env = sbx._build_env()
        # Initialize git tracking if not already a git repo so capture_state_diff works
        if not any(sbx.workdir.rglob(".git")):
            sbx._exec_safe(
                [
                    "/bin/bash",
                    "-c",
                    "git init -q && git config user.email 'sbx@tpu.local' && git config user.name 'sbx' && git add -A && git commit -qm 'init' --allow-empty",
                ],
                cwd=sbx.workdir,
                env=env,
                timeout_s=5.0,
            )
        return sbx

    def _exec_safe(
        self,
        cmd_args: list[str],
        cwd: Path,
        env: dict[str, str],
        timeout_s: float,
    ) -> tuple[int, str, str]:
        """Run a command in a new process group with file-backed output to prevent pipe hangs."""
        import signal

        with tempfile.TemporaryFile() as out_f, tempfile.TemporaryFile() as err_f:
            proc = subprocess.Popen(
                cmd_args,
                cwd=str(cwd),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=out_f,
                stderr=err_f,
                start_new_session=True,
            )
            timed_out = False
            try:
                proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
            finally:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=1.0)
                except Exception:
                    pass
            out_f.seek(0)
            err_f.seek(0)
            stdout_str = out_f.read(256_000).decode("utf-8", errors="replace")
            stderr_str = err_f.read(256_000).decode("utf-8", errors="replace")
            rc = 124 if timed_out else (proc.returncode if proc.returncode is not None else 1)
            return rc, stdout_str, stderr_str

    def _build_env(self) -> dict[str, str]:
        shim_dir = Path("/tmp/tpu_distil_rootless_shims")
        if not (shim_dir / "zip").exists():
            shim_dir.mkdir(parents=True, exist_ok=True)
            for tool in ("apt-get", "apt"):
                p = shim_dir / tool
                p.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
                p.chmod(0o755)
            sudo_p = shim_dir / "sudo"
            sudo_p.write_text("#!/bin/sh\nexec \"$@\"\n", encoding="utf-8")
            sudo_p.chmod(0o755)
            gpg_p = shim_dir / "gpg"
            gpg_p.write_text(
                "#!/bin/sh\nexec /usr/bin/gpg --batch --yes --pinentry-mode loopback \"$@\"\n",
                encoding="utf-8",
            )
            gpg_p.chmod(0o755)
            zip_p = shim_dir / "zip"
            zip_p.write_text(
                "#!/usr/bin/env python3\n"
                "import os, sys, zipfile\n"
                "args = [a for a in sys.argv[1:] if not a.startswith('-')]\n"
                "if len(args) >= 2:\n"
                "    zpath, srcs = args[0], args[1:]\n"
                "    with zipfile.ZipFile(zpath, 'w', compression=zipfile.ZIP_STORED) as zf:\n"
                "        for s in srcs:\n"
                "            if os.path.isdir(s):\n"
                "                for r, _, fs in os.walk(s):\n"
                "                    for f in fs:\n"
                "                        fp = os.path.join(r, f)\n"
                "                        zf.write(fp, arcname=os.path.basename(fp))\n"
                "            elif os.path.exists(s):\n"
                "                zf.write(s, arcname=os.path.basename(s))\n",
                encoding="utf-8",
            )
            zip_p.chmod(0o755)
            unzip_p = shim_dir / "unzip"
            unzip_p.write_text(
                "#!/usr/bin/env python3\n"
                "import sys, zipfile\n"
                "args = [a for a in sys.argv[1:] if not a.startswith('-')]\n"
                "if args:\n"
                "    with zipfile.ZipFile(args[0], 'r') as zf:\n"
                "        zf.extractall('.')\n",
                encoding="utf-8",
            )
            unzip_p.chmod(0o755)
        env = dict(os.environ)
        env["PIP_BREAK_SYSTEM_PACKAGES"] = "1"
        env["PATH"] = f"{shim_dir}:{os.path.expanduser('~/.local/bin')}:{env.get('PATH', '')}"
        py_paths: list[str] = []
        if not (self.app_dir / "code.py").exists():
            py_paths.append(str(self.app_dir))
        if self.test_dir is not None:
            env["TEST_DIR"] = str(self.test_dir)
            py_paths.append(str(self.test_dir))
        if env.get("PYTHONPATH"):
            py_paths.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = ":".join(py_paths)
        return env

    def evaluate_tests(self, timeout_s: float = 5.0) -> tuple[int, int, str]:
        """Run the hidden Terminal-Bench `tests/test_outputs.py` suite if present."""
        if self.test_dir is None or not (self.test_dir / "test_outputs.py").exists():
            return 0, 0, ""
        rc, out_s, err_s = self._exec_safe(
            [
                sys.executable,
                "-P",
                "-m",
                "pytest",
                "-vv",
                "--tb=short",
                str(self.test_dir / "test_outputs.py"),
            ],
            cwd=self.workdir,
            env=self._build_env(),
            timeout_s=timeout_s,
        )
        out = f"{out_s}\n{err_s}"
        passed, total = parse_pytest_counts(out)
        if rc == 0 and total == 0:
            passed, total = 1, 1
        elif rc != 0 and total == 0:
            passed, total = 0, 1
        return passed, total, out.replace(str(self.app_dir), "/app")

    def run_step(self, action_cmd: str, timeout_s: float | None = None) -> SandboxResult:
        """Execute `action_cmd` inside `self.workdir` and parse unit-test progress."""
        timeout = timeout_s if timeout_s is not None else self.default_timeout_s
        if not action_cmd or not action_cmd.strip():
            return SandboxResult(stdout="", stderr="Empty action command", exit_code=2)
        try:
            (self.app_dir / "john" / "run" / "john.pot").unlink(missing_ok=True)
        except Exception:
            pass
        t_cmd = (
            action_cmd.replace(str(self.app_dir), "/app")
            .replace("-C / ", "-C /app ")
            .replace("/app/protected", "/protected")
            .replace("/app/opt/", "/opt/")
            .replace("/app/etc/", "/etc/")
            .replace("/app/tmp/", "/tmp/")
            .replace("/protected", "/app/protected")
            .replace("/opt/", "/app/opt/")
            .replace("/etc/", "/app/etc/")
        )
        t_cmd = re.sub(r"/tmp/(?!tpu_distil_)", "/app/tmp/", t_cmd)
        cmd_translated = t_cmd.replace("/app", str(self.app_dir))
        rc, stdout_raw, stderr_raw = self._exec_safe(
            ["/bin/bash", "-c", cmd_translated],
            cwd=self.workdir,
            env=self._build_env(),
            timeout_s=timeout,
        )
        stdout_clean = stdout_raw.replace(str(self.app_dir), "/app")
        stderr_clean = stderr_raw.replace(str(self.app_dir), "/app")
        if rc in (124, 137):
            return SandboxResult(
                stdout=stdout_clean,
                stderr=f"{stderr_clean}\n[TimeoutExpired after {timeout}s]".strip(),
                exit_code=124,
            )
        combined = f"{stdout_clean}\n{stderr_clean}"
        passed, total = parse_pytest_counts(combined)
        return SandboxResult(
            stdout=stdout_clean,
            stderr=stderr_clean,
            exit_code=rc,
            unit_tests_passed=passed,
            unit_tests_total=total,
        )

    def capture_state_diff(self) -> str:
        """Capture a git diff snapshot of the current workspace at error state `s_m`."""
        proc = subprocess.run(
            ["/bin/bash", "-c", "git add -N . >/dev/null 2>&1 && git diff HEAD || true"],
            cwd=str(self.workdir),
            env=self._build_env(),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return proc.stdout.replace(str(self.app_dir), "/app")

    def close(self) -> None:
        if self._tempdir is not None:
            self._tempdir.cleanup()
