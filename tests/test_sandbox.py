"""Unit tests for `tpu_distil.sandbox`."""

from __future__ import annotations

from tpu_distil.sandbox import ContainerSandbox, parse_pytest_counts


def test_container_sandbox_exec_and_timeout() -> None:
    sbx = ContainerSandbox(default_timeout_s=2.0)
    try:
        res_ok = sbx.run_step("echo '=== 3 passed, 1 failed in 0.12s ==='")
        assert res_ok.exit_code == 0
        assert res_ok.unit_tests_passed == 3
        assert res_ok.unit_tests_total == 4
        assert abs(res_ok.unit_test_pass_rate - 0.75) < 1e-6

        res_err = sbx.run_step("exit 42")
        assert res_err.exit_code == 42
        assert res_err.unit_test_pass_rate == 0.0

        res_timeout = sbx.run_step("sleep 5", timeout_s=0.2)
        assert res_timeout.exit_code == 124
        assert "TimeoutExpired" in res_timeout.stderr
    finally:
        sbx.close()


def test_parse_pytest_counts() -> None:
    passed, total = parse_pytest_counts("===== 12 passed, 3 failed in 1.04s =====")
    assert passed == 12
    assert total == 15
