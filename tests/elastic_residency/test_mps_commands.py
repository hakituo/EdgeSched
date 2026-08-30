"""MPS 控制命令通道的单元测试：命令必须走 stdin，argv 只含程序路径。"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import pytest

from experiments.elastic_residency.sm.mps_commands import (
    CLIENT_ENV_KEYS,
    MpsRuntimeError,
    as_text,
    parse_pid_lines,
    parse_server_lines,
    read_controller_pid,
    unmanaged_environment,
    validate_percentage,
)
from experiments.elastic_residency.sm.mps_protocol import (
    CommandResult,
    CommandSpec,
    control_command_stdin,
    subprocess_runner,
)


def test_control_command_is_encoded_as_stdin_text() -> None:
    assert control_command_stdin("get_server_list") == "get_server_list\n"
    assert control_command_stdin("  quit  ") == "quit\n"


def test_command_result_records_argv_and_stdin_separately() -> None:
    result = CommandResult(
        argv=("/usr/bin/nvidia-cuda-mps-control",),
        stdin_command="get_server_list\n",
        returncode=0,
        stdout="server 1\n",
        stderr="",
    )
    payload = result.to_dict()
    assert payload["argv"] == ["/usr/bin/nvidia-cuda-mps-control"]
    assert payload["stdin_command"] == "get_server_list\n"
    assert payload["returncode"] == 0


def test_command_spec_carries_stdin_and_timeout() -> None:
    spec = CommandSpec(argv=("/bin/mps",), stdin="quit\n", timeout_s=3.5)
    assert spec.to_dict() == {
        "argv": ["/bin/mps"],
        "stdin": "quit\n",
        "timeout_s": 3.5,
    }


def test_subprocess_runner_sends_command_via_stdin() -> None:
    """真实子进程验证：控制命令出现在 stdin，argv 只有解释器与脚本。"""

    spec = CommandSpec(
        argv=(sys.executable, "-c", "import sys; print(sys.stdin.read().strip())"),
        stdin=control_command_stdin("get_server_list"),
        timeout_s=30.0,
    )
    result = subprocess_runner(spec, dict(os.environ))
    assert result.returncode == 0
    assert result.stdout.strip() == "get_server_list"
    assert result.stdin_command == "get_server_list\n"
    assert result.argv[0] == sys.executable


def test_subprocess_runner_reports_timeout_and_os_errors() -> None:
    timed_out = subprocess_runner(
        CommandSpec(
            argv=(sys.executable, "-c", "import time; time.sleep(10)"),
            timeout_s=0.5,
        ),
        dict(os.environ),
    )
    assert timed_out.timed_out is True
    assert timed_out.error == "timeout"
    failed = subprocess_runner(
        CommandSpec(argv=("no-such-binary-xyz",), timeout_s=1.0), {"PATH": ""}
    )
    assert failed.returncode is None
    assert failed.error.startswith("os_error:")


def test_server_and_client_parsers() -> None:
    assert parse_server_lines("server 4242\n") == ("server 4242",)
    assert parse_server_lines("No servers are currently running.\n") == ()
    assert parse_pid_lines("  PID 1234\n  PID 5678\n\n") == (1234, 5678)
    assert parse_pid_lines("No clients\n") == ()


def test_read_controller_pid_reports_missing_or_invalid(tmp_path: Path) -> None:
    missing, reason = read_controller_pid(str(tmp_path / "nope.pid"))
    assert missing is None
    assert reason == "controller_pid_file_missing"

    invalid = tmp_path / "nvidia-cuda-mps-control.pid"
    invalid.write_text("not-a-pid\n", encoding="utf-8")
    assert read_controller_pid(str(invalid)) == (
        None,
        "controller_pid_file_not_numeric",
    )

    valid = tmp_path / "ok.pid"
    valid.write_text("4242\n", encoding="utf-8")
    assert read_controller_pid(str(valid)) == (4242, None)


def test_validate_percentage_and_unmanaged_environment() -> None:
    assert validate_percentage(75) == 75
    for invalid in (0, -1, 101):
        with pytest.raises(MpsRuntimeError):
            validate_percentage(invalid)

    env: Mapping[str, Any] = {key: "x" for key in CLIENT_ENV_KEYS}
    env = dict(env, PATH="/usr/bin")
    cleaned = unmanaged_environment(env)
    for key in CLIENT_ENV_KEYS:
        assert key not in cleaned
    assert cleaned["PATH"] == "/usr/bin"


def test_as_text_handles_bytes_and_none() -> None:
    assert as_text(None) == ""
    assert as_text(b"abc") == "abc"
    assert as_text(5) == "5"
