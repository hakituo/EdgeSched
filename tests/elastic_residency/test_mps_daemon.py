"""MPS daemon 就绪判定与查询语义的单元测试。"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from experiments.elastic_residency.sm.mps_commands import MpsRuntimeError
from tests.elastic_residency.mps_daemon_helpers import (
    BINARY_NAME,
    SERVER_PID,
    LaunchRecorder,
    ScriptedRunner,
    build_handle,
    build_runtime,
)


@pytest.fixture(autouse=True)
def _allow_fake_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    """让显式 binary_path 在 Windows 上也能通过可执行性检查。"""

    monkeypatch.setattr(os, "access", lambda *_args: True)


def test_controller_ready_does_not_require_server(tmp_path: Path) -> None:
    runner = ScriptedRunner()
    handle = build_handle(tmp_path, runner)
    evidence = handle.wait_until_controller_ready(timeout_s=0.2)
    assert evidence.ready is True
    assert evidence.pid == 4242
    # server 列表为空是合法就绪状态，不能据此判定启动失败。
    assert handle.query_servers() == ()


def test_missing_pid_file_reports_controller_not_ready(tmp_path: Path) -> None:
    handle = build_handle(tmp_path, ScriptedRunner(), with_pid_file=False)
    evidence = handle.controller_ready_evidence()
    assert evidence.ready is False
    assert evidence.reason == "controller_pid_file_missing"
    with pytest.raises(MpsRuntimeError, match="nvidia-cuda-mps-control.pid"):
        handle.wait_until_controller_ready(timeout_s=0.15)


def test_start_failure_never_blames_missing_server(tmp_path: Path) -> None:
    """启动失败必须来自 controller 未就绪，不能来自 server 未出现。"""

    runtime = build_runtime(
        tmp_path,
        ScriptedRunner(),
        launcher=LaunchRecorder(write_pid=False),
        start_timeout_s=0.15,
    )
    with pytest.raises(MpsRuntimeError) as excinfo:
        runtime.start(label="pct50")
    message = str(excinfo.value)
    assert "nvidia-cuda-mps-control.pid" in message
    assert "get_server_list" not in message
    assert not any(
        path.is_dir() and path.name.startswith("edgesched_mps_")
        for path in tmp_path.iterdir()
    )


def test_control_commands_go_through_stdin_only(tmp_path: Path) -> None:
    runner = ScriptedRunner()
    handle = build_handle(tmp_path, runner)
    handle.query_servers()
    handle.cleanup()
    assert runner.calls, "必须发起控制命令"
    for argv, _stdin in runner.calls:
        assert len(argv) == 1, "argv 只能是可执行文件路径"
        assert argv[0].endswith(BINARY_NAME)
    assert "get_server_list" in runner.commands
    assert "quit" in runner.commands


def test_server_and_client_appear_only_after_client_connects(tmp_path: Path) -> None:
    runner = ScriptedRunner()
    handle = build_handle(tmp_path, runner)
    assert handle.query_servers() == ()
    assert handle.query_clients() == ()
    runner.client_connected = True
    assert handle.query_servers() == (f"server {SERVER_PID}",)
    assert handle.query_clients() == (999,)


def test_client_environment_injects_mps_variables(tmp_path: Path) -> None:
    handle = build_handle(tmp_path, ScriptedRunner())
    from experiments.elastic_residency.sm.mps_commands import CLIENT_ENV_KEYS

    env = handle.client_environment(25)
    for key in CLIENT_ENV_KEYS:
        assert key in env
    assert env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] == "25"
    with pytest.raises(MpsRuntimeError):
        handle.client_environment(0)


def test_start_server_uses_stdin_and_same_uid(tmp_path: Path) -> None:
    runner = ScriptedRunner()
    handle = build_handle(tmp_path, runner)
    if not hasattr(os, "getuid"):
        with pytest.raises(MpsRuntimeError, match="getuid"):
            handle.start_server()
        return
    handle.start_server()
    assert runner.commands == [f"start_server -uid {os.getuid()}"]


def test_health_check_records_query_result(tmp_path: Path) -> None:
    runner = ScriptedRunner()
    handle = build_handle(tmp_path, runner)
    runner.client_connected = True
    health = handle.health_check()
    assert health.server_count == 1
    assert health.client_count == 1
    assert health.query is not None
    assert health.query.stdin_command is not None
