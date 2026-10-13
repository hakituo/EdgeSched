"""MPS daemon 清理与证据记录的单元测试。"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Mapping

import pytest

from experiments.elastic_residency.sm.mps_commands import (
    MpsRuntimeError,
)
from experiments.elastic_residency.sm.mps_protocol import (
    CommandResult,
    CommandSpec,
)
from tests.elastic_residency.mps_daemon_helpers import (
    CONTROLLER_PID,
    SERVER_PID,
    LaunchRecorder,
    PostQuitServerRunner,
    ProcessRegistry,
    QuitKillsController,
    ScriptedRunner,
    build_handle,
    build_runtime,
    observe_server,
)


@pytest.fixture(autouse=True)
def _allow_fake_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "access", lambda *_args: True)


def test_cleanup_removes_pid_file_server_and_private_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # controller 退出后 server 进程已消失。
    ProcessRegistry(
        alive=set(),
        identities={
            CONTROLLER_PID: {
                "pid": CONTROLLER_PID,
                "starttime": 500,
                "executable": "/ctl",
            },
            SERVER_PID: {"pid": SERVER_PID, "starttime": 100, "executable": "/srv"},
        },
    ).install(monkeypatch)
    runner = ScriptedRunner()
    handle = build_handle(tmp_path, runner)
    runner.client_connected = True
    assert handle.query_servers() == (f"server {SERVER_PID}",)

    assert handle.cleanup(archive_dir=tmp_path / "archive") is True
    # quit 之后 server 与 controller PID 文件都应消失。
    assert handle.controller_pid_file.exists() is False
    assert handle.root.exists() is False
    assert handle.residual_artifacts() == []
    assert handle.post_quit_evidence["server_count"] == 0
    assert handle.post_quit_evidence["reached_quiescent"] is True


def test_cleanup_failure_is_recorded_not_swallowed(tmp_path: Path) -> None:
    class FailingQuit(ScriptedRunner):
        def __call__(self, spec: CommandSpec, env: Mapping[str, str]) -> CommandResult:
            if (spec.stdin or "").strip() == "quit":
                return CommandResult(tuple(spec.argv), spec.stdin, 255, "", "boom")
            return super().__call__(spec, env)

    handle = build_handle(tmp_path, FailingQuit())
    assert handle.cleanup() is False
    assert handle.cleanup_error == "quit_returncode_255"
    assert handle.snapshot()["cleanup_succeeded"] is False


def test_unconfirmed_shutdown_retains_pid_file_and_private_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """controller 未确认退出时，私有目录与 PID 必须保留供人工处理。"""

    class KeepPidFile(ScriptedRunner):
        def __call__(self, spec: CommandSpec, env: Mapping[str, str]) -> CommandResult:
            return CommandResult(tuple(spec.argv), spec.stdin, 0, "", "")

    # controller 始终“存活”：quit 后仍未确认退出，必须 fail-closed 保留现场。
    ProcessRegistry(alive={CONTROLLER_PID}).install(monkeypatch)
    handle = build_handle(tmp_path, KeepPidFile())
    assert handle.cleanup() is False
    assert handle.controller_pid_file.exists() is True
    assert handle.root.exists() is True
    assert handle.residual_artifacts() == ["controller_pid_file", "private_root"]
    assert handle.retained_artifacts["private_root"] == str(handle.root)
    assert handle.retained_artifacts["controller_pid_file"] == (
        str(handle.controller_pid_file)
    )


def test_quit_timeout_is_fail_closed(tmp_path: Path) -> None:
    """quit 超时：returncode 缺失且 timed_out，必须判失败。"""

    class TimeoutQuit(ScriptedRunner):
        def __call__(self, spec: CommandSpec, env: Mapping[str, str]) -> CommandResult:
            if (spec.stdin or "").strip() == "quit":
                return CommandResult(
                    tuple(spec.argv), spec.stdin, None, "", "", timed_out=True
                )
            return super().__call__(spec, env)

    handle = build_handle(tmp_path, TimeoutQuit())
    assert handle.cleanup() is False
    assert handle.cleanup_error == "quit_timeout"
    assert handle.root.exists(), "未确认关闭时不得删除私有目录"


def test_quit_os_error_is_fail_closed(tmp_path: Path) -> None:
    """OS 错误：即使 returncode 为 None 也不能当成成功。"""

    class OSErrorQuit(ScriptedRunner):
        def __call__(self, spec: CommandSpec, env: Mapping[str, str]) -> CommandResult:
            if (spec.stdin or "").strip() == "quit":
                return CommandResult(
                    tuple(spec.argv),
                    spec.stdin,
                    None,
                    "",
                    "",
                    error="os_error:FileNotFoundError",
                )
            return super().__call__(spec, env)

    handle = build_handle(tmp_path, OSErrorQuit())
    assert handle.cleanup() is False
    assert handle.cleanup_error == "quit_error:os_error:FileNotFoundError"


def test_quit_returncode_none_is_fail_closed(tmp_path: Path) -> None:
    """returncode 为 None（既没超时也没 error）同样不能接受。"""

    class NoReturnCodeQuit(ScriptedRunner):
        def __call__(self, spec: CommandSpec, env: Mapping[str, str]) -> CommandResult:
            if (spec.stdin or "").strip() == "quit":
                return CommandResult(tuple(spec.argv), spec.stdin, None, "", "")
            return super().__call__(spec, env)

    handle = build_handle(tmp_path, NoReturnCodeQuit())
    assert handle.cleanup() is False
    assert handle.cleanup_error == "quit_returncode_unknown"


def test_post_quit_server_query_timeout_is_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """controller 仍在时的 post-quit 查询超时：不能折叠成空列表。"""

    ProcessRegistry(
        alive={CONTROLLER_PID},
        identities={
            CONTROLLER_PID: {
                "pid": CONTROLLER_PID,
                "starttime": 500,
                "executable": "/ctl",
            }
        },
    ).install(monkeypatch)

    class TimeoutAfterQuit(ScriptedRunner):
        def __init__(self) -> None:
            super().__init__()
            self.quit_issued = False

        def __call__(self, spec: CommandSpec, env: Mapping[str, str]) -> CommandResult:
            command = (spec.stdin or "").strip()
            if command == "quit":
                self.quit_issued = True
                return CommandResult(tuple(spec.argv), spec.stdin, 0, "", "")
            if command == "get_server_list" and self.quit_issued:
                self.calls.append((tuple(spec.argv), spec.stdin))
                return CommandResult(
                    tuple(spec.argv), spec.stdin, None, "", "", timed_out=True
                )
            return super().__call__(spec, env)

    runner = TimeoutAfterQuit()
    handle = build_handle(tmp_path, runner)
    observe_server(handle, runner)
    assert handle.cleanup() is False
    assert "controller_gone_timeout" in (handle.cleanup_error or "")
    evidence = handle.post_quit_evidence
    assert evidence["verification_mode"] == "control_protocol"
    assert evidence["server_query_error"] == "get_server_list_timeout"
    assert evidence["reached_quiescent"] is False
    assert handle.root.exists()


def test_post_quit_server_query_nonzero_is_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """controller 仍在时的 post-quit 查询非零返回码：同样必须判失败。"""

    ProcessRegistry(
        alive={CONTROLLER_PID},
        identities={
            CONTROLLER_PID: {
                "pid": CONTROLLER_PID,
                "starttime": 500,
                "executable": "/ctl",
            }
        },
    ).install(monkeypatch)

    class NonZeroAfterQuit(ScriptedRunner):
        def __init__(self) -> None:
            super().__init__()
            self.quit_issued = False

        def __call__(self, spec: CommandSpec, env: Mapping[str, str]) -> CommandResult:
            command = (spec.stdin or "").strip()
            if command == "quit":
                self.quit_issued = True
                return CommandResult(tuple(spec.argv), spec.stdin, 0, "", "")
            if command == "get_server_list" and self.quit_issued:
                self.calls.append((tuple(spec.argv), spec.stdin))
                return CommandResult(
                    tuple(spec.argv), spec.stdin, 805, "", "MPS server not ready"
                )
            return super().__call__(spec, env)

    runner = NonZeroAfterQuit()
    handle = build_handle(tmp_path, runner)
    observe_server(handle, runner)
    assert handle.cleanup() is False
    evidence = handle.post_quit_evidence
    assert evidence["server_query_error"] == "get_server_list_returncode_805"
    assert evidence["reached_quiescent"] is False
    assert handle.root.exists()



def test_context_manager_always_cleans_up(
    short_tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # LaunchRecorder 写出的 controller PID 是 4321；写出时标记存活，
    # quit 时“进程退出”，随后清理成功。
    registry = ProcessRegistry(
        alive=set(),
        identities={4321: {"pid": 4321, "starttime": 1, "executable": "/ctl"}},
    ).install(monkeypatch)
    runtime = build_runtime(
        short_tmp,
        QuitKillsController(registry),
        launcher=LaunchRecorder(registry=registry),
    )
    with runtime.start(label="pct25", run_id="rid") as handle:
        assert handle.readiness is not None and handle.readiness.ready is True
    assert handle.cleanup_succeeded is True
    assert not any(
        path.is_dir() and path.name.startswith("edgesched_mps_")
        for path in short_tmp.iterdir()
    )


def test_snapshot_records_readiness_and_post_quit_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 假 controller PID 的存活状态由 registry 控制，不探测环境 PID。
    registry = ProcessRegistry(alive={CONTROLLER_PID}).install(monkeypatch)
    handle = build_handle(tmp_path, QuitKillsController(registry))
    handle.wait_until_controller_ready(timeout_s=0.2)
    handle.cleanup()
    snapshot = handle.snapshot()
    assert snapshot["controller_readiness"]["ready"] is True
    assert snapshot["controller_pid_file"].endswith("nvidia-cuda-mps-control.pid")
    assert "post_quit_evidence" in snapshot
    assert isinstance(snapshot["events"], list)
    assert snapshot["events"], "控制命令必须被记录为事件"


def test_wait_until_controller_ready_times_out_with_evidence(tmp_path: Path) -> None:
    handle = build_handle(tmp_path, ScriptedRunner(), with_pid_file=False)
    started = time.monotonic()
    with pytest.raises(MpsRuntimeError):
        handle.wait_until_controller_ready(timeout_s=0.1)
    assert time.monotonic() - started < 2.0
    assert handle.readiness is not None
    assert handle.readiness.pid_file_exists is False
