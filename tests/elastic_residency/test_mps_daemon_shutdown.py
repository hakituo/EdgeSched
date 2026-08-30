"""MPS 关闭阶段按**进程身份**核对 server 是否真的消失。

controller 退出之后不得再调用 ``get_server_list``（前端可能重新拉起 control
daemon，协议也不可用），只能核对退出前记录的 server PID + starttime/executable。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

import pytest

from experiments.elastic_residency.sm.mps_protocol import CommandResult, CommandSpec
from tests.elastic_residency.mps_daemon_helpers import (
    CONTROLLER_PID,
    SERVER_PID,
    PostQuitServerRunner,
    ProcessRegistry,
    ScriptedRunner,
    build_handle,
    observe_server,
)


@pytest.fixture(autouse=True)
def _allow_fake_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "access", lambda *_args: True)


class _ProtocolDeadAfterQuit(PostQuitServerRunner):
    """post-quit 的 ``get_server_list`` 会连接失败，因此绝不能被调用。"""

    def __call__(self, spec: CommandSpec, env: Mapping[str, str]) -> CommandResult:
        command = (spec.stdin or "").strip()
        if command == "get_server_list" and self.quit_issued:
            self.calls.append((tuple(spec.argv), spec.stdin))
            return CommandResult(tuple(spec.argv), spec.stdin, 805, "", "unreachable")
        return super().__call__(spec, env)


def _proc_identity(pid: int, starttime: int, executable: str) -> dict:
    return {"pid": pid, "starttime": starttime, "executable": executable}


def _identity(starttime: int, executable: str) -> dict:
    return _proc_identity(SERVER_PID, starttime, executable)


def _controller_identity(starttime: int = 500, executable: str = "/ctl") -> dict:
    return _proc_identity(CONTROLLER_PID, starttime, executable)


def _gone_registry() -> ProcessRegistry:
    """controller 与 server 都已消失（identity 可核对）的默认替身。"""

    return ProcessRegistry(
        alive=set(),
        identities={
            CONTROLLER_PID: _controller_identity(),
            SERVER_PID: _identity(100, "/srv"),
        },
    )


def test_server_gone_after_controller_exit_succeeds_without_post_quit_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """a. controller 退出 + 记录的 server 已消失 → 成功且不查 server list。"""

    _gone_registry().install(monkeypatch)

    runner = _ProtocolDeadAfterQuit()
    handle = build_handle(tmp_path, runner)
    observe_server(handle, runner)

    assert handle.cleanup() is True
    evidence = handle.post_quit_evidence
    assert evidence["controller_present"] is False
    assert evidence["controller_status"] == "gone"
    assert evidence["verification_mode"] == "recorded_process_identity"
    assert evidence["post_quit_get_server_list_calls"] == 0
    assert evidence["server_status"] == {str(SERVER_PID): "gone"}
    assert evidence["reached_quiescent"] is True
    assert handle.root.exists() is False


def test_server_still_alive_after_controller_exit_is_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """b. controller 已消失，但记录的 server 进程身份仍存活 → 必须失败。"""

    ProcessRegistry(
        alive={SERVER_PID},
        identities={
            CONTROLLER_PID: _controller_identity(),
            SERVER_PID: _identity(100, "/x"),
        },
    ).install(monkeypatch)

    runner = PostQuitServerRunner()
    handle = build_handle(tmp_path, runner)
    observe_server(handle, runner)

    assert handle.cleanup() is False
    assert "controller_gone_timeout" in (handle.cleanup_error or "")
    evidence = handle.post_quit_evidence
    assert evidence["verification_mode"] == "recorded_process_identity"
    assert evidence["server_status"] == {str(SERVER_PID): "alive"}
    assert evidence["reached_quiescent"] is False
    assert handle.root.exists(), "server 残留时不得删除私有目录"


def test_reused_pid_is_not_mistaken_for_lingering_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """c. PID 被别的进程复用（starttime/executable 不同）不得判为 server 残留。"""

    ProcessRegistry(
        alive={SERVER_PID},
        identities={
            CONTROLLER_PID: _controller_identity(),
            SERVER_PID: _identity(999, "/other"),
        },
    ).install(monkeypatch)

    runner = PostQuitServerRunner()
    handle = build_handle(tmp_path, runner)
    observe_server(handle, runner)
    # 观察到的身份是 starttime=100，之后该 PID 被别的进程占用。
    handle.observed_server_identities[SERVER_PID] = _identity(
        100, "/usr/bin/nvidia-cuda-mps-server"
    )

    assert handle.cleanup() is True
    evidence = handle.post_quit_evidence
    assert evidence["controller_status"] == "gone"
    assert evidence["server_status"] == {str(SERVER_PID): "reused"}
    assert evidence["reached_quiescent"] is True


def test_unverifiable_server_identity_is_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """server 身份无法确认时 fail-closed，并保留结构化原因与私有目录。"""

    ProcessRegistry(
        alive={SERVER_PID},
        identities={CONTROLLER_PID: _controller_identity()},
    ).install(monkeypatch)

    runner = PostQuitServerRunner()
    handle = build_handle(tmp_path, runner)
    observe_server(handle, runner)

    assert handle.cleanup() is False
    assert handle.post_quit_evidence["server_status"] == {str(SERVER_PID): "unknown"}
    assert handle.post_quit_evidence["unverifiable_servers"] == [str(SERVER_PID)]
    assert handle.retained_artifacts["private_root"] == str(handle.root)
    assert handle.root.exists()


def test_lingering_server_across_protocol_and_identity_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """controller 先仍在（协议路径看到残留），PID 文件消失后按身份继续判残留。"""

    ProcessRegistry(
        alive={CONTROLLER_PID, SERVER_PID},
        identities={
            CONTROLLER_PID: _controller_identity(),
            SERVER_PID: _identity(100, "/x"),
        },
    ).install(monkeypatch)

    class LingeringServer(PostQuitServerRunner):
        def __init__(self) -> None:
            super().__init__()
            self.server_lists = 0

        def __call__(self, spec: CommandSpec, env: Mapping[str, str]) -> CommandResult:
            command = (spec.stdin or "").strip()
            if command == "get_server_list":
                self.calls.append((tuple(spec.argv), spec.stdin))
                self.server_lists += 1
                if self.server_lists == 2:
                    self._drop_pid_file(env)
                return CommandResult(
                    tuple(spec.argv), spec.stdin, 0, f"server {SERVER_PID}\n", ""
                )
            if command == "quit":
                self.quit_issued = True
                return CommandResult(tuple(spec.argv), spec.stdin, 0, "", "")
            return super().__call__(spec, env)

    runner = LingeringServer()
    handle = build_handle(tmp_path, runner)
    observe_server(handle, runner)
    assert handle.cleanup() is False
    assert "controller_gone_timeout" in (handle.cleanup_error or "")
    assert runner.server_lists >= 2
    assert handle.post_quit_evidence["server_count"] >= 1
    assert handle.post_quit_evidence["server_status"], "必须观察到残留 server"
    assert all(
        status == "alive"
        for status in handle.post_quit_evidence["server_status"].values()
    )


def test_persistent_controller_pid_is_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """controller PID 文件持续存在且 PID 存活 → 必须失败。"""

    ProcessRegistry(
        alive={CONTROLLER_PID},
        identities={CONTROLLER_PID: _controller_identity()},
    ).install(monkeypatch)

    class AliveController(ScriptedRunner):
        """quit 成功但 PID 文件仍在、server 已清空。"""

        def __call__(self, spec: CommandSpec, env: Mapping[str, str]) -> CommandResult:
            command = (spec.stdin or "").strip()
            if command == "quit":
                return CommandResult(tuple(spec.argv), spec.stdin, 0, "", "")
            if command == "get_server_list":
                self.calls.append((tuple(spec.argv), spec.stdin))
                return CommandResult(
                    tuple(spec.argv),
                    spec.stdin,
                    0,
                    "No servers are currently running.\n",
                    "",
                )
            return super().__call__(spec, env)

    handle = build_handle(tmp_path, AliveController())
    assert handle.cleanup() is False
    assert "controller_gone_timeout" in (handle.cleanup_error or "")
    assert handle.controller_pid_file.exists() is True
    assert handle.root.exists() is True


def test_controller_still_present_uses_control_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """controller 仍在时使用控制协议查询，而不是进程身份。"""

    class ControllerStaysUp(PostQuitServerRunner):
        """quit 成功但 controller 进程仍在（PID 文件未删）。"""

        def __call__(self, spec: CommandSpec, env: Mapping[str, str]) -> CommandResult:
            if (spec.stdin or "").strip() == "quit":
                self.quit_issued = True
                return CommandResult(tuple(spec.argv), spec.stdin, 0, "", "")
            return super().__call__(spec, env)

    ProcessRegistry(
        alive={CONTROLLER_PID},
        identities={
            CONTROLLER_PID: _controller_identity(),
            SERVER_PID: _identity(100, "/x"),
        },
    ).install(monkeypatch)

    runner = ControllerStaysUp()
    handle = build_handle(tmp_path, runner)
    observe_server(handle, runner)

    assert handle.cleanup() is False
    evidence = handle.post_quit_evidence
    assert evidence["controller_present"] is True
    assert evidence["controller_status"] == "alive"
    assert evidence["verification_mode"] == "control_protocol"
    assert evidence["post_quit_get_server_list_calls"] >= 1
    assert evidence["server_count"] >= 1
    assert handle.root.exists(), "controller 未退出时不得删除私有目录"
