"""controller 身份退出验证的单元测试。

PID 文件消失后不再调用控制协议，而是用退出前记录的 controller
PID + starttime/executable 判断原 controller 是否真的消失。
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
    ProcessRegistry,
    ScriptedRunner,
    build_handle,
)


@pytest.fixture(autouse=True)
def _allow_fake_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "access", lambda *_args: True)


def _proc_identity(pid: int, starttime: int, executable: str) -> dict:
    return {"pid": pid, "starttime": starttime, "executable": executable}


def _controller_identity(starttime: int = 500, executable: str = "/ctl") -> dict:
    return _proc_identity(CONTROLLER_PID, starttime, executable)


def test_controller_identity_alive_after_pid_file_gone_is_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PID 文件消失但原 controller 身份仍存活 → 必须失败。"""

    # controller 进程仍在且身份与退出前记录一致；server 已消失。
    ProcessRegistry(
        alive={CONTROLLER_PID},
        identities={
            CONTROLLER_PID: _controller_identity(),
            SERVER_PID: _proc_identity(SERVER_PID, 100, "/srv"),
        },
    ).install(monkeypatch)

    # ScriptedRunner 的 quit 会删除 PID 文件，get_server_list 返回空。
    runner = ScriptedRunner()
    handle = build_handle(tmp_path, runner)

    assert handle.cleanup() is False
    assert "controller_gone_timeout" in (handle.cleanup_error or "")
    evidence = handle.post_quit_evidence
    assert evidence["controller_present"] is True
    assert evidence["controller_status"] == "alive"
    assert evidence["reached_quiescent"] is False
    assert handle.root.exists(), "controller 仍存活时不得删除私有目录"


def test_controller_pid_reused_after_pid_file_gone_is_not_residual(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PID 文件消失且 controller PID 被复用（starttime 不同）→ 不算残留。"""

    # 第一次查询（capture 时）身份是 starttime=500；之后同一 PID 属于另一进程。
    ProcessRegistry(
        alive={CONTROLLER_PID},
        identities={
            CONTROLLER_PID: [
                _controller_identity(),
                _controller_identity(starttime=999, executable="/other"),
            ],
            SERVER_PID: _proc_identity(SERVER_PID, 100, "/srv"),
        },
    ).install(monkeypatch)
    runner = ScriptedRunner()
    handle = build_handle(tmp_path, runner)

    assert handle.cleanup() is True
    evidence = handle.post_quit_evidence
    assert evidence["controller_status"] == "reused"
    assert evidence["controller_present"] is False
    assert evidence["reached_quiescent"] is True
    assert handle.root.exists() is False


def test_controller_identity_unverifiable_is_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """controller 身份无法验证 → fail-closed，保留私有目录与原因。"""

    # 观察过 controller PID 文件，但没有可核对的身份快照。
    ProcessRegistry(
        alive={CONTROLLER_PID, SERVER_PID},
        identities={SERVER_PID: _proc_identity(SERVER_PID, 100, "/srv")},
    ).install(monkeypatch)

    runner = ScriptedRunner()
    handle = build_handle(tmp_path, runner)
    handle.controller_snapshot = {}

    assert handle.cleanup() is False
    evidence = handle.post_quit_evidence
    assert evidence["controller_status"] == "unknown"
    assert evidence["unverifiable_controller"] is True
    assert evidence["reached_quiescent"] is False
    assert handle.retained_artifacts["private_root"] == str(handle.root)
    assert handle.root.exists()
