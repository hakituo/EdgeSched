"""MPS 启动器（``MpsRuntime``）的单元测试：探测与启动装配。

进程启动通过 ``process_launcher`` 注入，不 monkeypatch 全局
``subprocess.Popen``，避免影响其它组件。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import pytest

from experiments.elastic_residency.sm.mps_commands import (
    ENV_LOG_DIRECTORY,
    ENV_PIPE_DIRECTORY,
    MpsRuntimeError,
)
from experiments.elastic_residency.sm.mps_protocol import (
    CommandResult,
    CommandSpec,
)
from experiments.elastic_residency.sm.mps_runtime import MpsRuntime

BINARY_NAME = "nvidia-cuda-mps-control"


class _FakeProcess:
    def __init__(self, argv: Any = None) -> None:
        self.argv = argv
        self.pid = 4321

    def poll(self) -> int | None:
        return None

    def wait(self, timeout: float | None = None) -> int:
        return 0

    def terminate(self) -> None:
        pass

    def kill(self) -> None:
        pass


class _LaunchRecorder:
    """替身进程启动器；可选地模拟 daemon 写出 controller PID 文件。"""

    def __init__(self, *, write_pid: bool = True, registry: Any = None) -> None:
        self.write_pid = write_pid
        self.registry = registry
        self.launches: list[list[Any]] = []

    def __call__(self, argv: Any, **kwargs: Any) -> _FakeProcess:
        self.launches.append(list(argv))
        env = kwargs.get("env") or {}
        pipe_dir = env.get(ENV_PIPE_DIRECTORY)
        if self.write_pid and pipe_dir:
            Path(pipe_dir, "nvidia-cuda-mps-control.pid").write_text(
                "4321", encoding="utf-8"
            )
            if self.registry is not None:
                self.registry.register_pid_file(pipe_dir)
        return _FakeProcess(argv)


class _QuietRunner:
    """最小执行器：``quit`` 时移除 PID 文件，模拟 controller 正常关闭。"""

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], str | None]] = []

    def __call__(self, spec: CommandSpec, env: Mapping[str, str]) -> CommandResult:
        self.calls.append((tuple(spec.argv), spec.stdin))
        if (spec.stdin or "").strip() == "quit":
            pipe_dir = env.get("CUDA_MPS_PIPE_DIRECTORY")
            if pipe_dir:
                pid_file = Path(pipe_dir, "nvidia-cuda-mps-control.pid")
                if pid_file.exists():
                    pid_file.unlink()
        return CommandResult(tuple(spec.argv), spec.stdin, 0, "", "")


@pytest.fixture(autouse=True)
def _allow_fake_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    """让显式 binary_path 在 Windows 上也能通过可执行性检查。"""

    monkeypatch.setattr(os, "access", lambda *_args: True)


def _write_fake_binary(tmp_path: Path) -> str:
    binary = tmp_path / "bin" / BINARY_NAME
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    return str(binary)


def _runtime(
    tmp_path: Path,
    launcher: _LaunchRecorder,
    *,
    runner: Any = None,
    start_timeout_s: float = 0.2,
) -> MpsRuntime:
    return MpsRuntime(
        binary_path=_write_fake_binary(tmp_path),
        temp_root=tmp_path,
        start_timeout_s=start_timeout_s,
        command_timeout_s=0.1,
        runner=runner or _QuietRunner(),
        env={"PATH": str(tmp_path)},
        process_launcher=launcher,
    )


def test_detect_reports_missing_binary_with_reason(tmp_path: Path) -> None:
    runtime = MpsRuntime(binary="definitely-not-mps", env={"PATH": str(tmp_path)})
    info = runtime.detect()
    assert info.available is False
    assert info.reason == "binary_not_found_on_path"
    assert info.to_dict()["binary"] == "definitely-not-mps"


def test_detect_accepts_explicit_path_and_rejects_missing(tmp_path: Path) -> None:
    binary = tmp_path / BINARY_NAME
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    assert MpsRuntime(binary_path=str(binary)).detect().reason == "explicit_path"
    assert (
        MpsRuntime(binary_path=str(tmp_path / "nope")).detect().reason
        == "explicit_path_not_executable"
    )


def test_unavailable_backend_fails_closed(tmp_path: Path) -> None:
    runtime = MpsRuntime(
        binary="definitely-not-mps", temp_root=tmp_path, env={"PATH": str(tmp_path)}
    )
    with pytest.raises(MpsRuntimeError, match="MPS 控制二进制不可用"):
        runtime.start(label="pct50")
    assert not any(tmp_path.iterdir()), "失败时不得残留私有目录"


def test_start_launches_dash_d_and_creates_private_dirs(tmp_path: Path) -> None:
    launcher = _LaunchRecorder()
    runtime = _runtime(tmp_path, launcher)
    handle = runtime.start(label="pct50", run_id="rid")
    try:
        binary = _write_fake_binary(tmp_path)
        assert launcher.launches == [[binary, "-d"]]
        assert handle.pipe_dir.is_dir()
        assert handle.log_dir.is_dir()
        assert handle.env[ENV_PIPE_DIRECTORY] == str(handle.pipe_dir)
        assert handle.env[ENV_LOG_DIRECTORY] == str(handle.log_dir)
        assert handle.controller_pid_file.is_file()
        assert handle.readiness is not None and handle.readiness.ready is True
    finally:
        handle.cleanup()


def test_start_failure_comes_from_missing_controller_pid(tmp_path: Path) -> None:
    """controller 未写出 PID 文件才算启动失败，与 server 列表无关。"""

    launcher = _LaunchRecorder(write_pid=False)
    runtime = _runtime(tmp_path, launcher, start_timeout_s=0.15)
    with pytest.raises(MpsRuntimeError) as excinfo:
        runtime.start(label="pct50")
    message = str(excinfo.value)
    assert "nvidia-cuda-mps-control.pid" in message
    assert "get_server_list" not in message


def test_start_creates_unique_private_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # LaunchRecorder 写出的 controller PID 是 4321；写出时标记存活，
    # quit 时“进程退出”，随后清理成功（两次启动各自独立清理）。
    from tests.elastic_residency.mps_daemon_helpers import (
        ProcessRegistry,
        QuitKillsController,
    )

    registry = ProcessRegistry(
        alive=set(),
        identities={4321: {"pid": 4321, "starttime": 1, "executable": "/ctl"}},
    ).install(monkeypatch)
    launcher = _LaunchRecorder(registry=registry)
    runtime = _runtime(tmp_path, launcher, runner=QuitKillsController(registry))
    first = runtime.start(label="pct50")
    first.cleanup()
    second = runtime.start(label="pct25")
    second.cleanup()
    assert first.root != second.root
    assert not first.root.exists()
    assert not second.root.exists()
