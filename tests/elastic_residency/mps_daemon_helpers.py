"""MPS daemon 测试共享替身（不是 pytest 收集对象）。

进程启动通过 ``MpsRuntime.process_launcher`` 注入；控制命令按 stdin 回放。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from experiments.elastic_residency.sm.mps_protocol import (
    CommandResult,
    CommandSpec,
)
from experiments.elastic_residency.sm.mps_daemon import MpsDaemonHandle
from experiments.elastic_residency.sm.mps_runtime import MpsRuntime

BINARY_NAME = "nvidia-cuda-mps-control"
PID_FILENAME = "nvidia-cuda-mps-control.pid"

#: controller 与 server 使用不同 PID，便于区分两类证据。
CONTROLLER_PID = 4242
SERVER_PID = 5353


class ProcessRegistry:
    """可控进程存活/身份替身；``identities[pid]`` 可为 dict 列表（按调用次序返回）。"""

    def __init__(
        self,
        *,
        alive: set[int] | None = None,
        identities: dict[int, Any] | None = None,
    ) -> None:
        self.alive = set(alive or set())
        self.identities = dict(identities or {})
        self._identity_calls: dict[int, int] = {}

    def identity_for(self, pid: int) -> dict:
        entry = self.identities.get(pid, {})
        if isinstance(entry, list):
            index = self._identity_calls.get(pid, 0)
            self._identity_calls[pid] = index + 1
            return entry[min(index, len(entry) - 1)]
        return dict(entry or {})

    def register_pid_file(self, pipe_dir: str) -> None:
        """读取刚写出的 controller PID 文件并标记该 PID 存活。"""

        pid_path = Path(pipe_dir) / PID_FILENAME
        if not pid_path.is_file():
            return
        try:
            self.alive.add(int(pid_path.read_text(encoding="utf-8").strip()))
        except ValueError:
            return

    def kill_pid_file(self, pipe_dir: str) -> None:
        """quit 时把 PID 文件里的 controller 标记为已退出。"""

        pid_path = Path(pipe_dir) / PID_FILENAME
        if not pid_path.is_file():
            return
        try:
            self.alive.discard(int(pid_path.read_text(encoding="utf-8").strip()))
        except ValueError:
            return

    def install(self, monkeypatch: Any) -> "ProcessRegistry":
        """替换实现模块与调用方命名空间里引用的名字。"""

        from experiments.elastic_residency.sm import (
            mps_commands,
            mps_daemon,
            mps_queries,
            mps_shutdown,
        )

        for target in (mps_commands, mps_daemon, mps_queries, mps_shutdown):
            if hasattr(target, "pid_is_alive"):
                monkeypatch.setattr(
                    target, "pid_is_alive", lambda pid: pid in self.alive
                )
            if hasattr(target, "process_identity"):
                monkeypatch.setattr(
                    target, "process_identity", self.identity_for
                )
        return self


class FakeProcess:
    """替身 daemon 进程；不启动真实子进程。"""

    def __init__(self, argv: Any = None) -> None:
        self.argv = argv
        self.pid = 4321
        self._returncode: int | None = None

    def poll(self) -> int | None:
        return self._returncode

    def wait(self, timeout: float | None = None) -> int:
        self._returncode = 0 if self._returncode is None else self._returncode
        return self._returncode

    def terminate(self) -> None:
        self._returncode = -15

    def kill(self) -> None:
        self._returncode = -9


class LaunchRecorder:
    """替身进程启动器；可模拟写出 controller PID 文件（并用 registry 标记存活）。"""

    def __init__(
        self, *, write_pid: bool = True, registry: ProcessRegistry | None = None
    ) -> None:
        self.write_pid = write_pid
        self.registry = registry
        self.launches: list[list[Any]] = []

    def __call__(self, argv: Any, **kwargs: Any) -> FakeProcess:
        self.launches.append(list(argv))
        env = kwargs.get("env") or {}
        pipe_dir = env.get("CUDA_MPS_PIPE_DIRECTORY")
        if self.write_pid and pipe_dir:
            Path(pipe_dir, PID_FILENAME).write_text("4321", encoding="utf-8")
            if self.registry is not None:
                self.registry.register_pid_file(pipe_dir)
        return FakeProcess(argv)


class ScriptedRunner:
    """按 stdin 控制命令回放结果的命令执行器。"""

    def __init__(self, script: Mapping[str, str] | None = None) -> None:
        self._script = dict(script or {})
        self.calls: list[tuple[tuple[str, ...], str | None]] = []
        self.server_response = "No servers are currently running.\n"
        self.client_connected = False

    def __call__(self, spec: CommandSpec, env: Mapping[str, str]) -> CommandResult:
        self.calls.append((tuple(spec.argv), spec.stdin))
        command = (spec.stdin or "").strip()
        if command in self._script:
            return CommandResult(
                tuple(spec.argv), spec.stdin, 0, self._script[command], ""
            )
        if command == "get_server_list":
            stdout = (
                f"server {SERVER_PID}\n"
                if self.client_connected
                else self.server_response
            )
            return CommandResult(tuple(spec.argv), spec.stdin, 0, stdout, "")
        if command.startswith("get_client_list"):
            stdout = "  PID 999\n" if self.client_connected else ""
            return CommandResult(tuple(spec.argv), spec.stdin, 0, stdout, "")
        if command == "quit":
            # 真实 control daemon 退出时会移除自己的 PID 文件并带走 server。
            self.client_connected = False
            pipe_dir = env.get("CUDA_MPS_PIPE_DIRECTORY")
            if pipe_dir:
                pid_file = Path(pipe_dir, PID_FILENAME)
                if pid_file.exists():
                    pid_file.unlink()
            return CommandResult(tuple(spec.argv), spec.stdin, 0, "", "")
        return CommandResult(tuple(spec.argv), spec.stdin, 0, "", "")

    @property
    def commands(self) -> list[str]:
        return [(stdin or "").strip() for _argv, stdin in self.calls]


class QuitKillsController(ScriptedRunner):
    """quit 时把 PID 文件中的 controller 标记为已退出。"""

    def __init__(self, registry: ProcessRegistry) -> None:
        super().__init__()
        self.registry = registry

    def __call__(self, spec: CommandSpec, env: Mapping[str, str]) -> CommandResult:
        if (spec.stdin or "").strip() == "quit":
            pipe_dir = env.get("CUDA_MPS_PIPE_DIRECTORY")
            if pipe_dir:
                self.registry.kill_pid_file(pipe_dir)
        return super().__call__(spec, env)


def write_fake_binary(tmp_path: Path) -> str:
    binary = tmp_path / "bin" / BINARY_NAME
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    return str(binary)


def build_runtime(
    tmp_path: Path,
    runner: ScriptedRunner,
    *,
    launcher: LaunchRecorder | None = None,
    start_timeout_s: float = 0.2,
) -> MpsRuntime:
    return MpsRuntime(
        binary_path=write_fake_binary(tmp_path),
        temp_root=tmp_path,
        start_timeout_s=start_timeout_s,
        command_timeout_s=0.1,
        runner=runner,
        env={"PATH": str(tmp_path)},
        process_launcher=launcher or LaunchRecorder(),
    )


class PostQuitServerRunner(ScriptedRunner):
    """quit 前按 ``client_connected`` 应答，quit 后改用指定行为（server 残留/查询失败）。"""

    def __init__(self, response: CommandResult | None = None) -> None:
        super().__init__()
        self.quit_issued = False
        self._response = response

    def _drop_pid_file(self, env: Mapping[str, str]) -> None:
        pipe_dir = env.get("CUDA_MPS_PIPE_DIRECTORY")
        if not pipe_dir:
            return
        pid_file = Path(pipe_dir, PID_FILENAME)
        if pid_file.exists():
            pid_file.unlink()

    def __call__(self, spec: CommandSpec, env: Mapping[str, str]) -> CommandResult:
        command = (spec.stdin or "").strip()
        if command == "quit":
            self.quit_issued = True
            self._drop_pid_file(env)
            return CommandResult(tuple(spec.argv), spec.stdin, 0, "", "")
        if command == "get_server_list":
            self.calls.append((tuple(spec.argv), spec.stdin))
            if not self.quit_issued:
                return super().__call__(spec, env)
            if self._response is not None:
                return CommandResult(
                    tuple(spec.argv),
                    spec.stdin,
                    self._response.returncode,
                    self._response.stdout,
                    self._response.stderr,
                    timed_out=self._response.timed_out,
                    error=self._response.error,
                )
            return CommandResult(
                tuple(spec.argv), spec.stdin, 0, f"server {SERVER_PID}\n", ""
            )
        return super().__call__(spec, env)


def observe_server(handle: MpsDaemonHandle, runner: ScriptedRunner) -> None:
    """在退出前真实观察到一次 MPS server（建立持久的身份记录）。"""

    runner.client_connected = True
    assert handle.query_servers() == (f"server {SERVER_PID}",)
    assert handle.server_observed is True
    assert SERVER_PID in handle.observed_server_identities


def build_handle(
    tmp_path: Path,
    runner: ScriptedRunner,
    *,
    with_pid_file: bool = True,
) -> MpsDaemonHandle:
    """构造一个不启动真实进程的 daemon 句柄。"""

    root = tmp_path / "run"
    pipe = root / "pipe"
    log = root / "log"
    pipe.mkdir(parents=True, exist_ok=True)
    log.mkdir(parents=True, exist_ok=True)
    if with_pid_file:
        (pipe / PID_FILENAME).write_text(str(CONTROLLER_PID), encoding="utf-8")
    return MpsDaemonHandle(
        runtime=build_runtime(tmp_path, runner),
        binary_path=write_fake_binary(tmp_path),
        label="pct50",
        run_id="rid",
        root=root,
        pipe_dir=pipe,
        log_dir=log,
        env={"CUDA_MPS_PIPE_DIRECTORY": str(pipe)},
        process=FakeProcess(),
        daemon_stdout_path=root / "daemon_stdout.log",
        daemon_stderr_path=root / "daemon_stderr.log",
        stdout_handle=None,
        stderr_handle=None,
        start_timeout_s=0.2,
        # 测试环境对 unlink/rmtree 有删除守卫，单次可能耗数百毫秒，
        # 0.1s 的等待窗口会被整个吃掉导致假超时。
        command_timeout_s=1.0,
    )
