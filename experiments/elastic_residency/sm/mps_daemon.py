"""单个 MPS control daemon 的句柄：就绪判定、客户端环境与清理。

协议要点：

- 控制命令（``get_server_list`` / ``get_client_list`` / ``ps`` / ``quit`` /
  ``start_server``）通过 **stdin** 发送给 ``nvidia-cuda-mps-control``；
- ``-d`` 只拉起 control daemon，MPS server 由首个 CUDA client 按需启动，
  所以 controller 就绪判据是私有 pipe 目录中的 PID 文件，而不是 server 列表；
- 清理只针对本次 run 通过私有 pipe 目录启动的 daemon，以及本次
  ``tempfile.mkdtemp`` 创建的目录；不调用任何全局 kill。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

from .mps_artifacts import (
    archive_daemon_logs,
    build_daemon_snapshot,
    close_streams,
    residual_paths,
    terminate_process_if_alive,
)
from .mps_commands import (
    CLIENT_ENV_KEYS,
    CONTROLLER_PID_FILENAME,
    ControllerReadiness,
    MpsRuntimeError,
)
from .mps_shutdown import (
    build_post_quit_evidence,
    capture_shutdown_snapshot,
    wait_for_controller_gone,
)
from .mps_protocol import (
    CommandResult,
    CommandSpec,
    build_client_environment,
    control_command_stdin,
    quit_failure_reason,
)
from .mps_queries import MpsQueryMixin


class MpsDaemonHandle(MpsQueryMixin):
    """单个 MPS control daemon 的句柄；退出路径必须显式调用 ``cleanup()``。"""

    def __init__(
        self,
        *,
        runtime: Any,
        binary_path: str,
        label: str,
        run_id: str,
        root: Path,
        pipe_dir: Path,
        log_dir: Path,
        env: Mapping[str, str],
        process: subprocess.Popen,
        daemon_stdout_path: Path,
        daemon_stderr_path: Path,
        stdout_handle: Any,
        stderr_handle: Any,
        start_timeout_s: float,
        command_timeout_s: float,
    ) -> None:
        self._runtime = runtime
        self._binary_path = binary_path
        self.label = label
        self.run_id = run_id
        self.root = root
        self.pipe_dir = pipe_dir
        self.log_dir = log_dir
        self.env = dict(env)
        self.process = process
        self.daemon_stdout_path = daemon_stdout_path
        self.daemon_stderr_path = daemon_stderr_path
        self._stdout_handle = stdout_handle
        self._stderr_handle = stderr_handle
        self._start_timeout_s = start_timeout_s
        self._command_timeout_s = command_timeout_s
        self.events: list[dict[str, Any]] = []
        self.health_checks: list[Any] = []
        self.observed_server_identities: dict[int, dict[str, Any]] = {}
        self.server_snapshot: dict[int, dict[str, Any]] = {}
        self.controller_snapshot: dict[str, Any] = {}
        self.controller_pid_file_observed = False
        self.post_quit_query_count = 0
        self.archived_logs: dict[str, str] = {}
        self.retained_artifacts: dict[str, Any] = {}
        self.readiness: ControllerReadiness | None = None
        self.post_quit_evidence: dict[str, Any] = {}
        self.cleanup_succeeded: bool | None = None
        self.cleanup_error: str | None = None
        self.cleanup_commands: list[CommandResult] = []
        self.started = True

    # -------------------- 命令通道 --------------------

    @property
    def controller_pid_file(self) -> Path:
        return self.pipe_dir / CONTROLLER_PID_FILENAME

    def _runner(self) -> Any:
        injected = getattr(self._runtime, "runner", None)
        if injected is not None:
            return injected
        from .mps_protocol import subprocess_runner

        return subprocess_runner

    def _run(self, command: str, *, timeout_s: float | None = None) -> CommandResult:
        """把 interactive control command 写入 stdin；argv 只含程序路径。"""

        spec = CommandSpec(
            argv=(self._binary_path,),
            stdin=control_command_stdin(command),
            timeout_s=timeout_s or self._command_timeout_s,
        )
        result = self._runner()(spec, self.env)
        self.events.append(
            {
                "action": "control_command",
                "monotonic_ns": time.monotonic_ns(),
                **result.to_dict(),
            }
        )
        return result

    # -------------------- 就绪判定 --------------------

    # -------------------- 客户端环境 --------------------

    def client_environment(
        self,
        percentage: int,
        *,
        base_env: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        """构造客户端环境变量（在 CUDA 初始化之前注入）。"""

        return build_client_environment(
            pipe_dir=str(self.pipe_dir),
            log_dir=str(self.log_dir),
            percentage=percentage,
            base_env=base_env,
        )

    # -------------------- 清理 --------------------

    def cleanup(self, *, archive_dir: Path | None = None) -> bool:
        """停止本次 run 的 daemon；失败时**保留**私有目录与 PID 供人工处理。

        fail-closed 规则：

        - ``quit`` 必须 ``returncode == 0`` 且未超时且无 OS 错误；
        - ``_wait_for_controller_gone()`` 超时即抛错，绝不静默成功；
        - 只有 controller 确认退出后才回收私有目录，否则保留路径与 PID；
        - 关闭输出流之后再归档，保证日志包含最终关闭阶段。
        """

        if self.cleanup_succeeded is not None:
            return self.cleanup_succeeded
        error: str | None = None
        quiescent = False
        if self.started:
            self._capture_shutdown_snapshot()
            quit_result = self._run("quit", timeout_s=self._command_timeout_s)
            self.cleanup_commands.append(quit_result)
            error = quit_failure_reason(quit_result)
            if error is None:
                try:
                    quiescent = self._wait_for_controller_gone(
                        timeout_s=self._command_timeout_s
                    )
                except MpsRuntimeError as exc:
                    error = f"controller_gone_timeout:{exc}"
        if quiescent:
            try:
                self._terminate_process_if_alive()
            except MpsRuntimeError as exc:
                error = f"process_terminate_failed:{exc}"
                quiescent = False

        self._close_streams()
        self._archive_shutdown_logs(archive_dir)

        if error is None and quiescent:
            try:
                self._remove_private_dirs()
                self.started = False
            except Exception as exc:  # noqa: BLE001 - 清理必须记录并继续
                error = f"private_root_remove_failed:{exc}"
        else:
            # 未确认关闭：保留 pipe/log 目录与 controller PID 供人工处理。
            self.retained_artifacts = {
                "private_root": str(self.root),
                "controller_pid_file": str(self.controller_pid_file),
                "controller_pid": self.post_quit_evidence.get("controller_pid"),
                "reason": error or "controller_gone_not_confirmed",
            }

        self.cleanup_error = error or (
            None if quiescent else "controller_gone_not_confirmed"
        )
        self.cleanup_succeeded = error is None and quiescent
        return self.cleanup_succeeded

    def _archive_shutdown_logs(self, archive_dir: Path | None) -> None:
        """在 ``quit`` 与关闭输出流之后归档，确保包含最终关闭阶段。"""

        if archive_dir is None:
            return
        self.archived_logs = archive_daemon_logs(
            label=self.label,
            daemon_stdout=self.daemon_stdout_path,
            daemon_stderr=self.daemon_stderr_path,
            log_dir=self.log_dir,
            destination_root=archive_dir,
        )

    def _capture_shutdown_snapshot(self) -> None:
        capture_shutdown_snapshot(self)

    def _controller_gone_evidence(self) -> dict[str, Any]:
        return build_post_quit_evidence(self)

    def _wait_for_controller_gone(self, *, timeout_s: float) -> bool:
        """等待 controller 与所有已观察 server 消失；超时抛 ``MpsRuntimeError``。"""

        return wait_for_controller_gone(self, timeout_s=timeout_s)

    def residual_artifacts(self) -> list[str]:
        """清理后仍残存的文件系统痕迹。

        只做文件检查：此时不应再发起任何控制命令，否则前端可能重新拉起
        control daemon。server 是否已停止由 ``post_quit_evidence`` 记录。
        """

        return residual_paths(self.controller_pid_file, self.root)

    def _terminate_process_if_alive(self) -> None:
        terminate_process_if_alive(self.process)

    def _close_streams(self) -> None:
        close_streams(self._stdout_handle, self._stderr_handle)

    def _remove_private_dirs(self) -> None:
        """只删除本次 mkdtemp 创建的根目录及其子目录。"""

        if self.root is not None and self.root.exists():
            shutil.rmtree(self.root, ignore_errors=False)

    @property
    def binary_path(self) -> str:
        return self._binary_path

    def snapshot(self) -> dict[str, Any]:
        return build_daemon_snapshot(self)

    def __enter__(self) -> "MpsDaemonHandle":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.cleanup()
