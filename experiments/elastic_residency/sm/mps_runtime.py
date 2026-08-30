"""CUDA MPS control daemon 启动器（仅用于弹性驻留实验）。

``MpsDaemonHandle`` 位于 ``mps_daemon``，命令原语位于 ``mps_commands``。
设计边界：

- 每个静态配置使用全新子进程，并为单次 run 创建私有 pipe/log 目录；
- ``-d`` 只启动 control daemon；MPS server 由首个 CUDA client 按需拉起，
  因此启动阶段**不**等待 server 出现；
- 只 ``quit`` 本次 run 通过私有 pipe 目录启动的 daemon，只删除本次创建的目录；
- 后端不存在或启动失败时返回明确原因，由调用方给出 ``UNSUPPORTED``，
  不得静默退回普通 CUDA 共享。

第一版只使用 ``CUDA_MPS_ACTIVE_THREAD_PERCENTAGE``，对应能力只能是
``MPS_CAP``（客户端执行资源上限），不是独占 SM 分区。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping

from .mps_commands import (
    ENV_LOG_DIRECTORY,
    ENV_PIPE_DIRECTORY,
    MPS_CONTROL_BINARY_DEFAULT,
    MpsBinaryInfo,
    MpsRuntimeError,
)
from .mps_protocol import CommandRunner, subprocess_runner


class MpsRuntime:
    """MPS control daemon 的启动器；所有状态都限定在单次 run 的私有目录内。"""

    def __init__(
        self,
        *,
        binary: str = MPS_CONTROL_BINARY_DEFAULT,
        binary_path: str | None = None,
        temp_root: str | Path | None = None,
        start_timeout_s: float = 15.0,
        command_timeout_s: float = 10.0,
        runner: CommandRunner | None = None,
        env: Mapping[str, str] | None = None,
        process_launcher: Callable[..., Any] | None = None,
    ) -> None:
        self._binary = binary
        self._binary_path = binary_path
        self._temp_root = Path(temp_root) if temp_root is not None else None
        self._start_timeout_s = start_timeout_s
        self._command_timeout_s = command_timeout_s
        self._runner = runner
        self._env_base = dict(env if env is not None else os.environ)
        #: 可注入的进程启动器（默认 ``subprocess.Popen``），用于测试替换。
        self._process_launcher = process_launcher or subprocess.Popen

    @property
    def binary(self) -> str:
        return self._binary

    @property
    def runner(self) -> CommandRunner | None:
        """注入的命令执行器；``None`` 表示由 handle 使用默认执行器。"""

        return self._runner

    def detect(self) -> MpsBinaryInfo:
        """探测控制二进制是否存在；缺失时给出明确原因而不是静默降级。"""

        if self._binary_path:
            candidate = Path(self._binary_path)
            if candidate.is_file() and os.access(str(candidate), os.X_OK):
                return MpsBinaryInfo(self._binary, str(candidate), True, "explicit_path")
            return MpsBinaryInfo(
                self._binary, str(candidate), False, "explicit_path_not_executable"
            )
        found = shutil.which(self._binary, path=self._env_base.get("PATH"))
        if found:
            return MpsBinaryInfo(self._binary, found, True, "found_on_path")
        return MpsBinaryInfo(self._binary, None, False, "binary_not_found_on_path")

    def start(self, *, label: str, run_id: str = "run") -> Any:
        """启动一个只属于本次 run 的 control daemon。

        只等待 controller 就绪（PID 文件），不等 MPS server 出现。
        """

        # 延迟导入避免与 mps_daemon 形成模块级循环依赖。
        from .mps_daemon import MpsDaemonHandle

        info = self.detect()
        if not info.available or not info.path:
            raise MpsRuntimeError(
                f"MPS 控制二进制不可用: {info.binary} ({info.reason})"
            )
        root = self._create_private_root(label=label)
        pipe_dir = root / "pipe"
        log_dir = root / "log"
        pipe_dir.mkdir(parents=False, exist_ok=False)
        log_dir.mkdir(parents=False, exist_ok=False)
        env = dict(self._env_base)
        env[ENV_PIPE_DIRECTORY] = str(pipe_dir)
        env[ENV_LOG_DIRECTORY] = str(log_dir)
        stdout_path = root / "daemon_stdout.log"
        stderr_path = root / "daemon_stderr.log"
        stdout_handle = stdout_path.open("w", encoding="utf-8")
        stderr_handle = stderr_path.open("w", encoding="utf-8")
        process = self._process_launcher(  # noqa: S603 - 固定参数列表，无 shell
            [info.path, "-d"],
            env=env,
            stdout=stdout_handle,
            stderr=stderr_handle,
            stdin=subprocess.DEVNULL,
            shell=False,
        )
        handle = MpsDaemonHandle(
            runtime=self,
            binary_path=info.path,
            label=label,
            run_id=run_id,
            root=root,
            pipe_dir=pipe_dir,
            log_dir=log_dir,
            env=env,
            process=process,
            daemon_stdout_path=stdout_path,
            daemon_stderr_path=stderr_path,
            stdout_handle=stdout_handle,
            stderr_handle=stderr_handle,
            start_timeout_s=self._start_timeout_s,
            command_timeout_s=self._command_timeout_s,
        )
        try:
            handle.wait_until_controller_ready(timeout_s=self._start_timeout_s)
        except MpsRuntimeError:
            handle.cleanup()
            raise
        return handle

    def _create_private_root(self, *, label: str) -> Path:
        safe_label = "".join(
            char if char.isalnum() or char in "-_" else "_" for char in label
        )[:48]
        prefix = f"edgesched_mps_{safe_label}_"
        if self._temp_root is None:
            return Path(tempfile.mkdtemp(prefix=prefix))
        self._temp_root.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix=prefix, dir=str(self._temp_root)))
