"""MPS interactive 控制协议的命令描述、结果证据与执行器。

``nvidia-cuda-mps-control`` 的控制命令（``get_server_list`` /
``get_client_list`` / ``ps`` / ``quit`` / ``start_server``）必须从 **stdin**
读取，argv 只含可执行文件路径；全程禁用 ``shell=True``。
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .mps_commands import (
    ENV_ACTIVE_THREAD_PERCENTAGE,
    ENV_LOG_DIRECTORY,
    ENV_PIPE_DIRECTORY,
    as_text,
    validate_percentage,
)


@dataclass(frozen=True)
class CommandSpec:
    """一次外部命令的完整描述。"""

    argv: tuple[str, ...]
    stdin: str | None = None
    timeout_s: float = 10.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "argv": list(self.argv),
            "stdin": self.stdin,
            "timeout_s": self.timeout_s,
        }


@dataclass(frozen=True)
class CommandResult:
    """一次外部命令执行的完整证据。

    控制命令位于 ``stdin_command``，不在 ``argv``；两者分别记录，
    便于审查命令通道是否走 interactive stdin 协议。
    """

    argv: tuple[str, ...]
    stdin_command: str | None
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "argv": list(self.argv),
            "stdin_command": self.stdin_command,
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "timed_out": self.timed_out,
            "error": self.error,
        }


CommandRunner = Callable[[CommandSpec, Mapping[str, str]], CommandResult]


def control_command_stdin(command: str) -> str:
    """把控制命令编码成 stdin 文本（结尾换行）。"""

    return f"{command.strip()}\n"


def build_client_environment(
    *,
    pipe_dir: str,
    log_dir: str,
    percentage: int,
    base_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """构造客户端环境变量。

    必须在客户端创建任何 CUDA context、导入 torch CUDA 后端或加载模型
    **之前**注入。``percentage`` 只是客户端执行资源上限。
    """

    value = validate_percentage(percentage)
    env = dict(base_env if base_env is not None else os.environ)
    env[ENV_PIPE_DIRECTORY] = str(pipe_dir)
    env[ENV_LOG_DIRECTORY] = str(log_dir)
    env[ENV_ACTIVE_THREAD_PERCENTAGE] = str(value)
    return env


def command_failure_reason(result: CommandResult, *, label: str) -> str | None:
    """命令只有完全成功才返回 ``None``。

    ``returncode=None``（进程没有产生退出码）、超时、OS 错误、非零退出码
    一律视为失败，绝不能折叠成“空结果”或“也许成功了”。
    """

    if result.timed_out:
        return f"{label}_timeout"
    if result.error is not None:
        return f"{label}_error:{result.error}"
    if result.returncode is None:
        return f"{label}_returncode_unknown"
    if result.returncode != 0:
        return f"{label}_returncode_{result.returncode}"
    return None


def quit_failure_reason(result: CommandResult) -> str | None:
    return command_failure_reason(result, label="quit")


def subprocess_runner(spec: CommandSpec, env: Mapping[str, str]) -> CommandResult:
    """默认命令执行器：命令写 stdin，参数列表形式，绝不使用 shell。"""

    items = [str(item) for item in spec.argv]
    try:
        completed = subprocess.run(  # noqa: S603 - 固定参数列表，无 shell
            items,
            env=dict(env),
            input=spec.stdin,
            timeout=spec.timeout_s,
            capture_output=True,
            text=True,
            check=False,
            shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        return CommandResult(
            argv=tuple(items),
            stdin_command=spec.stdin,
            returncode=None,
            stdout=as_text(exc.stdout),
            stderr=as_text(exc.stderr),
            timed_out=True,
            error="timeout",
        )
    except OSError as exc:
        return CommandResult(
            argv=tuple(items),
            stdin_command=spec.stdin,
            returncode=None,
            stdout="",
            stderr="",
            timed_out=False,
            error=f"os_error:{exc.__class__.__name__}",
        )
    return CommandResult(
        argv=tuple(items),
        stdin_command=spec.stdin,
        returncode=completed.returncode,
        stdout=as_text(completed.stdout),
        stderr=as_text(completed.stderr),
    )
