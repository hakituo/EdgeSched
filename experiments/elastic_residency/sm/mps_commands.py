"""MPS 常量、结果与解析原语。

命令描述与执行器位于 ``mps_protocol``；本模块只保留常量、状态结构与
纯函数解析，避免单文件过大。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

MPS_CONTROL_BINARY_DEFAULT = "nvidia-cuda-mps-control"

ENV_PIPE_DIRECTORY = "CUDA_MPS_PIPE_DIRECTORY"
ENV_LOG_DIRECTORY = "CUDA_MPS_LOG_DIRECTORY"
ENV_ACTIVE_THREAD_PERCENTAGE = "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"

#: 必须早于客户端创建任何 CUDA context 注入的环境变量。
CLIENT_ENV_KEYS = (
    ENV_PIPE_DIRECTORY,
    ENV_LOG_DIRECTORY,
    ENV_ACTIVE_THREAD_PERCENTAGE,
)

#: control daemon 就绪后在私有 pipe 目录写出的 PID 文件（NVIDIA 文档约定）。
CONTROLLER_PID_FILENAME = "nvidia-cuda-mps-control.pid"

PERCENTAGE_MIN = 1
PERCENTAGE_MAX = 100


class MpsRuntimeError(RuntimeError):
    """MPS daemon 生命周期操作失败。"""


@dataclass(frozen=True)
class MpsBinaryInfo:
    """``nvidia-cuda-mps-control`` 的探测结果。"""

    binary: str
    path: str | None
    available: bool
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "binary": self.binary,
            "path": self.path,
            "available": self.available,
            "reason": self.reason,
        }


@dataclass
class MpsDaemonHealth:
    """一次查询的快照。"""

    server_count: int
    client_count: int
    servers: tuple[str, ...] = ()
    clients: tuple[int, ...] = ()
    query: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "server_count": self.server_count,
            "client_count": self.client_count,
            "servers": list(self.servers),
            "clients": list(self.clients),
            "query": self.query.to_dict() if self.query else None,
        }


@dataclass(frozen=True)
class ControllerReadiness:
    """control daemon 就绪证据。

    server 列表为空是**合法**就绪状态：``-d`` 只拉起 controller，
    MPS server 由首个 CUDA client 按需启动。
    """

    pid_file: str
    pid_file_exists: bool
    pid: int | None
    pid_alive: bool | None
    ready: bool
    reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "pid_file": self.pid_file,
            "pid_file_exists": self.pid_file_exists,
            "pid": self.pid,
            "pid_alive": self.pid_alive,
            "ready": self.ready,
            "reason": self.reason,
            "server_count": None,
        }


def as_text(value: Any) -> str:
    """把子进程输出统一成字符串。"""

    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def parse_server_lines(stdout: str) -> tuple[str, ...]:
    """解析 ``get_server_list`` 输出；没有 server 是正常状态。"""

    return tuple(
        line.strip()
        for line in stdout.splitlines()
        if line.strip() and not line.strip().lower().startswith("no server")
    )


def parse_server_pids(server_lines: Any) -> tuple[int, ...]:
    """从 ``get_server_list`` 的行里解析 server PID。

    MPS 的 server 标识本身就是 PID，持久记录它才能在 controller 退出后
    继续核对进程是否真的消失。
    """

    pids: list[int] = []
    for line in server_lines:
        for part in str(line).replace("server", " ").split():
            if part.isdigit():
                pids.append(int(part))
                break
    return tuple(pids)


def process_starttime(pid: int) -> int | None:
    """读取 ``/proc/<pid>/stat`` 的 starttime（第 22 个字段）。"""

    stat_path = Path(f"/proc/{pid}/stat")
    if not stat_path.is_file():
        return None
    try:
        raw = stat_path.read_text(encoding="utf-8")
    except OSError:
        return None
    # comm 字段可能含空格与括号，从最后一个 ')' 之后再切分。
    tail = raw.rpartition(")")[2].split()
    if len(tail) < 20:
        return None
    try:
        return int(tail[19])
    except ValueError:
        return None


def process_executable(pid: int) -> str | None:
    """读取 ``/proc/<pid>/exe``；非 POSIX 或不可读时返回 ``None``。"""

    readlink = getattr(os, "readlink", None)
    if readlink is None:
        return None
    try:
        return str(readlink(f"/proc/{pid}/exe"))
    except OSError:
        return None


def process_identity(pid: int) -> dict[str, Any]:
    """采集用于识别进程身份的证据（避免 PID 复用误判）。"""

    return {
        "pid": pid,
        "starttime": process_starttime(pid),
        "executable": process_executable(pid),
    }


def server_process_status(pid: int, identity: Mapping[str, Any]) -> str:
    """判断退出前记录的 server 进程当前状态。

    返回值：``gone``（已消失）/ ``alive``（同一进程仍在）/
    ``reused``（PID 被别的进程复用，原进程已消失）/ ``unknown``（无法确认）。
    """

    alive = pid_is_alive(pid)
    if alive is None:
        return "unknown"
    if not alive:
        return "gone"
    current = process_identity(pid)
    recorded_start = identity.get("starttime")
    current_start = current.get("starttime")
    if recorded_start is not None and current_start is not None:
        return "alive" if recorded_start == current_start else "reused"
    recorded_exe = identity.get("executable")
    current_exe = current.get("executable")
    if recorded_exe and current_exe:
        return "alive" if recorded_exe == current_exe else "reused"
    return "unknown"


def parse_pid_lines(stdout: str) -> tuple[int, ...]:
    """解析 ``get_client_list`` 输出中的 PID。"""

    pids: list[int] = []
    for line in stdout.splitlines():
        token = line.strip()
        if not token:
            continue
        for part in token.replace("PID", " ").split():
            if part.isdigit():
                pids.append(int(part))
                break
    return tuple(pids)


def read_controller_pid(pid_file: str) -> tuple[int | None, str | None]:
    """读取 control daemon PID 文件；返回 ``(pid, 失败原因)``。"""

    path = Path(pid_file)
    if not path.is_file():
        return None, "controller_pid_file_missing"
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None, "controller_pid_file_unreadable"
    if not raw.isdigit():
        return None, "controller_pid_file_not_numeric"
    return int(raw), None


def pid_is_alive(pid: int) -> bool | None:
    """POSIX 上检查 PID 是否存活；非 POSIX 返回 ``None``（不可判断）。

    Windows 的 ``os.kill`` 语义与 POSIX 不同（信号集受限且可能抛
    ``PermissionError``），因此只在 POSIX 上给出确定结论。
    """

    if os.name != "posix":
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # 进程存在但无权发信号，仍视为存活。
        return True
    except OSError:
        return None
    return True


def validate_percentage(percentage: int) -> int:
    """校验客户端执行资源上限百分比。"""

    if isinstance(percentage, bool) or not isinstance(percentage, int):
        raise MpsRuntimeError("percentage 必须是整数")
    if not PERCENTAGE_MIN <= percentage <= PERCENTAGE_MAX:
        raise MpsRuntimeError(
            f"percentage 必须位于 [{PERCENTAGE_MIN}, {PERCENTAGE_MAX}]"
        )
    return percentage


def unmanaged_environment(
    base_env: Any | None = None,
) -> dict[str, str]:
    """构造普通 CUDA（不受控共享）环境：显式移除 MPS 变量。"""

    env = dict(base_env if base_env is not None else os.environ)
    for key in CLIENT_ENV_KEYS:
        env.pop(key, None)
    return env
