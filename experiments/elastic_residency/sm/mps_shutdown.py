"""MPS 关闭阶段的证据采集与等待逻辑。

关键约定：**controller 退出之后不再调用 ``get_server_list``**。因为
``nvidia-cuda-mps-control`` 前端在 daemon 不存在时可能重新拉起一个
control daemon，协议查询本身也不可用。此时改用退出前记录的 server 进程
身份（PID + starttime/executable）核对进程是否真的消失。
"""

from __future__ import annotations

import time
from typing import Any

from .mps_commands import (
    MpsRuntimeError,
    pid_is_alive,
    process_identity,
    read_controller_pid,
    server_process_status,
)


def capture_shutdown_snapshot(handle: Any) -> None:
    """发送 ``quit`` 之前冻结 controller PID 与已观察 server 的进程身份。"""

    pid, _reason = read_controller_pid(str(handle.controller_pid_file))
    handle.controller_pid_file_observed = pid is not None
    handle.controller_snapshot = process_identity(pid) if pid is not None else {}
    handle.server_snapshot = dict(handle.observed_server_identities)


def _controller_identity_status(
    handle: Any, *, pid_file_present: bool
) -> tuple[str, bool, bool]:
    """用退出前记录的 controller 身份判断原 controller 是否还在。

    返回 ``(controller_status, controller_present, unverifiable_controller)``：

    - ``gone`` / ``reused``：原 controller 已消失（PID 被复用不算残留）；
    - ``alive``：确认是同一个 controller，仍在运行；
    - ``unknown``：无法确认 → fail-closed。
    """

    snapshot = handle.controller_snapshot or {}
    recorded_pid = snapshot.get("pid")
    if pid_file_present and recorded_pid is not None:
        status = server_process_status(int(recorded_pid), snapshot)
    elif pid_file_present:
        # PID 文件指向一个存在但无法核对的进程。
        status = "unknown"
    elif recorded_pid is not None:
        # PID 文件已消失：核对记录的 controller PID 身份，不再调用控制协议。
        status = server_process_status(int(recorded_pid), snapshot)
    elif not getattr(handle, "controller_pid_file_observed", False):
        # 从未观察到 controller PID 文件（例如启动阶段就失败），不存在可残留
        # 的 controller 进程，按已消失处理。
        status = "gone"
    else:
        status = "unknown"

    if status == "alive":
        return status, True, False
    if status in ("gone", "reused"):
        return status, False, False
    return status, False, True


def build_post_quit_evidence(handle: Any) -> dict[str, Any]:
    """构造一次退出后证据快照。

    ``verification_mode`` 说明本轮用哪种方式确认状态：

    - ``control_protocol``：controller 确认仍在（身份匹配），使用控制协议查询；
    - ``recorded_process_identity``：controller 已消失或无法用协议，只核对
      退出前记录的 controller / server 进程身份。
    """

    pid, _reason = read_controller_pid(str(handle.controller_pid_file))
    alive = pid_is_alive(pid) if pid is not None else None
    pid_file_exists = handle.controller_pid_file.is_file()
    pid_file_present = pid is not None and pid_file_exists and alive is not False

    controller_status, controller_present, unverifiable_controller = (
        _controller_identity_status(handle, pid_file_present=pid_file_present)
    )

    server_status: dict[str, str] = {}
    server_query_error: str | None = None
    servers_seen = 0
    if controller_present:
        verification_mode = "control_protocol"
        handle.post_quit_query_count += 1
        servers = handle.query_servers()
        server_query_error = handle.last_server_query_error
        for server in servers:
            server_status[str(server)] = (
                "alive" if server_query_error is None else "query_failed"
            )
    else:
        verification_mode = "recorded_process_identity"
        for server_pid, identity in handle.server_snapshot.items():
            server_status[str(server_pid)] = server_process_status(server_pid, identity)
    servers_seen = sum(
        1 for status in server_status.values() if status in ("alive", "query_failed")
    )

    unknown_servers = sorted(
        name for name, status in server_status.items() if status == "unknown"
    )
    all_servers_gone = set(server_status.values()) <= {"gone", "reused"}
    reached_quiescent = (
        controller_status in ("gone", "reused")
        and all_servers_gone
        and not unverifiable_controller
        and not unknown_servers
        and server_query_error is None
    )
    return {
        "controller_pid": pid,
        "controller_pid_alive": alive,
        "controller_pid_file_exists": pid_file_exists,
        "controller_present": controller_present,
        "controller_status": controller_status,
        "unverifiable_controller": unverifiable_controller,
        "controller_snapshot": dict(handle.controller_snapshot),
        "server_count": servers_seen,
        "server_status": server_status,
        "server_query_error": server_query_error,
        "server_observed_during_lifetime": handle.server_observed,
        "server_snapshot": {
            str(name): identity for name, identity in handle.server_snapshot.items()
        },
        "verification_mode": verification_mode,
        "post_quit_get_server_list_calls": handle.post_quit_query_count,
        "unverifiable_servers": unknown_servers,
        "reached_quiescent": reached_quiescent,
    }


def wait_for_controller_gone(handle: Any, *, timeout_s: float) -> bool:
    """等待 controller 与所有已观察 server 都确认消失；超时抛错。"""

    deadline = time.monotonic() + timeout_s
    while True:
        evidence = build_post_quit_evidence(handle)
        handle.post_quit_evidence = evidence
        if evidence["reached_quiescent"]:
            return True
        if time.monotonic() >= deadline:
            raise MpsRuntimeError(
                "MPS controller/server 在 "
                f"{timeout_s:.1f}s 后仍未退出 (controller_status="
                f"{evidence['controller_status']}, "
                f"server_status={evidence['server_status']}, "
                f"unverifiable_controller={evidence['unverifiable_controller']}, "
                f"unverifiable_servers={evidence['unverifiable_servers']})"
            )
        time.sleep(min(0.2, max(0.01, timeout_s / 20.0)))
