"""MPS daemon 的查询操作：server / client / ps / start_server。

所有命令都通过 ``handle._run()`` 以 interactive stdin 发送；本模块只负责
命令选择与结果解析，不负责生命周期。
"""

from __future__ import annotations

import os
import time
from typing import Any

import time

from .mps_commands import (
    ControllerReadiness,
    MpsDaemonHealth,
    MpsRuntimeError,
    parse_pid_lines,
    parse_server_lines,
    parse_server_pids,
    pid_is_alive,
    process_identity,
    read_controller_pid,
)
from .mps_protocol import CommandResult, command_failure_reason

CONTROLLER_PID_FILENAME = "nvidia-cuda-mps-control.pid"


class MpsQueryMixin:
    """供 ``MpsDaemonHandle`` 继承的查询与就绪判定能力。"""

    health_checks: list[MpsDaemonHealth]
    readiness: ControllerReadiness | None
    pipe_dir: Any
    #: 本次生命周期内是否曾观察到 MPS server；持久记录，不从单次证据推断。
    server_observed: bool = False
    #: 最近一次 ``get_server_list`` 的结构化失败原因（成功为 ``None``）。
    last_server_query_error: str | None = None
    #: 观察到的 server PID → 进程身份证据（starttime / executable）。
    observed_server_identities: dict[int, dict[str, Any]]
    #: 发送 ``quit`` 之前冻结的 server 身份快照。
    server_snapshot: dict[int, dict[str, Any]]

    def _run(self, command: str, *, timeout_s: float | None = None) -> CommandResult:
        raise NotImplementedError  # pragma: no cover - 由 handle 实现

    @property
    def controller_pid_file(self) -> Any:
        raise NotImplementedError  # pragma: no cover - 由 handle 实现

    def controller_ready_evidence(self) -> ControllerReadiness:
        """controller 就绪证据：PID 文件存在且其中 PID 存活。

        server 列表为空是**合法**的就绪状态（server 按需启动）。
        """

        pid_file = str(self.controller_pid_file)
        pid, reason = read_controller_pid(pid_file)
        exists = self.controller_pid_file.is_file()
        alive = pid_is_alive(pid) if pid is not None else None
        ready = pid is not None and alive is not False
        if ready:
            reason = None
        elif reason is None:
            reason = "controller_pid_not_alive"
        return ControllerReadiness(
            pid_file=pid_file,
            pid_file_exists=exists,
            pid=pid,
            pid_alive=alive,
            ready=bool(ready),
            reason=reason,
        )

    def wait_until_controller_ready(self, *, timeout_s: float) -> ControllerReadiness:
        """只等待 controller（PID 文件），不等待 MPS server 出现。"""

        deadline = time.monotonic() + timeout_s
        evidence = self.controller_ready_evidence()
        while time.monotonic() < deadline:
            if evidence.ready:
                self.readiness = evidence
                return evidence
            time.sleep(min(0.2, max(0.01, timeout_s / 20.0)))
            evidence = self.controller_ready_evidence()
        self.readiness = evidence
        raise MpsRuntimeError(
            "MPS control daemon 在 "
            f"{timeout_s:.1f}s 内没有写出可用的 {CONTROLLER_PID_FILENAME} "
            f"(pid_file={evidence.pid_file}, reason={evidence.reason})"
        )

    def query_servers(self) -> tuple[str, ...]:
        """``get_server_list``；server 未启动时返回空元组（正常状态）。

        查询失败（超时/OS 错误/``returncode=None``/非零）**不会**折叠成空列表：
        失败原因写入 ``last_server_query_error``，供清理阶段判定。
        """

        result = self._run("get_server_list")
        failure = command_failure_reason(result, label="get_server_list")
        self.last_server_query_error = failure
        servers = () if failure else parse_server_lines(result.stdout)
        if servers:
            self.server_observed = True
        # 持久记录实际 server PID 与进程身份，供 controller 退出后核对。
        for server_pid in parse_server_pids(servers):
            self.observed_server_identities[server_pid] = process_identity(server_pid)
        self.health_checks.append(
            MpsDaemonHealth(
                server_count=len(servers),
                client_count=0,
                servers=servers,
                clients=(),
                query=result,
            )
        )
        return servers

    def query_clients(self) -> tuple[int, ...]:
        """对每个 server 执行 ``get_client_list``，返回观察到的客户端 PID。"""

        clients: list[int] = []
        for server in self.query_servers():
            result = self._run(f"get_client_list {server}")
            pids = parse_pid_lines(result.stdout) if result.returncode == 0 else ()
            clients.extend(pids)
            self.health_checks.append(
                MpsDaemonHealth(
                    server_count=1,
                    client_count=len(pids),
                    servers=(server,),
                    clients=pids,
                    query=result,
                )
            )
        return tuple(clients)

    def health_check(self) -> MpsDaemonHealth:
        servers = self.query_servers()
        clients = self.query_clients() if servers else ()
        last = self.health_checks[-1] if self.health_checks else None
        health = MpsDaemonHealth(
            server_count=len(servers),
            client_count=len(clients),
            servers=servers,
            clients=clients,
            query=last.query if last else None,
        )
        self.health_checks.append(health)
        return health

    def process_snapshot(self) -> CommandResult:
        """``ps``：列出 server 与 client 的只读快照。"""

        return self._run("ps")

    def start_server(self) -> CommandResult:
        """显式拉起 MPS server（可选）。

        说明：默认依赖首个 CUDA client 按需启动 server；只有在需要把
        "server 创建" 与 "client 连接" 解耦排查时才显式调用。该命令同样通过
        stdin 发送，并使用当前进程 uid 保持同一用户语义（MPS 拒绝跨用户 server）。
        """

        if not hasattr(os, "getuid"):
            raise MpsRuntimeError("当前平台不提供 getuid，无法显式 start_server")
        return self._run(f"start_server -uid {os.getuid()}")

    def poll_until_client_observed(
        self,
        pid: int,
        *,
        timeout_s: float,
        interval_s: float = 0.5,
    ) -> bool:
        """在超时窗口内轮询，直到目标 PID 出现在 client list 中。"""

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if int(pid) in self.query_clients():
                return True
            time.sleep(max(0.01, interval_s))
        return False

    def snapshot_health(self) -> dict[str, Any]:
        """把最近一次健康检查序列化，供事件属性使用。"""

        if not self.health_checks:
            return {}
        return self.health_checks[-1].to_dict()
