"""SM-A WSL 集成硬门的判定逻辑（不是 pytest 收集对象）。

单独成模块，便于用合成数据做纯单元测试，证明硬门真的会拦截缺陷。

判定原则：

- ``sm_evidence[*]["failure"]``：环境能力白名单以外的**任何**非空失败都判失败，
  不只匹配已知 implementation marker；
- 受控记录必须**显式**包含 ``daemon.residual_artifacts`` 且严格等于 ``[]``；
- 受控客户端必须**显式**包含 ``sm_measurement.driver_query``；
- 白名单同时用于聚合原因与逐项 failure，避免把合法的
  ``cuCtxGetExecAffinity`` 不可用误判成实现缺陷。
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

#: 环境能力限制：这些原因只是“当前环境做不到”，不是实现缺陷。
ENVIRONMENT_LIMITATION_MARKERS: tuple[str, ...] = (
    # MPS 控制程序本身不存在。
    "mps_binary_unavailable",
    # cuCtxGetExecAffinity 在本环境查不到 context 可见 SM 数。
    "effective_sm_count_not_queryable_from_client_context",
    # 聚合层：至少两个 percentage 没有可验证的有效 SM 数。
    "effective_sm_count_not_verified_for_at_least_two_percentages",
    # 驱动返回码证据本身。
    "cuCtxGetExecAffinity",
)

BASELINE_PHASE_ID = "baseline_unmanaged"


def is_environment_limitation(text: str) -> bool:
    return any(marker in text for marker in ENVIRONMENT_LIMITATION_MARKERS)


def assert_no_implementation_fault(summary: Mapping[str, Any]) -> None:
    """白名单以外的任何失败原因或逐项 failure 都必须触发 AssertionError。"""

    offenders: list[str] = []
    for reason in summary.get("unsupported_reasons") or ():
        if not is_environment_limitation(str(reason)):
            offenders.append(f"unsupported_reasons: {reason}")
    for item in summary.get("sm_evidence") or ():
        failure = item.get("failure")
        if not failure:
            continue
        if not is_environment_limitation(str(failure)):
            offenders.append(f"{item.get('phase_id')}.failure: {failure}")
    assert not offenders, (
        "MPS binary 存在但出现非环境限制的失败，属于实现问题而不是环境不支持: "
        f"{offenders}"
    )


def assert_cleanup_closed(records: Iterable[Mapping[str, Any]]) -> None:
    """清理未闭合必须失败：字段缺失、非 True、有残迹都不接受。"""

    problems: list[str] = []
    for item in records:
        if item.get("phase_id") == BASELINE_PHASE_ID:
            continue
        phase = item.get("phase_id")
        daemon = item.get("daemon")
        if not daemon:
            problems.append(f"{phase}: 缺少 daemon 快照")
            continue
        if daemon.get("cleanup_succeeded") is not True:
            problems.append(
                f"{phase}: cleanup_succeeded={daemon.get('cleanup_succeeded')}"
            )
        evidence = daemon.get("post_quit_evidence") or {}
        if evidence.get("reached_quiescent") is not True:
            problems.append(f"{phase}: post_quit_evidence={evidence}")
        if "residual_artifacts" not in daemon:
            problems.append(f"{phase}: daemon 缺少 residual_artifacts 字段")
        elif daemon.get("residual_artifacts") != []:
            problems.append(f"{phase}: residual={daemon['residual_artifacts']}")
    assert not problems, f"MPS 清理未闭合或证据字段缺失: {problems}"


def assert_affinity_query_present(client_payloads: Sequence[Mapping[str, Any]]) -> None:
    """受控客户端必须显式携带 driver_query，且字段类型正确。"""

    problems: list[str] = []
    if not client_payloads:
        problems.append("缺少受控客户端 payload")
    for index, payload in enumerate(client_payloads):
        sm_measurement = payload.get("sm_measurement")
        if not isinstance(sm_measurement, Mapping):
            problems.append(f"client[{index}]: 缺少 sm_measurement")
            continue
        driver_query = sm_measurement.get("driver_query")
        if not isinstance(driver_query, Mapping):
            problems.append(f"client[{index}]: 缺少 sm_measurement.driver_query")
            continue
        if driver_query.get("requested_affinity_type") != 1:
            problems.append(
                f"client[{index}]: requested_affinity_type="
                f"{driver_query.get('requested_affinity_type')}"
            )
        return_code = driver_query.get("cu_query_returncode")
        if return_code is not None and not isinstance(return_code, int):
            problems.append(f"client[{index}]: cu_query_returncode={return_code}")
        reason = driver_query.get("context_sm_count_unavailable_reason")
        if reason is not None and not isinstance(reason, str):
            problems.append(f"client[{index}]: unavailable_reason={reason}")
    assert not problems, f"cuCtxGetExecAffinity 查询证据缺失或异常: {problems}"


def assert_controller_ready(records: Iterable[Mapping[str, Any]]) -> None:
    problems: list[str] = []
    for item in records:
        if item.get("phase_id") == BASELINE_PHASE_ID:
            continue
        phase = item.get("phase_id")
        daemon = item.get("daemon") or {}
        readiness = daemon.get("controller_readiness")
        if not readiness:
            problems.append(f"{phase}: 缺少 controller_readiness")
            continue
        if readiness.get("ready") is not True:
            problems.append(f"{phase}: controller 未就绪 {readiness}")
    assert not problems, f"controller 就绪证据缺失: {problems}"
