"""把客户端探针输出转换为逐配置结构化记录。

判定口径：

- ``MPS_CAP`` 的 ``verified`` 只表示“配置确实生效且客户端证据自洽”，
  **不**表示独占 SM 分区，也不表示 SM 数已被实测；
- ``effective_sm_count`` 只接受客户端 CUDA context 的实际查询结果；
- 物理 SM 总数只能来自不受控 CUDA 基线（``physical_total_sms`` 入参），
  MPS 客户端的 ``multi_processor_count`` 只是 ``context_visible_sm_count``，
  可能被 active-thread percentage 限制，绝不能覆盖物理总量；
- 客户端 PID 未出现在 MPS client list、workload 未通过、context 不是在配置
  注入之后创建，都会导致 ``verified=False`` 与明确的 ``failure_reason``。
"""

from __future__ import annotations

from typing import Any, Mapping

from .sm_partition_rules import GpuPartitionCapability, GpuPartitionProbeResult
from .sm_profiles import GpuComputeBackend, backend_capability, estimate_requested_sm_count
from .sm_share_profile import GpuShareProfile, unsupported_profile
from .sm_probe_env import utc_now_iso

BASELINE_PHASE_ID = "baseline_unmanaged"


def configuration_applied(payload: dict[str, Any] | None) -> bool:
    if not payload:
        return False
    environment = payload.get("environment") or {}
    return bool(environment.get("configuration_applied"))


def workload_passed(payload: dict[str, Any] | None) -> bool:
    if not payload:
        return False
    workload = payload.get("workload") or {}
    return bool(workload.get("passed"))


def build_mps_cap_result(
    *,
    phase_id: str,
    percentage: int,
    client: Mapping[str, Any],
    daemon_snapshot: Mapping[str, Any],
    cleanup_succeeded: bool | None,
    sm_estimate_rounding: str,
    physical_total_sms: int | None,
) -> GpuPartitionProbeResult:
    """把一次 MPS_CAP 客户端运行转换为结构化探测记录。

    ``physical_total_sms`` 必须来自不受控 CUDA 基线；MPS 客户端自己看到的
    ``context_visible_sm_count`` 只作核对，不参与物理总量与请求值计算。
    """

    payload = client.get("payload")
    sm_measurement = (payload or {}).get("sm_measurement") or {}
    gpu = (payload or {}).get("gpu") or {}
    cuda = (payload or {}).get("cuda") or {}
    context_visible_sm_count = sm_measurement.get("context_visible_sm_count")
    if context_visible_sm_count is None:
        context_visible_sm_count = gpu.get("context_visible_sm_count")
    total_physical_sms = physical_total_sms
    effective_sm_count = sm_measurement.get("effective_sm_count")
    context_created = cuda.get("context_created_after_configuration")

    reasons: list[str] = []
    if client.get("json_parse_error"):
        reasons.append(f"client_json_parse_error:{client['json_parse_error']}")
    if client.get("timed_out"):
        reasons.append("client_timeout")
    if payload and payload.get("status") != "ok":
        reasons.append(f"client_status:{payload.get('failure_reason')}")
    if not client.get("mps_client_pid_observed"):
        reasons.append("mps_client_pid_not_observed_in_get_client_list")
    if not workload_passed(payload):
        reasons.append("cuda_workload_not_passed")
    if context_created is not True:
        reasons.append("cuda_context_not_created_after_configuration")
    if effective_sm_count is None:
        reasons.append("effective_sm_count_not_queryable_from_client_context")
    if (
        context_visible_sm_count is not None
        and total_physical_sms is not None
        and int(context_visible_sm_count) > int(total_physical_sms)
    ):
        reasons.append("context_visible_sm_count_exceeds_physical_total_sms")
    if (
        effective_sm_count is not None
        and total_physical_sms is not None
        and int(effective_sm_count) > int(total_physical_sms)
    ):
        # 越界值不可信，不能写进 profile；记为未测量并给出原因。
        reasons.append("effective_sm_count_exceeds_physical_total_sms")
        effective_sm_count = None
    failure_reason = "; ".join(reasons) if reasons else None

    profile = GpuShareProfile(
        phase_id=phase_id,
        backend=GpuComputeBackend.MPS_CAP,
        capability=backend_capability(GpuComputeBackend.MPS_CAP),
        requested_percentage=percentage,
        requested_sm_count=estimate_requested_sm_count(
            total_physical_sms, percentage, rounding=sm_estimate_rounding
        ),
        effective_sm_count=effective_sm_count,
        total_physical_sms=total_physical_sms,
        cuda_context_created_after_configuration=context_created,
        verification_method=(
            "mps_client_subprocess:cuda_driver_cuCtxGetExecAffinity"
            if effective_sm_count is not None
            else "mps_client_subprocess:configuration_and_workload_only"
        ),
        verified=failure_reason is None,
        failure_reason=failure_reason,
    )
    capability = GpuPartitionCapability(
        backend=GpuComputeBackend.MPS_CAP,
        capability=backend_capability(GpuComputeBackend.MPS_CAP),
        backend_available=bool(daemon_snapshot),
        total_physical_sms=total_physical_sms,
        requested_percentage=percentage,
        effective_sm_count=effective_sm_count,
        dynamic_reconfiguration_supported=False,
        cuda_context_created_after_configuration=context_created,
        verification_method=profile.verification_method,
        failure_reason=(
            None
            if effective_sm_count is not None
            else "cuCtxGetExecAffinity_not_available_or_zero"
        ),
        details={
            "daemon": dict(daemon_snapshot),
            "client_returncode": client.get("returncode"),
            "mps_client_pid_observed": client.get("mps_client_pid_observed"),
            "mps_client_pids_observed": list(
                client.get("mps_client_pids_observed") or ()
            ),
            "context_sm_count_unavailable_reason": sm_measurement.get(
                "context_sm_count_unavailable_reason"
            ),
            "context_visible_sm_count": context_visible_sm_count,
            "physical_total_sms": total_physical_sms,
        },
    )
    return GpuPartitionProbeResult(
        phase_id=phase_id,
        profile=profile,
        capability=capability,
        client_returncode=client.get("returncode"),
        client_stdout_path=client.get("stdout_path"),
        client_stderr_path=client.get("stderr_path"),
        daemon=dict(daemon_snapshot),
        cleanup_succeeded=cleanup_succeeded,
        fallback_detected=not configuration_applied(payload),
        failure=failure_reason,
        wall_time_iso=utc_now_iso(),
    )


def build_baseline_result(
    *,
    client: Mapping[str, Any],
) -> GpuPartitionProbeResult:
    """普通 CUDA 不受控共享基线；它不是受控份额，只作为对照。

    基线没有 MPS cap，因此它的 ``context_visible_sm_count`` 就是设备物理 SM
    总数，并作为后续所有受控配置的 ``physical_total_sms`` 基准。
    """

    payload = client.get("payload")
    gpu = (payload or {}).get("gpu") or {}
    cuda = (payload or {}).get("cuda") or {}
    sm_measurement = (payload or {}).get("sm_measurement") or {}
    total_physical_sms = gpu.get("context_visible_sm_count")
    if total_physical_sms is None:
        total_physical_sms = sm_measurement.get("context_visible_sm_count")

    reasons: list[str] = []
    if client.get("json_parse_error"):
        reasons.append(f"client_json_parse_error:{client['json_parse_error']}")
    if payload and payload.get("status") != "ok":
        reasons.append(f"client_status:{payload.get('failure_reason')}")
    if not workload_passed(payload):
        reasons.append("cuda_workload_not_passed")
    failure_reason = "; ".join(reasons) if reasons else None

    profile = GpuShareProfile(
        phase_id=BASELINE_PHASE_ID,
        backend=GpuComputeBackend.UNMANAGED,
        capability=backend_capability(GpuComputeBackend.UNMANAGED),
        requested_sm_count=total_physical_sms,
        effective_sm_count=sm_measurement.get("effective_sm_count"),
        total_physical_sms=total_physical_sms,
        cuda_context_created_after_configuration=cuda.get(
            "context_created_after_configuration"
        ),
        verification_method="unmanaged_cuda_baseline_client_subprocess",
        verified=failure_reason is None,
        failure_reason=failure_reason,
    )
    capability = GpuPartitionCapability(
        backend=GpuComputeBackend.UNMANAGED,
        capability=backend_capability(GpuComputeBackend.UNMANAGED),
        backend_available=payload is not None,
        total_physical_sms=total_physical_sms,
        effective_sm_count=sm_measurement.get("effective_sm_count"),
        cuda_context_created_after_configuration=cuda.get(
            "context_created_after_configuration"
        ),
        verification_method="unmanaged_cuda_baseline_client_subprocess",
        failure_reason=failure_reason,
        details={
            "client_returncode": client.get("returncode"),
            "context_visible_sm_count": total_physical_sms,
            "note": (
                "不受控共享基线，不是受控计算份额；其 context_visible_sm_count "
                "即物理 SM 总数"
            ),
        },
    )
    return GpuPartitionProbeResult(
        phase_id=BASELINE_PHASE_ID,
        profile=profile,
        capability=capability,
        client_returncode=client.get("returncode"),
        client_stdout_path=client.get("stdout_path"),
        client_stderr_path=client.get("stderr_path"),
        failure=failure_reason,
        wall_time_iso=utc_now_iso(),
    )


def build_unsupported_result(
    *,
    phase_id: str,
    percentage: int,
    failure_reason: str,
    total_physical_sms: int | None,
    binary: Mapping[str, Any] | None = None,
) -> GpuPartitionProbeResult:
    """后端不可用或 daemon 启动失败时的 fail-closed 记录。"""

    profile = unsupported_profile(
        phase_id,
        failure_reason,
        requested_percentage=percentage,
        total_physical_sms=total_physical_sms,
    )
    capability = GpuPartitionCapability(
        backend=GpuComputeBackend.UNSUPPORTED,
        capability=backend_capability(GpuComputeBackend.UNSUPPORTED),
        backend_available=False,
        total_physical_sms=total_physical_sms,
        requested_percentage=percentage,
        verification_method="unsupported_no_measurement",
        failure_reason=failure_reason,
    )
    return GpuPartitionProbeResult(
        phase_id=phase_id,
        profile=profile,
        capability=capability,
        daemon={"binary": dict(binary or {})},
        fallback_detected=False,
        failure=failure_reason,
        wall_time_iso=utc_now_iso(),
    )
