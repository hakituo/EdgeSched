"""SM-A 的 Go / No-Go 判定与汇总结构。

Go 条件（来自论文指南 SM-A）：

- MPS client 成功运行；
- 至少两个不同请求得到可验证、合理不同的实际 SM 数；
- context 创建顺序正确；
- CUDA workload 成功并同步；
- 无 fallback；
- daemon 与临时目录清理成功。

No-Go 不是代码失败：仍要产出完整的结构化 unsupported 报告。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .sm_partition_rules import (
    GpuPartitionProbeResult,
    validate_exclusive_sm_budget,
)
from .sm_probe_records import BASELINE_PHASE_ID, workload_passed
from .sm_profiles import GpuComputeBackend

NEXT_ACTION_GO = "可以进入 SM-B（受控 SM 隔离曲线）"
NEXT_ACTION_NO_GO = (
    "SM-A No-Go：保留 UNSUPPORTED 报告，停止 SM-B，不要修改生产调度器，"
    "也不要自动转写 Green Context"
)


def distinct_effective_sm_counts(
    results: Sequence[GpuPartitionProbeResult],
) -> set[int]:
    """统计真正来自客户端 context 实测的有效 SM 数。"""

    return {
        result.profile.effective_sm_count
        for result in results
        if result.profile.effective_sm_count is not None
    }


def build_go_conditions(
    results: Sequence[GpuPartitionProbeResult],
    *,
    fallback_count: int,
    cleanup_failures: Sequence[str],
) -> dict[str, bool]:
    mps_results = [
        result for result in results if result.phase_id != BASELINE_PHASE_ID
    ]
    # UNSUPPORTED 配置没有真实运行客户端，不参与“是否跑通”的判断；
    # 它们由 no_unsupported_configurations 单独记录。
    executed = [
        result
        for result in results
        if result.profile.backend is not GpuComputeBackend.UNSUPPORTED
    ]
    return {
        "mps_client_success": bool(mps_results)
        and all(result.failure is None for result in mps_results),
        "no_unsupported_configurations": all(
            result.profile.backend is not GpuComputeBackend.UNSUPPORTED
            for result in mps_results
        ),
        "distinct_effective_sm_counts_at_least_two": (
            len(distinct_effective_sm_counts(results)) >= 2
        ),
        "cuda_context_created_after_configuration": bool(executed)
        and all(
            result.profile.cuda_context_created_after_configuration is True
            for result in executed
        ),
        "cuda_workload_succeeded": bool(executed)
        and all(_workload_ok(result) for result in executed),
        "no_fallback": fallback_count == 0,
        "daemon_and_temp_cleanup_succeeded": not cleanup_failures,
    }


def _workload_ok(result: GpuPartitionProbeResult) -> bool:
    """基线用失败原因为准，受控配置还要求 verified。"""

    if result.phase_id == BASELINE_PHASE_ID:
        return result.failure is None
    return result.failure is None and result.profile.verified


def build_acceptance_checks(
    *,
    go_conditions: Mapping[str, bool],
    gpu_environment: Mapping[str, Any],
    binary_available: bool,
    daemon_start_failures: Sequence[str],
    baseline_failed: bool,
    git_provenance_complete: bool,
    green_context_supported: bool,
) -> dict[str, Any]:
    return {
        **go_conditions,
        "gpu_telemetry_is_direct_nvml": (
            gpu_environment.get("telemetry_backend") == "direct_nvml"
            and gpu_environment.get("gpu_cli_subprocess_used") is False
        ),
        "mps_binary_available": binary_available,
        "mps_daemon_started_without_failure": not daemon_start_failures,
        "baseline_client_succeeded": not baseline_failed,
        "git_provenance_complete": git_provenance_complete,
        "green_context_not_claimed_supported": green_context_supported is False,
    }


def build_unsupported_reasons(
    *,
    baseline_failed: bool,
    baseline_failure: str | None,
    binary_available: bool,
    binary_reason: str | None,
    daemon_start_failures: Sequence[str],
    results: Sequence[GpuPartitionProbeResult],
    fallback_count: int,
) -> list[str]:
    reasons: list[str] = []
    if baseline_failed:
        reasons.append(f"baseline_unmanaged_failed:{baseline_failure}")
    if not binary_available:
        reasons.append(f"mps_binary_unavailable:{binary_reason}")
    reasons.extend(str(item) for item in daemon_start_failures)
    if len(distinct_effective_sm_counts(results)) < 2:
        reasons.append(
            "effective_sm_count_not_verified_for_at_least_two_percentages"
        )
    if fallback_count:
        reasons.append(f"fallback_detected:{fallback_count}")
    return reasons


def build_sm_evidence(
    results: Sequence[GpuPartitionProbeResult],
) -> list[dict[str, Any]]:
    return [
        {
            "phase_id": result.phase_id,
            "backend": result.profile.backend.value,
            "capability": result.profile.capability.value,
            "requested_percentage": result.profile.requested_percentage,
            "physical_total_sms": result.profile.total_physical_sms,
            "requested_sm_count": result.profile.requested_sm_count,
            "effective_sm_count": result.profile.effective_sm_count,
            "context_visible_sm_count": (result.capability.details or {}).get(
                "context_visible_sm_count"
            ),
            "sm_count_verified": result.profile.sm_count_verified,
            "verified": result.profile.verified,
            "verification_method": result.profile.verification_method,
            "cuda_context_created_after_configuration": (
                result.profile.cuda_context_created_after_configuration
            ),
            "mps_client_pid_observed": (result.capability.details or {}).get(
                "mps_client_pid_observed"
            ),
            "cleanup_succeeded": result.cleanup_succeeded,
            "failure": result.failure,
        }
        for result in results
    ]


def build_summary(
    *,
    run_id: str,
    experiment: str,
    percentages: Sequence[int],
    results: Sequence[GpuPartitionProbeResult],
    baseline_payload: Mapping[str, Any],
    baseline_failed: bool,
    baseline_failure: str | None,
    gpu_environment: Mapping[str, Any],
    binary: Mapping[str, Any],
    binary_available: bool,
    binary_reason: str | None,
    daemon_start_failures: Sequence[str],
    cleanup_failures: Sequence[str],
    fallback_count: int,
    green_context: Mapping[str, Any],
    git_provenance_complete: bool,
) -> dict[str, Any]:
    profiles = [result.profile for result in results]
    baseline_gpu = baseline_payload.get("gpu") or {}
    # 物理 SM 总数只来自不受控基线：基线没有 MPS cap，其 context 可见值即物理值。
    total_physical_sms = baseline_gpu.get("context_visible_sm_count")
    go_conditions = build_go_conditions(
        results,
        fallback_count=fallback_count,
        cleanup_failures=cleanup_failures,
    )
    acceptance_checks = build_acceptance_checks(
        go_conditions=go_conditions,
        gpu_environment=gpu_environment,
        binary_available=binary_available,
        daemon_start_failures=daemon_start_failures,
        baseline_failed=baseline_failed,
        git_provenance_complete=git_provenance_complete,
        green_context_supported=bool(green_context.get("supported")),
    )
    verdict = (
        "GO" if (not baseline_failed and all(go_conditions.values())) else "NO_GO"
    )
    return {
        "run_id": run_id,
        "experiment": experiment,
        "report_complete": not baseline_failed,
        "verdict": verdict,
        "go_conditions": go_conditions,
        "acceptance_checks": acceptance_checks,
        "passed": all(acceptance_checks.values()),
        "unsupported_reasons": build_unsupported_reasons(
            baseline_failed=baseline_failed,
            baseline_failure=baseline_failure,
            binary_available=binary_available,
            binary_reason=binary_reason,
            daemon_start_failures=daemon_start_failures,
            results=results,
            fallback_count=fallback_count,
        ),
        "percentages": list(percentages),
        "physical_total_sms": total_physical_sms,
        "total_physical_sms": total_physical_sms,
        "physical_total_sms_source": (
            "unmanaged_cuda_baseline_context_visible_sm_count"
            if total_physical_sms is not None
            else None
        ),
        "gpu_uuid": gpu_environment.get("device_uuid") or baseline_gpu.get("uuid"),
        "compute_capability": baseline_gpu.get("compute_capability"),
        "driver_version": gpu_environment.get("driver_version"),
        "cuda_runtime_version": (
            (baseline_payload.get("cuda") or {}).get("cuda_runtime_version")
        ),
        "torch_version": (baseline_payload.get("cuda") or {}).get("torch_version"),
        "mps_binary": dict(binary),
        "fallback_count": fallback_count,
        "cleanup_failures": list(cleanup_failures),
        "exclusive_sm_budget": validate_exclusive_sm_budget(
            profiles, total_physical_sms
        ),
        "green_context": dict(green_context),
        "sm_evidence": build_sm_evidence(results),
        "next_action": NEXT_ACTION_GO if verdict == "GO" else NEXT_ACTION_NO_GO,
    }
