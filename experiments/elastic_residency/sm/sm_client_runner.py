"""SM-A 的客户端编排：基线运行与逐 percentage 受控配置运行。

进程执行细节位于 ``sm_client_process``，记录构造位于 ``sm_probe_records``。
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Sequence

from ..contracts import ExperimentArtifactWriter, FailureRecord
from .mps_commands import CLIENT_ENV_KEYS, MpsRuntimeError, unmanaged_environment
from .mps_runtime import MpsRuntime
from .sm_client_process import run_client_probe
from .sm_partition_rules import GpuPartitionProbeResult
from .sm_probe_records import (
    BASELINE_PHASE_ID,
    build_baseline_result,
    build_mps_cap_result,
    build_unsupported_result,
)
from .sm_profiles import GpuComputeBackend

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def config_phase_id(percentage: int) -> str:
    return f"mps_cap_{percentage}"


def run_baseline(
    args: argparse.Namespace,
    writer: ExperimentArtifactWriter,
) -> tuple[GpuPartitionProbeResult, dict[str, Any]]:
    """普通 CUDA 不受控共享基线；必须显式移除 MPS 环境变量。"""

    client = run_client_probe(
        artifact_root=writer.layout.root,
        project_root=PROJECT_ROOT,
        phase_id=BASELINE_PHASE_ID,
        env=unmanaged_environment(),
        expected_backend=GpuComputeBackend.UNMANAGED,
        expected_percentage=None,
        require_env=(),
        python_executable=args.python_executable,
        timeout_s=args.client_timeout,
        hold_seconds=0.0,
        matrix_size=args.matrix_size,
        iterations=args.iterations,
    )
    return build_baseline_result(client=client), client


def run_mps_configurations(
    args: argparse.Namespace,
    writer: ExperimentArtifactWriter,
    runtime: MpsRuntime,
    percentages: Sequence[int],
    physical_total_sms: int | None,
) -> tuple[list[GpuPartitionProbeResult], int, list[str], list[str]]:
    """逐 percentage 启动独立 daemon 与独立客户端子进程。

    顺序：启动 control daemon（只等 controller 就绪）→ 启动客户端并保持
    context 存活 → 期间查询 server/client → 客户端退出 → stdin 发送 quit →
    验证 controller/server 消失并清理私有目录。

    ``physical_total_sms`` 必须来自不受控基线，不用 MPS 客户端的受限属性。

    返回：结果列表、fallback 次数、daemon 启动失败列表、清理失败列表。
    """

    binary_info = runtime.detect()
    results: list[GpuPartitionProbeResult] = []
    fallback_count = 0
    daemon_start_failures: list[str] = []
    cleanup_failures: list[str] = []
    for percentage in percentages:
        phase_id = config_phase_id(percentage)
        if not binary_info.available:
            unsupported = build_unsupported_result(
                phase_id=phase_id,
                percentage=percentage,
                failure_reason=f"mps_binary_unavailable:{binary_info.reason}",
                total_physical_sms=physical_total_sms,
                binary=binary_info.to_dict(),
            )
            results.append(unsupported)
            writer.append_iteration(unsupported.to_dict())
            continue
        handle = None
        result: GpuPartitionProbeResult | None = None
        try:
            handle = runtime.start(label=phase_id, run_id=args.run_id)
            if getattr(args, "mps_start_server", False):
                # 可选：显式经 stdin 拉起 server（同一 uid），默认由首个 client 按需拉起。
                handle.start_server()
            client = run_client_probe(
                artifact_root=writer.layout.root,
                project_root=PROJECT_ROOT,
                phase_id=phase_id,
                env=handle.client_environment(percentage),
                expected_backend=GpuComputeBackend.MPS_CAP,
                expected_percentage=percentage,
                require_env=CLIENT_ENV_KEYS,
                python_executable=args.python_executable,
                timeout_s=args.client_timeout,
                hold_seconds=args.hold_seconds,
                matrix_size=args.matrix_size,
                iterations=args.iterations,
                daemon_handle=handle,
                poll_interval_s=args.poll_interval,
            )
            result = build_mps_cap_result(
                phase_id=phase_id,
                percentage=percentage,
                client=client,
                daemon_snapshot=handle.snapshot(),
                cleanup_succeeded=None,
                sm_estimate_rounding=args.sm_estimate_rounding,
                physical_total_sms=physical_total_sms,
            )
            results.append(result)
            if result.fallback_detected:
                fallback_count += 1
            if result.failure:
                writer.append_failure(
                    FailureRecord(
                        stage=phase_id,
                        error_type="MpsCapProbeFailed",
                        message=str(result.failure),
                        monotonic_ns=time.monotonic_ns(),
                        attributes={
                            "percentage": percentage,
                            "returncode": client.get("returncode"),
                        },
                    )
                )
        except MpsRuntimeError as exc:
            daemon_start_failures.append(f"{phase_id}: {exc}")
            result = build_unsupported_result(
                phase_id=phase_id,
                percentage=percentage,
                failure_reason=f"mps_daemon_start_failed:{exc}",
                total_physical_sms=physical_total_sms,
                binary=binary_info.to_dict(),
            )
            results.append(result)
            writer.append_iteration(result.to_dict())
            writer.append_failure(
                FailureRecord(
                    stage=phase_id,
                    error_type="MpsDaemonStartFailed",
                    message=str(exc),
                    monotonic_ns=time.monotonic_ns(),
                    attributes={"percentage": percentage},
                )
            )
        finally:
            if handle is not None:
                cleanup_succeeded = handle.cleanup(
                    archive_dir=writer.layout.root / "mps_logs"
                )
                residual = handle.residual_artifacts()
                if residual:
                    cleanup_succeeded = False
                snapshot = handle.snapshot()
                snapshot["residual_artifacts"] = residual
                if result is not None:
                    object.__setattr__(result, "cleanup_succeeded", cleanup_succeeded)
                    object.__setattr__(result, "daemon", snapshot)
                    writer.append_iteration(result.to_dict())
                if not cleanup_succeeded:
                    message = handle.cleanup_error or f"residual={residual}"
                    cleanup_failures.append(f"{phase_id}: {message}")
                    writer.append_failure(
                        FailureRecord(
                            stage=f"{phase_id}_cleanup",
                            error_type="MpsCleanupFailed",
                            message=str(message),
                            monotonic_ns=time.monotonic_ns(),
                            attributes={
                                "private_root": str(handle.root),
                                "residual_artifacts": residual,
                                "post_quit_evidence": dict(handle.post_quit_evidence),
                            },
                        )
                    )
    return results, fallback_count, daemon_start_failures, cleanup_failures
