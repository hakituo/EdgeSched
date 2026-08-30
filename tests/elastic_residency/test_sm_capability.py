"""SM-A 判定、汇总与客户端进程契约的单元测试（不依赖 MPS）。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from experiments.elastic_residency.sm.sm_partition_rules import (
    GpuPartitionProbeResult,
)
from experiments.elastic_residency.sm.sm_probe_records import (
    build_baseline_result,
    build_mps_cap_result,
    build_unsupported_result,
)
from experiments.elastic_residency.sm.sm_probe_verdict import (
    build_go_conditions,
    build_sm_evidence,
    build_summary,
    distinct_effective_sm_counts,
)
from tests.elastic_residency.sm_payload_builders import (
    PHYSICAL_TOTAL_SMS,
    client_payload,
    client_run,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _static_results(counts: list[int]) -> list[GpuPartitionProbeResult]:
    results = [
        build_baseline_result(client=client_run(client_payload(), mps_pid_observed=False))
    ]
    for index, count in enumerate(counts):
        payload = client_payload(
            effective_sm_count=count,
            requested_percentage=count * 2,
            context_visible_sm_count=count,
        )
        results.append(
            build_mps_cap_result(
                phase_id=f"mps_cap_{index}",
                percentage=count * 2,
                client=client_run(payload),
                daemon_snapshot={},
                cleanup_succeeded=True,
                sm_estimate_rounding="ceil",
                physical_total_sms=PHYSICAL_TOTAL_SMS,
            )
        )
    return results


def _summary(results: list[GpuPartitionProbeResult], **overrides: Any) -> dict:
    params: dict[str, Any] = dict(
        run_id="sm_a_test",
        experiment="elastic_residency_sm_capability",
        percentages=[50, 25],
        results=results,
        baseline_payload=client_payload(),
        baseline_failed=False,
        baseline_failure=None,
        gpu_environment={
            "telemetry_backend": "direct_nvml",
            "gpu_cli_subprocess_used": False,
            "device_uuid": "GPU-abc",
            "driver_version": "581.29",
        },
        binary={"available": True},
        binary_available=True,
        binary_reason="found_on_path",
        daemon_start_failures=[],
        cleanup_failures=[],
        fallback_count=0,
        green_context={"supported": False},
        git_provenance_complete=True,
    )
    params.update(overrides)
    return build_summary(**params)


def test_go_requires_two_distinct_measured_sm_counts() -> None:
    single = _static_results([24])
    assert distinct_effective_sm_counts(single) == {24}
    conditions = build_go_conditions(single, fallback_count=0, cleanup_failures=[])
    assert conditions["distinct_effective_sm_counts_at_least_two"] is False
    assert conditions["mps_client_success"] is True

    two = _static_results([24, 12])
    assert distinct_effective_sm_counts(two) == {24, 12}
    conditions = build_go_conditions(two, fallback_count=0, cleanup_failures=[])
    assert conditions["distinct_effective_sm_counts_at_least_two"] is True
    assert conditions["no_unsupported_configurations"] is True


def test_unsupported_configurations_do_not_pretend_to_have_run() -> None:
    results = [
        build_baseline_result(
            client=client_run(client_payload(), mps_pid_observed=False)
        ),
        build_unsupported_result(
            phase_id="mps_cap_50",
            percentage=50,
            failure_reason="mps_binary_unavailable:binary_not_found_on_path",
            total_physical_sms=PHYSICAL_TOTAL_SMS,
        ),
    ]
    conditions = build_go_conditions(results, fallback_count=0, cleanup_failures=[])
    assert conditions["mps_client_success"] is False
    assert conditions["no_unsupported_configurations"] is False
    # 没有真实运行的配置时，“是否跑通”不能因空集合而显示为真。
    assert conditions["cuda_workload_succeeded"] is True
    assert conditions["cuda_context_created_after_configuration"] is True


def test_fallback_or_cleanup_failure_forces_no_go() -> None:
    results = _static_results([24, 12])
    assert (
        build_go_conditions(results, fallback_count=1, cleanup_failures=[])[
            "no_fallback"
        ]
        is False
    )
    assert (
        build_go_conditions(
            results, fallback_count=0, cleanup_failures=["mps_cap_0: boom"]
        )["daemon_and_temp_cleanup_succeeded"]
        is False
    )


def test_summary_records_manifest_evidence_and_next_action() -> None:
    summary = _summary(_static_results([24, 12]))
    assert summary["verdict"] == "GO"
    assert summary["report_complete"] is True
    assert summary["gpu_uuid"] == "GPU-abc"
    assert summary["driver_version"] == "581.29"
    assert summary["acceptance_checks"]["gpu_telemetry_is_direct_nvml"] is True
    assert summary["exclusive_sm_budget"]["within_budget"] is True
    evidence = summary["sm_evidence"][-1]
    for field in (
        "phase_id",
        "backend",
        "requested_percentage",
        "physical_total_sms",
        "requested_sm_count",
        "effective_sm_count",
        "context_visible_sm_count",
        "sm_count_verified",
        "verification_method",
        "cuda_context_created_after_configuration",
        "mps_client_pid_observed",
        "cleanup_succeeded",
    ):
        assert field in evidence
    assert len(build_sm_evidence(_static_results([24, 12]))) == 3


def test_summary_keeps_physical_total_from_baseline_not_mps_client() -> None:
    """汇总中的物理 SM 必须来自基线，不被 MPS 客户端受限属性覆盖。"""

    results = _static_results([24, 12])
    mps_evidence = [
        item for item in build_sm_evidence(results) if item["backend"] == "MPS_CAP"
    ]
    assert mps_evidence
    for item in mps_evidence:
        assert item["physical_total_sms"] == PHYSICAL_TOTAL_SMS
        assert item["context_visible_sm_count"] in (24, 12)
    summary = _summary(results)
    assert summary["physical_total_sms"] == PHYSICAL_TOTAL_SMS
    assert summary["physical_total_sms_source"] == (
        "unmanaged_cuda_baseline_context_visible_sm_count"
    )


def test_summary_marks_no_go_with_unsupported_reasons() -> None:
    summary = _summary(
        _static_results([24]),
        percentages=[50],
        binary={"available": False},
        binary_available=False,
        binary_reason="binary_not_found_on_path",
    )
    assert summary["verdict"] == "NO_GO"
    assert any(
        "mps_binary_unavailable" in reason
        for reason in summary["unsupported_reasons"]
    )
    assert "停止 SM-B" in summary["next_action"]


def test_summary_reports_incomplete_report_when_baseline_fails() -> None:
    summary = _summary(
        _static_results([24]),
        baseline_failed=True,
        baseline_failure="cuda_workload_not_passed",
    )
    assert summary["verdict"] == "NO_GO"
    assert summary["report_complete"] is False


def test_client_probe_rejects_missing_environment_before_cuda_init() -> None:
    """环境变量缺失必须在 CUDA 初始化之前被拒绝，且输出单个 JSON。"""

    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("CUDA_MPS_")
    }
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.elastic_residency.sm.sm_client_probe",
            "--phase-id",
            "missing_env",
            "--require-env",
            "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE",
        ],
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        shell=False,
    )
    assert completed.returncode == 1
    payload = json.loads(completed.stdout)
    assert payload["environment"]["captured_before_cuda_init"] is True
    assert payload["failure_reason"].startswith(
        "required_environment_missing_before_cuda_init"
    )
    assert payload["sm_measurement"]["effective_sm_count"] is None
    assert payload["cuda"]["initialized_at_import"] is None
