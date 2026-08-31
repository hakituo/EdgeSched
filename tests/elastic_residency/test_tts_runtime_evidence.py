from copy import deepcopy

from experiments.elastic_residency.tts_runtime_evidence import (
    assess_gpu_snapshots,
    compact_runtime_evidence,
    evaluate_swap_timeline,
    finalize_child_runtime_evidence,
)


def _snapshot(bound_ms: float = 2.0) -> dict:
    return {
        "gpu_telemetry_backend": "direct_nvml",
        "gpu_cli_subprocess_used": False,
        "gpu_sample_freshness_bound_ms": bound_ms,
    }


def _child() -> dict:
    phase = _snapshot()
    record = {
        "passed": True,
        "acceptance_checks": {"existing_check": True},
        "memory_talker_peak": deepcopy(phase),
        "memory_transfer_peak": deepcopy(phase),
        "memory_codec_peak": deepcopy(phase),
    }
    return {
        "passed": True,
        "memory_baseline": deepcopy(phase),
        "memory_ready": deepcopy(phase),
        "warmup": {
            "memory_talker_peak": deepcopy(phase),
            "memory_transfer_peak": deepcopy(phase),
            "memory_codec_peak": deepcopy(phase),
        },
        "records": [record],
    }


def test_swap_timeline_reports_peak_growth_and_gaps() -> None:
    report = evaluate_swap_timeline(
        [
            {"monotonic_ns": 1_000_000_000, "used_bytes": 1024},
            {"monotonic_ns": 1_050_000_000, "used_bytes": 4096},
            {"monotonic_ns": 1_100_000_000, "used_bytes": 2048},
        ],
        interval_s=0.05,
        tolerance_bytes=4096,
    )
    assert report["start_bytes"] == 1024
    assert report["max_bytes"] == 4096
    assert report["growth_bytes"] == 3072
    assert report["within_tolerance"] is True
    assert report["actual_gap_max_ms"] == 50.0


def test_gpu_snapshot_freshness_uses_synchronous_query_bound() -> None:
    good = assess_gpu_snapshots([_snapshot(10.0)], max_freshness_bound_ms=750.0)
    stale = assess_gpu_snapshots([_snapshot(751.0)], max_freshness_bound_ms=750.0)
    assert good["all_direct_nvml_without_cli"] is True
    assert good["all_freshness_bounds_valid"] is True
    assert stale["all_freshness_bounds_valid"] is False


def test_finalize_child_evidence_updates_iterations_and_summary_view() -> None:
    child = finalize_child_runtime_evidence(
        _child(),
        swap_report={
            "samples": [{"monotonic_ns": 1, "used_bytes": 0}],
            "sample_count": 1,
            "start_bytes": 0,
            "max_bytes": 0,
            "growth_bytes": 0,
            "within_tolerance": True,
        },
        max_gpu_freshness_ms=750.0,
    )
    record = child["records"][0]
    assert child["passed"] is True
    assert record["acceptance_checks"]["gpu_samples_fresh"] is True
    assert record["acceptance_checks"]["swap_growth_within_tolerance"] is True
    assert record["swap_growth_bytes"] == 0
    compact = compact_runtime_evidence(child)
    assert "samples" not in compact["swap"]


def test_swap_failure_fails_child_and_record() -> None:
    child = finalize_child_runtime_evidence(
        _child(),
        swap_report={
            "samples": [],
            "sample_count": 2,
            "start_bytes": 0,
            "max_bytes": 4096,
            "growth_bytes": 4096,
            "within_tolerance": False,
        },
        max_gpu_freshness_ms=750.0,
    )
    assert child["passed"] is False
    assert child["records"][0]["passed"] is False
