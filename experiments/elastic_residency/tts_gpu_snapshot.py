"""TTS 实验的同步 direct NVML 显存快照。"""

from __future__ import annotations

import datetime as dt
import time
from typing import Any

from experiments.elastic_residency.gpu_telemetry import (
    DirectNvmlProbe,
    NvmlUnavailableError,
    get_direct_nvml_probe,
)


def nvidia_memory_snapshot(
    pid: int,
    *,
    probe: DirectNvmlProbe | None = None,
) -> dict[str, Any]:
    """直接查询 NVML，并记录查询耗时作为样本年龄的保守上界。"""

    probe_started_ns = time.monotonic_ns()
    try:
        active_probe = probe or get_direct_nvml_probe(include_process_memory=True)
        sampled = active_probe.sample(pid)
    except NvmlUnavailableError as exc:
        sampled = {
            "available": False,
            "backend": "direct_nvml",
            "unavailable_reason": f"direct_nvml_init_failed: {exc}",
        }
    probe_completed_ns = time.monotonic_ns()
    probe_duration_ms = (probe_completed_ns - probe_started_ns) / 1_000_000.0
    probe_timing = {
        "gpu_probe_started_monotonic_ns": probe_started_ns,
        "gpu_probe_completed_monotonic_ns": probe_completed_ns,
        "gpu_probe_duration_ms": probe_duration_ms,
        "gpu_sample_wall_time_iso": dt.datetime.now(dt.timezone.utc).isoformat(),
        "gpu_sample_freshness_bound_ms": probe_duration_ms,
        "gpu_sample_freshness_semantics": (
            "synchronous_direct_nvml_query_duration_upper_bound"
        ),
    }
    if not sampled.get("available", False):
        return {
            "gpu_card_available": False,
            "gpu_card_used_mb": None,
            "gpu_card_total_mb": None,
            "gpu_process_total_mb": None,
            "gpu_process_total_available": False,
            "gpu_process_total_unavailable_reason": sampled.get(
                "unavailable_reason",
                "direct_nvml_query_failed",
            ),
            "gpu_telemetry_backend": "direct_nvml",
            "gpu_cli_subprocess_used": False,
            **probe_timing,
        }
    process_used_mb = sampled.get("process_used_mb")
    return {
        "gpu_card_available": True,
        "gpu_card_used_mb": sampled["card_used_mb"],
        "gpu_card_total_mb": sampled["card_total_mb"],
        "gpu_process_total_mb": process_used_mb,
        "gpu_process_total_available": process_used_mb is not None,
        "gpu_process_total_unavailable_reason": sampled.get(
            "process_memory_unavailable_reason"
        ),
        "gpu_telemetry_backend": sampled.get("backend", "direct_nvml"),
        "gpu_cli_subprocess_used": False,
        **probe_timing,
    }
