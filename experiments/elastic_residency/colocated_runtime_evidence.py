"""LLM+TTS 共置容量测量的 GPU 新鲜度与 swap 增长门禁。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from experiments.elastic_residency.forge_runtime_evidence import (
    evaluate_gpu_sample_freshness,
    evaluate_swap_growth,
)
from experiments.elastic_residency.tts_runtime_evidence import assess_gpu_snapshots


def assess_colocated_runtime(
    samples: Sequence[Any],
    *,
    start_ns: int,
    end_ns: int,
    capacity_snapshots: Sequence[dict[str, Any]],
    max_gpu_sample_age_ms: float,
    max_swap_growth_bytes: int,
) -> dict[str, Any]:
    """校验所有参与 ``B_LLM/P_colocated`` 计算的运行期证据。"""

    gpu_periodic = evaluate_gpu_sample_freshness(
        samples,
        start_ns=start_ns,
        end_ns=end_ns,
        max_sample_age_ms=max_gpu_sample_age_ms,
    )
    gpu_snapshots = assess_gpu_snapshots(
        capacity_snapshots,
        max_freshness_bound_ms=max_gpu_sample_age_ms,
    )
    swap_values = [int(sample.system_swap_used_bytes) for sample in samples]
    swap = dict(evaluate_swap_growth(swap_values, max_swap_growth_bytes))
    swap.update(
        sample_count=len(swap_values),
        end_bytes=swap_values[-1] if swap_values else None,
        tolerance_bytes=int(max_swap_growth_bytes),
    )
    checks = {
        "gpu_periodic_samples_fresh": gpu_periodic["within_limit"] is True,
        "gpu_capacity_snapshots_fresh": (
            gpu_snapshots["all_freshness_bounds_valid"] is True
        ),
        "swap_growth_within_tolerance": (
            bool(swap_values) and swap["within_tolerance"] is True
        ),
    }
    return {
        "checks": checks,
        "gpu_periodic": gpu_periodic,
        "gpu_capacity_snapshots": gpu_snapshots,
        "swap": swap,
    }
