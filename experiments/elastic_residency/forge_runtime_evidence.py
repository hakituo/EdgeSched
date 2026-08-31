"""Forge b=1 校准的 swap 与周期 GPU 样本有效性判定。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def evaluate_swap_growth(
    swap_values: Sequence[int],
    tolerance_bytes: int,
) -> Mapping[str, Any]:
    """评估时间线 swap 增长是否在容忍范围内。"""

    if not swap_values:
        return {
            "start_bytes": None,
            "max_bytes": None,
            "growth_bytes": None,
            "within_tolerance": True,
        }
    start = int(swap_values[0])
    maximum = int(max(swap_values))
    growth = max(0, maximum - start)
    return {
        "start_bytes": start,
        "max_bytes": maximum,
        "growth_bytes": growth,
        "within_tolerance": growth <= int(tolerance_bytes),
    }


def evaluate_gpu_sample_freshness(
    samples: Sequence[Any],
    *,
    max_sample_age_ms: float,
    start_ns: int | None = None,
    end_ns: int | None = None,
) -> dict[str, float | int | bool | None]:
    """评估一个参与容量计算的窗口内，周期 GPU 样本是否足够新鲜。"""

    ages: list[float] = []
    for sample in samples:
        if not getattr(sample, "gpu_available", False):
            continue
        timestamp = int(sample.monotonic_ns)
        if start_ns is not None and timestamp < start_ns:
            continue
        if end_ns is not None and timestamp > end_ns:
            continue
        age = getattr(sample, "gpu_sample_age_ms", None)
        if isinstance(age, (int, float)) and age >= 0:
            ages.append(float(age))
    maximum = max(ages) if ages else None
    return {
        "sample_count": len(ages),
        "max_sample_age_ms": maximum,
        "limit_ms": float(max_sample_age_ms),
        "within_limit": maximum is not None and maximum <= max_sample_age_ms,
    }
