"""时钟分辨率测量。

Windows 的 ``time.monotonic_ns()`` 粒度约为 15.6 ms，短于一个 tick 的执行
区间会被测成 0；Linux/WSL2 通常为亚微秒。不记录分辨率，时间线里的 0 值
无法与「真的零耗时」区分，因此每次运行都必须把它写进事件与结果。
"""

from __future__ import annotations

import time

_CACHED_RESOLUTION_NS: int | None = None


def measure_clock_resolution_ns(
    *,
    max_samples: int = 1_000_000,
    max_seconds: float = 0.05,
) -> int:
    """连续采样 ``monotonic_ns``，返回观测到的最小非零步进。

    Windows 粒度约 15.6 ms，固定次数的采样可能全部落在同一 tick 内
    而误报 0；因此同时按墙钟时间预算采样，保证至少跨过一个 tick。
    """

    global _CACHED_RESOLUTION_NS
    if _CACHED_RESOLUTION_NS is not None:
        return _CACHED_RESOLUTION_NS
    last = time.monotonic_ns()
    wall_deadline = time.monotonic() + max_seconds
    minimum = 0
    for _ in range(max(1, max_samples)):
        current = time.monotonic_ns()
        delta = current - last
        if delta > 0:
            minimum = delta if minimum == 0 else min(minimum, delta)
            if minimum <= 1000:
                break
        last = current
        if time.monotonic() > wall_deadline and minimum > 0:
            break
    _CACHED_RESOLUTION_NS = minimum
    return minimum


def reset_clock_resolution_cache() -> None:
    """仅供测试重置缓存。"""

    global _CACHED_RESOLUTION_NS
    _CACHED_RESOLUTION_NS = None
