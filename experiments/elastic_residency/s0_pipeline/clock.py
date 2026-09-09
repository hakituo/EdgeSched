"""时钟分辨率测量。

Windows 的 ``time.monotonic_ns()`` 粒度约为 15.6 ms，短于一个 tick 的执行
区间会被测成 0；Linux/WSL2 通常为亚微秒。不记录分辨率，时间线里的 0 值
无法与「真的零耗时」区分，因此每次运行都必须把它写进事件与结果。
"""

from __future__ import annotations

import time

_CACHED_RESOLUTION_NS: int | None = None


def measure_clock_resolution_ns(*, max_samples: int = 2000) -> int:
    """连续采样 ``monotonic_ns``，返回观测到的最小非零步进。"""

    global _CACHED_RESOLUTION_NS
    if _CACHED_RESOLUTION_NS is not None:
        return _CACHED_RESOLUTION_NS
    last = time.monotonic_ns()
    minimum = 0
    for _ in range(max(1, max_samples)):
        current = time.monotonic_ns()
        delta = current - last
        if delta > 0:
            minimum = delta if minimum == 0 else min(minimum, delta)
            if minimum <= 1000:
                break
        last = current
    _CACHED_RESOLUTION_NS = minimum
    return minimum


def reset_clock_resolution_cache() -> None:
    """仅供测试重置缓存。"""

    global _CACHED_RESOLUTION_NS
    _CACHED_RESOLUTION_NS = None
