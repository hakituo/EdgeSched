"""播放时钟映射：只输出「经过验证的映射」，不按偏移量大小判定。

PortAudio 规定流时间单调递增但**起点未指定**，因此 device 时间与 host
monotonic 相差很大本身不能证明时钟不可用；真正要验证的是这条映射是否可用：

- **单位一致**：配对采样的 device 跨度 / host 跨度应≈1（混用 wall clock 或
  单位错误会在这里暴露）；
- **残差有界**：多次配对后 ``device - host`` 的离散度应在容差内；
- **漂移有界**：残差随时间的斜率应在容差内；
- **单调**：device 时间必须严格递增。

device 时间取 callback 的 ``outputBufferDacTime``（缓冲首样本的 DAC 输出时刻），
而不是 ``currentTime``（callback 被调用的时刻）——两者语义不同，混用会让
「实际播放时刻」失去意义。两者之差只作观察记录，不作通过判据。
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any

#: 默认容差：残差 5 ms、漂移 1 ms/s、跨度比 ±5%。尚未冻结，仅用于判「不可用」。
DEFAULT_MAX_RESIDUAL_NS = 5_000_000
DEFAULT_MAX_DRIFT_NS_PER_S = 1_000_000
DEFAULT_SPAN_RATIO_RANGE = (0.95, 1.05)
MIN_SAMPLES = 4
#: 漂移只在足够长的观测跨度上才有意义：跨度短于该值时只记录、不判定，
#: 否则 42.7 ms 的缓冲区量化会在 0.3 s 跨度上产生没有意义的斜率。
DEFAULT_MIN_SPAN_FOR_DRIFT_NS = 1_000_000_000


@dataclass(frozen=True)
class ClockCalibration:
    """一次时钟校准的完整证据；``verified`` 为假时不得用于实际播放时刻。"""

    method: str
    sample_count: int
    offset_ns: int
    residual_max_ns: int
    residual_mean_abs_ns: int
    drift_ns_per_s: float
    drift_evaluated: bool
    sample_cost_ns: int
    span_ratio: float
    host_span_ns: int
    device_span_ns: int
    monotone: bool
    dac_minus_current_mean_ns: int | None
    #: callback(currentTime) 跨度比；与 span_ratio 一起用于区分「取错字段」与
    #: 「设备没给可用 DAC 时钟」。
    current_span_ratio: float | None
    verified: bool
    reasons: tuple[str, ...]

    def map_ns(self, device_ns: int) -> int:
        """把 device 时间映到 host monotonic；仅在 verified 时使用。"""

        return int(device_ns) - self.offset_ns

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "sample_count": self.sample_count,
            "offset_ns": self.offset_ns,
            "residual_max_ns": self.residual_max_ns,
            "residual_mean_abs_ns": self.residual_mean_abs_ns,
            "drift_ns_per_s": round(self.drift_ns_per_s, 3),
            "drift_evaluated": self.drift_evaluated,
            "sample_cost_ns": self.sample_cost_ns,
            "span_ratio": round(self.span_ratio, 6),
            "host_span_ns": self.host_span_ns,
            "device_span_ns": self.device_span_ns,
            "monotone": self.monotone,
            "dac_minus_current_mean_ns": self.dac_minus_current_mean_ns,
            "current_span_ratio": (
                round(self.current_span_ratio, 6)
                if self.current_span_ratio is not None
                else None
            ),
            "verified": self.verified,
            "reasons": list(self.reasons),
        }


def calibrate_pairs(
    pairs: list[tuple[int, int]],
    *,
    currents: list[int] | None = None,
    method: str = "callback_dac_time_vs_host_monotonic",
    max_residual_ns: int = DEFAULT_MAX_RESIDUAL_NS,
    max_drift_ns_per_s: int = DEFAULT_MAX_DRIFT_NS_PER_S,
    span_ratio_range: tuple[float, float] = DEFAULT_SPAN_RATIO_RANGE,
    min_span_for_drift_ns: int = DEFAULT_MIN_SPAN_FOR_DRIFT_NS,
) -> ClockCalibration:
    """用 (host_monotonic_ns, device_ns) 配对样本估计并验证映射。"""

    count = len(pairs)
    empty = ClockCalibration(
        method=method,
        sample_count=count,
        offset_ns=0,
        residual_max_ns=0,
        residual_mean_abs_ns=0,
        drift_ns_per_s=0.0,
        drift_evaluated=False,
        sample_cost_ns=0,
        span_ratio=0.0,
        host_span_ns=0,
        device_span_ns=0,
        monotone=False,
        dac_minus_current_mean_ns=None,
        current_span_ratio=None,
        verified=False,
        reasons=("样本不足，无法校准",),
    )
    if count < MIN_SAMPLES:
        return empty

    hosts = [int(item[0]) for item in pairs]
    devices = [int(item[1]) for item in pairs]
    host_span = hosts[-1] - hosts[0]
    device_span = devices[-1] - devices[0]
    span_ratio = (device_span / host_span) if host_span > 0 else 0.0
    offset = int(sum(dev - hos for dev, hos in zip(devices, hosts)) / count)
    residuals = [(dev - hos) - offset for dev, hos in zip(devices, hosts)]
    residual_max = max(abs(item) for item in residuals)
    residual_mean_abs = int(sum(abs(item) for item in residuals) / count)
    drift = _drift_ns_per_s(hosts, residuals)
    drift_evaluated = host_span >= min_span_for_drift_ns
    sample_cost = int(host_span / max(1, count - 1))
    monotone = all(later > earlier for earlier, later in zip(devices, devices[1:]))
    dac_gap = None
    if currents and len(currents) == count:
        dac_gap = int(sum(dev - cur for dev, cur in zip(devices, currents)) / count)
        current_span = currents[-1] - currents[0]
        current_ratio = (current_span / host_span) if host_span > 0 else 0.0
    else:
        current_ratio = None

    low, high = span_ratio_range
    reasons: list[str] = []
    if host_span <= 0:
        reasons.append("host 时间跨度为 0，无法估计")
    if not monotone:
        reasons.append("device 时间非严格递增")
    if not (low <= span_ratio <= high):
        if current_ratio is not None and low <= current_ratio <= high:
            # callback 时间对得很准，只有 DAC 时间跑偏：这是设备不给可用 DAC 时钟，
            # 不是我们取错字段。能在真机/直通设备上重测前，实际播放时刻只能保持 pending。
            reasons.append(
                f"设备未提供可用的 DAC 时钟：DAC 跨度比 {span_ratio:.4f} 超出 {span_ratio_range}，"
                f"而 callback(currentTime) 跨度比 {current_ratio:.6f} 正常，"
                "说明是虚拟/重采样设备的时间源问题，不是字段取错"
            )
        else:
            reasons.append(
                f"跨度比 {span_ratio:.4f} 超出 {span_ratio_range}，单位或时钟源不一致"
            )
    if residual_max > max_residual_ns:
        reasons.append(f"残差 {residual_max} ns 超过容差 {max_residual_ns} ns")
    if drift_evaluated and abs(drift) > max_drift_ns_per_s:
        reasons.append(f"漂移 {drift:.1f} ns/s 超过容差 {max_drift_ns_per_s} ns/s")

    return ClockCalibration(
        method=method,
        sample_count=count,
        offset_ns=offset,
        residual_max_ns=int(residual_max),
        residual_mean_abs_ns=residual_mean_abs,
        drift_ns_per_s=drift,
        drift_evaluated=drift_evaluated,
        sample_cost_ns=sample_cost,
        span_ratio=span_ratio,
        host_span_ns=int(host_span),
        device_span_ns=int(device_span),
        monotone=monotone,
        dac_minus_current_mean_ns=dac_gap,
        current_span_ratio=current_ratio,
        verified=not reasons,
        reasons=tuple(reasons),
    )


class DeviceClockProbe:
    """在 callback 中采集 (host_monotonic, DAC 时间) 配对并校准。"""

    def __init__(self, *, maxlen: int = 128) -> None:
        self._pairs: deque[tuple[int, int]] = deque(maxlen=maxlen)
        self._currents: deque[int] = deque(maxlen=maxlen)

    def observe(self, time_info: Any, host_ns: int) -> bool:
        """记录一次 callback 的 DAC 时间；``currentTime`` 只作观察。"""

        dac_s = getattr(time_info, "outputBufferDacTime", None)
        if not isinstance(dac_s, (int, float)):
            return False
        current_s = getattr(time_info, "currentTime", None)
        self._pairs.append((host_ns, int(float(dac_s) * 1_000_000_000)))
        self._currents.append(
            int(float(current_s) * 1_000_000_000)
            if isinstance(current_s, (int, float))
            else 0
        )
        return True

    def calibrate(
        self,
        *,
        samples: int = 32,
        min_span_s: float = 1.0,
        timeout_s: float = 10.0,
    ) -> ClockCalibration:
        """等待足够配对与足够跨度后校准；不足时返回未通过验证的对象。

        跨度按**实际参与校准的那一段**计算：只看整个队列的跨度会在切片后
        把观测窗口缩短，导致漂移判据在不该判定的时候被判定。
        """

        needed = max(MIN_SAMPLES, samples)
        min_span_ns = int(min_span_s * 1_000_000_000)
        deadline = time.monotonic() + timeout_s
        while True:
            window = list(self._pairs)[-needed:]
            if len(window) >= needed and (
                window[-1][0] - window[0][0] >= min_span_ns
            ):
                break
            if time.monotonic() > deadline:
                break
            time.sleep(0.01)
        pairs = list(self._pairs)[-needed:]
        currents = list(self._currents)[-len(pairs) :] if pairs else []
        return calibrate_pairs(pairs, currents=currents or None)


def _drift_ns_per_s(hosts: list[int], residuals: list[int]) -> float:
    """残差对 host 时间的最小二乘斜率，单位 ns/s。"""

    span = hosts[-1] - hosts[0]
    if span <= 0 or len(hosts) < 2:
        return 0.0
    mean_t = sum(hosts) / len(hosts)
    mean_r = sum(residuals) / len(residuals)
    numerator = sum(
        (t - mean_t) * (r - mean_r) for t, r in zip(hosts, residuals)
    )
    denominator = sum((t - mean_t) ** 2 for t in hosts)
    if denominator == 0:
        return 0.0
    return (numerator / denominator) * 1_000_000_000.0
