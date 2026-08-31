"""TTS 隔离矩阵的运行期 swap 与同步 GPU 快照证据。"""

from __future__ import annotations

import statistics
import threading
import time
from collections.abc import Callable, Iterable
from typing import Any

import psutil


SwapPoint = dict[str, int]


def evaluate_swap_timeline(
    samples: Iterable[SwapPoint],
    *,
    interval_s: float,
    tolerance_bytes: int,
    sampler_error: str | None = None,
) -> dict[str, Any]:
    """从原始 swap 时间线生成可复核的增长门禁。"""

    points = list(samples)
    if not points:
        return {
            "samples": [],
            "sample_count": 0,
            "sample_interval_s": interval_s,
            "sampler_error": sampler_error or "no_swap_samples",
            "within_tolerance": False,
            "tolerance_bytes": tolerance_bytes,
        }
    used = [int(point["used_bytes"]) for point in points]
    times = [int(point["monotonic_ns"]) for point in points]
    gaps_ms = [
        (right - left) / 1_000_000.0 for left, right in zip(times, times[1:])
    ]
    growth = max(0, max(used) - used[0])
    return {
        "samples": points,
        "sample_count": len(points),
        "sample_interval_s": interval_s,
        "start_bytes": used[0],
        "max_bytes": max(used),
        "end_bytes": used[-1],
        "growth_bytes": growth,
        "tolerance_bytes": tolerance_bytes,
        "within_tolerance": sampler_error is None and growth <= tolerance_bytes,
        "sampler_error": sampler_error,
        "actual_gap_mean_ms": statistics.fmean(gaps_ms) if gaps_ms else 0.0,
        "actual_gap_max_ms": max(gaps_ms, default=0.0),
    }


class SwapUsageSampler:
    """在 TTS 子进程生命周期内按固定 deadline 采集系统 swap。"""

    def __init__(
        self,
        *,
        interval_s: float = 0.05,
        tolerance_bytes: int = 0,
        used_bytes_provider: Callable[[], int] | None = None,
    ) -> None:
        if interval_s <= 0:
            raise ValueError("swap sample interval 必须大于 0")
        if tolerance_bytes < 0:
            raise ValueError("swap tolerance 不能为负数")
        self.interval_s = interval_s
        self.tolerance_bytes = tolerance_bytes
        self._provider = used_bytes_provider or (
            lambda: int(psutil.swap_memory().used)
        )
        self._samples: list[SwapPoint] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: str | None = None

    def _capture(self) -> None:
        try:
            point = {
                "monotonic_ns": time.monotonic_ns(),
                "used_bytes": int(self._provider()),
            }
        except Exception as exc:  # pragma: no cover - 防御运行时系统探针异常
            self._error = f"{type(exc).__name__}: {exc}"
            self._stop.set()
            return
        with self._lock:
            self._samples.append(point)

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("SwapUsageSampler 不能重复启动")
        self._stop.clear()
        self._capture()
        self._thread = threading.Thread(
            target=self._run,
            name="edgesched-tts-swap-sampler",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        next_deadline = time.monotonic() + self.interval_s
        while not self._stop.is_set():
            wait_s = next_deadline - time.monotonic()
            if wait_s > 0 and self._stop.wait(wait_s):
                break
            self._capture()
            sampled_at = time.monotonic()
            if sampled_at > next_deadline + self.interval_s:
                next_deadline = sampled_at + self.interval_s
            else:
                next_deadline += self.interval_s

    def stop(self) -> dict[str, Any]:
        if self._thread is None:
            raise RuntimeError("SwapUsageSampler 尚未启动")
        self._stop.set()
        self._thread.join(timeout=max(1.0, self.interval_s * 4))
        if self._thread.is_alive():
            self._error = "swap_sampler_thread_did_not_stop"
        self._capture()
        with self._lock:
            points = list(self._samples)
        return evaluate_swap_timeline(
            points,
            interval_s=self.interval_s,
            tolerance_bytes=self.tolerance_bytes,
            sampler_error=self._error,
        )


def _record_snapshots(record: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    return tuple(
        record[name]
        for name in (
            "memory_talker_peak",
            "memory_transfer_peak",
            "memory_codec_peak",
        )
        if isinstance(record.get(name), dict)
    )


def assess_gpu_snapshots(
    snapshots: Iterable[dict[str, Any]], *, max_freshness_bound_ms: float
) -> dict[str, Any]:
    """验证每个显存点来自有界耗时的同步 direct NVML 查询。"""

    items = list(snapshots)
    bounds = [item.get("gpu_sample_freshness_bound_ms") for item in items]
    bounds_valid = all(
        isinstance(value, (int, float)) and 0 <= value <= max_freshness_bound_ms
        for value in bounds
    )
    direct = all(
        item.get("gpu_telemetry_backend") == "direct_nvml"
        and item.get("gpu_cli_subprocess_used") is False
        for item in items
    )
    return {
        "snapshot_count": len(items),
        "max_freshness_bound_ms": max(
            (float(value) for value in bounds if isinstance(value, (int, float))),
            default=None,
        ),
        "freshness_limit_ms": max_freshness_bound_ms,
        "freshness_semantics": "synchronous_direct_nvml_query_duration_upper_bound",
        "all_direct_nvml_without_cli": bool(items) and direct,
        "all_freshness_bounds_valid": bool(items) and bounds_valid,
    }


def finalize_child_runtime_evidence(
    child: dict[str, Any],
    *,
    swap_report: dict[str, Any],
    max_gpu_freshness_ms: float,
) -> dict[str, Any]:
    """把运行期门禁写回 child 与每条 iteration。"""

    records = child.get("records", [])
    all_snapshots: list[dict[str, Any]] = []
    for key in ("memory_baseline", "memory_ready"):
        if isinstance(child.get(key), dict):
            all_snapshots.append(child[key])
    warmup = child.get("warmup", {})
    if isinstance(warmup, dict):
        all_snapshots.extend(_record_snapshots(warmup))
    for record in records:
        all_snapshots.extend(_record_snapshots(record))

    gpu_report = assess_gpu_snapshots(
        all_snapshots,
        max_freshness_bound_ms=max_gpu_freshness_ms,
    )
    swap_ok = swap_report.get("within_tolerance") is True
    for record in records:
        record_gpu = assess_gpu_snapshots(
            _record_snapshots(record),
            max_freshness_bound_ms=max_gpu_freshness_ms,
        )
        record["acceptance_checks"].update(
            gpu_samples_fresh=record_gpu["all_freshness_bounds_valid"],
            swap_growth_within_tolerance=swap_ok,
        )
        record["gpu_sample_freshness_max_ms"] = record_gpu[
            "max_freshness_bound_ms"
        ]
        record["swap_start_bytes"] = swap_report.get("start_bytes")
        record["swap_max_bytes"] = swap_report.get("max_bytes")
        record["swap_growth_bytes"] = swap_report.get("growth_bytes")
        record["passed"] = all(record["acceptance_checks"].values())

    checks = {
        "gpu_snapshots_direct_nvml_without_cli": gpu_report[
            "all_direct_nvml_without_cli"
        ],
        "gpu_samples_fresh": gpu_report["all_freshness_bounds_valid"],
        "swap_growth_within_tolerance": swap_ok,
    }
    child["runtime_evidence"] = {
        "checks": checks,
        "gpu": gpu_report,
        "swap": swap_report,
    }
    child["passed"] = (
        child.get("passed") is True
        and all(checks.values())
        and all(record.get("passed") is True for record in records)
    )
    return child


def compact_runtime_evidence(child: dict[str, Any]) -> dict[str, Any] | None:
    """为 summary 去掉 swap 原始时间线，原始点仍保留在 child JSON。"""

    evidence = child.get("runtime_evidence")
    if not isinstance(evidence, dict):
        return None
    swap = dict(evidence.get("swap", {}))
    swap.pop("samples", None)
    return {
        "checks": evidence.get("checks"),
        "gpu": evidence.get("gpu"),
        "swap": swap,
    }
