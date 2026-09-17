"""TTS 画像的阶段内存快照：准备阶段（加载 + 图捕获）与执行阶段（合成）。

用途：新版 B2 子集画像必须同时给出「准备阶段」和「执行阶段」的内存峰值，
才能支撑后续共置判断。四个来源各记各的，不互相替代：

- ``torch`` 分配/保留峰值：进程内 CUDA 分配器视角，最贴近模型实际占用；
- ``NVML`` 整卡已用：包含进程外的其他占用，隔离实验里用来交叉核对；
- ``NVML`` 进程占用：可能不可用（无 compute 进程行），不可用时如实标注；
- 进程 RSS：主机内存侧，判断是否触发 swap。

NVML 依赖 ``pynvml``，隔离环境可能没装；此处一律降级为
``available: false`` 并写入原因，不因为探测失败就让画像整体失败。
"""

from __future__ import annotations

import time
from typing import Any


def reset_torch_peaks() -> None:
    """重置 torch 的峰值统计，作为阶段起点。"""

    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def process_rss_mb(pid: int) -> float | None:
    try:
        import psutil

        return round(psutil.Process(pid).memory_info().rss / 1048576.0, 1)
    except Exception:  # noqa: BLE001 - 探测失败不影响画像
        return None


def gpu_snapshot(pid: int) -> dict[str, Any]:
    """direct NVML 快照；不可用时返回原因而不是抛错。"""

    try:
        from experiments.elastic_residency.tts_gpu_snapshot import (
            nvidia_memory_snapshot,
        )

        return nvidia_memory_snapshot(pid)
    except Exception as exc:  # noqa: BLE001
        return {
            "gpu_card_available": False,
            "gpu_card_used_mb": None,
            "gpu_card_total_mb": None,
            "gpu_process_total_mb": None,
            "gpu_process_total_available": False,
            "gpu_telemetry_backend": "unavailable",
            "gpu_unavailable_reason": f"{type(exc).__name__}: {exc}",
        }


def torch_peaks_mb() -> dict[str, float | None]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"allocated_mb": None, "reserved_mb": None}
        return {
            "allocated_mb": round(torch.cuda.max_memory_allocated() / 1048576.0, 1),
            "reserved_mb": round(torch.cuda.max_memory_reserved() / 1048576.0, 1),
        }
    except Exception:  # noqa: BLE001
        return {"allocated_mb": None, "reserved_mb": None}


class PhaseMemory:
    """一个阶段的峰值记录器；阶段内可以多次 ``sample()`` 取最大值。"""

    def __init__(self, label: str, pid: int) -> None:
        self.label = label
        self.pid = pid
        self.started_s = time.perf_counter()
        self.samples: list[dict[str, Any]] = []
        reset_torch_peaks()

    def sample(self) -> dict[str, Any]:
        record = {
            "rss_mb": process_rss_mb(self.pid),
            **gpu_snapshot(self.pid),
        }
        self.samples.append(record)
        return record

    def stop(self) -> dict[str, Any]:
        peaks = torch_peaks_mb()
        card = [
            float(item["gpu_card_used_mb"])
            for item in self.samples
            if item.get("gpu_card_used_mb") is not None
        ]
        process = [
            float(item["gpu_process_total_mb"])
            for item in self.samples
            if item.get("gpu_process_total_mb") is not None
        ]
        rss = [float(item["rss_mb"]) for item in self.samples if item.get("rss_mb")]
        first = self.samples[0] if self.samples else {}
        return {
            "label": self.label,
            "duration_s": round(time.perf_counter() - self.started_s, 3),
            "sample_count": len(self.samples),
            "torch_peak_allocated_mb": peaks["allocated_mb"],
            "torch_peak_reserved_mb": peaks["reserved_mb"],
            "gpu_card_used_mb_max": max(card) if card else None,
            "gpu_card_used_mb_first": card[0] if card else None,
            "gpu_process_used_mb_max": max(process) if process else None,
            "rss_mb_max": max(rss) if rss else None,
            "gpu_card_available": bool(first.get("gpu_card_available")),
            "gpu_process_available": bool(first.get("gpu_process_total_available")),
            "gpu_unavailable_reason": first.get("gpu_unavailable_reason")
            or first.get("gpu_process_total_unavailable_reason"),
            "gpu_telemetry_backend": first.get("gpu_telemetry_backend"),
            "rss_mb_semantics": "psutil_process_rss_at_sampling_points",
            "torch_peak_semantics": "max_memory_allocated_or_reserved_since_phase_start",
        }
