"""主机内存分层、PSI 压力与逐进程记账采样。

现有时间线只有「子进程 RSS + 系统 available + swap used」三项，回答不了三个问题：

1. swap 换出的是谁 —— 匿名页还是本可回收的文件页；
2. 有没有真实内存压力停顿 —— 还是内核只是按 ``swappiness`` 选了换出匿名页；
3. 外部服务（例如 Forge）常驻占了多少主机内存 —— 它不在本进程 RSS 里。

本模块把可回收性（匿名/文件/可回收 slab）、PSI 停顿累计与指定进程逐点 RSS 补上。

WSL2 内核约束：``6.6.87.2-microsoft-standard-WSL2`` 的 ``/proc/meminfo`` 没有
``FilePages`` 字段，因此文件页用 ``Cached`` 与 ``Shmem`` 表达；``Cached`` 含
``Shmem``，可用文件缓存 = ``Cached - Shmem``。缺字段一律记 0，不猜。
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import psutil

MEMINFO_PATH = Path("/proc/meminfo")
PSI_MEMORY_PATH = Path("/proc/pressure/memory")
SWAPPINESS_PATH = Path("/proc/sys/vm/swappiness")
MIN_FREE_KBYTES_PATH = Path("/proc/sys/vm/min_free_kbytes")
SWAPS_PATH = Path("/proc/swaps")

# /proc/meminfo 字段 → 统一命名。只映射我们真正会用的字段。
MEMINFO_KEYS: dict[str, str] = {
    "MemTotal": "mem_total_bytes",
    "MemFree": "mem_free_bytes",
    "MemAvailable": "mem_available_bytes",
    "Buffers": "buffers_bytes",
    "Cached": "cached_bytes",
    "SwapCached": "swap_cached_bytes",
    "SwapTotal": "swap_total_bytes",
    "SwapFree": "swap_free_bytes",
    "AnonPages": "anon_bytes",
    "Shmem": "shmem_bytes",
    "SReclaimable": "sreclaimable_bytes",
    "SUnreclaim": "sunreclaim_bytes",
}


def read_text(path: Path) -> str | None:
    """读取伪文件；任何失败都返回 None，采样器不能因为探测失败而倒下。"""

    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def parse_meminfo(text: str) -> dict[str, int]:
    """把 ``/proc/meminfo`` 解析成字节。字段缺失记 0，不推断。"""

    parsed = {name: 0 for name in MEMINFO_KEYS.values()}
    for line in text.splitlines():
        key, separator, rest = line.partition(":")
        if not separator:
            continue
        name = MEMINFO_KEYS.get(key.strip())
        if name is None:
            continue
        parts = rest.split()
        if not parts:
            continue
        try:
            value = int(parts[0])
        except ValueError:
            continue
        # /proc/meminfo 除 HugePages 外都是 kB。
        unit = parts[1].lower() if len(parts) > 1 else "kb"
        parsed[name] = value if unit == "b" else value * 1024
    return parsed


def parse_psi_memory(text: str | None) -> dict[str, Any]:
    """解析 ``/proc/pressure/memory`` 的 some/full 累计停顿。

    ``total`` 是自启动以来累计停顿微秒数，逐点保存原始值，窗口内增量由汇总
    阶段计算，避免采样本模块丢掉中间状态。PSI 不可用时显式标 ``available``。
    """

    result: dict[str, Any] = {
        "available": False,
        "some_avg10": 0.0,
        "some_total_us": 0,
        "full_avg10": 0.0,
        "full_total_us": 0,
    }
    if not text:
        return result
    seen = False
    for line in text.splitlines():
        parts = line.split()
        if not parts or parts[0] not in ("some", "full"):
            continue
        scope = parts[0]
        for token in parts[1:]:
            key, separator, value = token.partition("=")
            if not separator:
                continue
            try:
                if key in ("avg10", "avg60", "avg300"):
                    if key != "avg10":
                        continue
                    result[f"{scope}_{key}"] = float(value)
                elif key == "total":
                    result[f"{scope}_total_us"] = int(value)
                else:
                    continue
            except ValueError:
                continue
            seen = True
    result["available"] = seen
    return result


def collect_host_memory_state(
    *,
    meminfo_path: Path = MEMINFO_PATH,
    psi_path: Path = PSI_MEMORY_PATH,
    swappiness_path: Path = SWAPPINESS_PATH,
    min_free_path: Path = MIN_FREE_KBYTES_PATH,
    swaps_path: Path = SWAPS_PATH,
) -> dict[str, Any]:
    """一次性记录「当次内存上限与 swap 状态」。

    论文环境约束是 WSL2 的有效内存上限而非宿主机容量，因此这两项必须随
    每次运行落盘；缺少它们，别人无法判断复现差异来自代码还是来自内存上限。
    """

    meminfo_text = read_text(meminfo_path) or ""
    meminfo = parse_meminfo(meminfo_text)
    psi_text = read_text(psi_path)
    swappiness_text = read_text(swappiness_path)
    min_free_text = read_text(min_free_path)
    swaps_text = read_text(swaps_path)

    def _int_of(text: str | None) -> int | None:
        if text is None:
            return None
        try:
            return int(text.strip())
        except ValueError:
            return None

    swap_devices: list[dict[str, Any]] = []
    for line in (swaps_text or "").splitlines():
        parts = line.split()
        if len(parts) < 5 or parts[0].lower() == "filename":
            continue
        try:
            swap_devices.append(
                {
                    "filename": parts[0],
                    "type": parts[1],
                    "size_bytes": int(parts[2]) * 1024,
                    "used_bytes": int(parts[3]) * 1024,
                    "priority": int(parts[4]),
                }
            )
        except ValueError:
            continue

    return {
        "mem_total_bytes": meminfo["mem_total_bytes"],
        "mem_available_bytes": meminfo["mem_available_bytes"],
        "swap_total_bytes": meminfo["swap_total_bytes"],
        "swap_free_bytes": meminfo["swap_free_bytes"],
        "vm_swappiness": _int_of(swappiness_text),
        "vm_min_free_kbytes": _int_of(min_free_text),
        "swap_devices": swap_devices,
        "meminfo_fields_present": sorted(
            name for name, value in meminfo.items() if value
        ),
        "psi_available": bool(psi_text),
        "psi_source": str(psi_path),
        "cgroup_memory_controller_present": Path(
            "/sys/fs/cgroup/memory.max"
        ).exists(),
    }


def resolve_watched_processes(
    patterns: Sequence[str],
    *,
    process_iter: Iterable[Any] | None = None,
) -> dict[str, int]:
    """按关键字匹配进程一次，返回 ``标签 → pid``。

    外部服务（Forge）不在被测进程 RSS 里，只能按进程记账。标签带 pid，
    便于同一个服务出现多进程时不被合并。
    """

    needles = [item.lower() for item in patterns if item]
    if not needles:
        return {}
    matched: dict[str, int] = {}
    iterator = process_iter if process_iter is not None else psutil.process_iter(
        ["pid", "name", "cmdline"]
    )
    for proc in iterator:
        try:
            pid = int(proc.info["pid"])
            name = str(proc.info.get("name") or "process")
            cmdline = proc.info.get("cmdline") or ()
        except (KeyError, TypeError, ValueError):
            continue
        haystack = " ".join(str(part) for part in cmdline).lower()
        if not any(needle in haystack for needle in needles):
            continue
        matched[f"{name}#{pid}"] = pid
    return matched


def watched_process_rss(label_to_pid: Mapping[str, int]) -> dict[str, int]:
    """读取被观察进程的 RSS；进程已退出时记 -1，表示不可观测而非 0。"""

    rss: dict[str, int] = {}
    for label, pid in label_to_pid.items():
        try:
            rss[label] = int(psutil.Process(pid).memory_info().rss)
        except (psutil.Error, OSError):
            rss[label] = -1
    return rss


def _as_mb(value: Any) -> float:
    try:
        return float(value) / 1048576.0
    except (TypeError, ValueError):
        return 0.0


def _get(row: Any, key: str, default: Any = 0) -> Any:
    if isinstance(row, Mapping):
        return row.get(key, default)
    return getattr(row, key, default)


def summarize_host_memory(
    rows: Sequence[Any],
    *,
    swap_onset_threshold_bytes: int = 1024 * 1024,
) -> dict[str, Any]:
    """汇总主机内存时间线，重点给出 swap 起点那一刻的分层构成。

    ``swap_onset`` 是归因的核心：如果 swap 开始增长时 ``available`` 仍很宽裕，
    就不能把失败解释成「内存耗尽」，而要去看回收了谁、压力从哪来。
    """

    if not rows:
        return {
            "sample_count": 0,
            "swap_onset": None,
            "notes": "无主机内存时间线样本",
        }

    available = [_get(row, "system_available_bytes") for row in rows]
    swap = [_get(row, "system_swap_used_bytes") for row in rows]
    anon = [_get(row, "system_anon_bytes") for row in rows]
    cached = [_get(row, "system_file_cached_bytes") for row in rows]
    shmem = [_get(row, "system_shmem_bytes") for row in rows]
    sreclaimable = [_get(row, "system_sreclaimable_bytes") for row in rows]
    sunreclaim = [_get(row, "system_sunreclaim_bytes") for row in rows]
    psi_some = [_get(row, "memory_pressure_some_total_us") for row in rows]
    psi_full = [_get(row, "memory_pressure_full_total_us") for row in rows]
    psi_avg10 = [_get(row, "memory_pressure_full_avg10") for row in rows]
    first_ns = int(_get(rows[0], "monotonic_ns", 0))

    min_index = min(range(len(rows)), key=lambda i: available[i])
    onset_index: int | None = None
    if swap:
        baseline = swap[0]
        for index, value in enumerate(swap):
            if value > baseline + swap_onset_threshold_bytes:
                onset_index = index
                break

    watched_peak: dict[str, float] = {}
    for row in rows:
        for label, value in (_get(row, "watched_process_rss_bytes", {}) or {}).items():
            if value is None or value < 0:
                continue
            watched_peak[label] = max(watched_peak.get(label, 0.0), _as_mb(value))

    def _elapsed(index: int | None) -> float | None:
        if index is None:
            return None
        return (int(_get(rows[index], "monotonic_ns", first_ns)) - first_ns) / 1e9

    swap_onset = None
    if onset_index is not None:
        swap_onset = {
            "index": onset_index,
            "elapsed_s": _elapsed(onset_index),
            "available_mb": _as_mb(available[onset_index]),
            "swap_used_mb": _as_mb(swap[onset_index]),
            "swap_baseline_mb": _as_mb(swap[0]),
            "anon_mb": _as_mb(anon[onset_index]),
            "cached_mb": _as_mb(cached[onset_index]),
            "shmem_mb": _as_mb(shmem[onset_index]),
            "sreclaimable_mb": _as_mb(sreclaimable[onset_index]),
            "sunreclaim_mb": _as_mb(sunreclaim[onset_index]),
            "memory_pressure_full_avg10": float(psi_avg10[onset_index] or 0.0),
        }

    return {
        "sample_count": len(rows),
        "available_min_mb": _as_mb(available[min_index]),
        "available_min_elapsed_s": _elapsed(min_index),
        "available_start_mb": _as_mb(available[0]),
        "available_end_mb": _as_mb(available[-1]),
        "anon_peak_mb": _as_mb(max(anon)),
        "anon_end_mb": _as_mb(anon[-1]),
        "cached_peak_mb": _as_mb(max(cached)),
        "shmem_peak_mb": _as_mb(max(shmem)),
        "sreclaimable_peak_mb": _as_mb(max(sreclaimable)),
        "sunreclaim_peak_mb": _as_mb(max(sunreclaim)),
        "swap_start_mb": _as_mb(swap[0]),
        "swap_peak_mb": _as_mb(max(swap)),
        "swap_end_mb": _as_mb(swap[-1]),
        "swap_growth_mb": _as_mb(max(swap) - swap[0]),
        "swap_last_index": max(
            index for index, value in enumerate(swap) if value == max(swap)
        ),
        "memory_pressure_some_stall_ms": (psi_some[-1] - psi_some[0]) / 1000.0,
        "memory_pressure_full_stall_ms": (psi_full[-1] - psi_full[0]) / 1000.0,
        "memory_pressure_full_avg10_peak": float(max(psi_avg10) if psi_avg10 else 0.0),
        "watched_process_rss_peak_mb": watched_peak,
        "swap_onset": swap_onset,
        "swap_onset_detection": (
            f"首点相对基线增长超过 {swap_onset_threshold_bytes} B"
        ),
    }


def default_forge_patterns() -> tuple[str, ...]:
    """外部 Forge 服务的进程标识；命令行与启动脚本都可命中。"""

    patterns = ["launch.py", "webui", "forge"]
    env_extra = os.environ.get("XIAOYOU_HOST_MEMORY_WATCH", "")
    patterns.extend(part.strip() for part in env_extra.split(",") if part.strip())
    return tuple(patterns)


def probe_once(patterns: Sequence[str]) -> dict[str, Any]:
    """一次活体采样：分层页计数、PSI、被观察进程 RSS 与单次采样成本。"""

    import time

    watched = resolve_watched_processes(patterns)
    started = time.perf_counter()
    meminfo = parse_meminfo(read_text(MEMINFO_PATH) or "")
    psi = parse_psi_memory(read_text(PSI_MEMORY_PATH))
    rss = watched_process_rss(watched)
    cost_us = (time.perf_counter() - started) * 1e6

    def _mb(key: str) -> float:
        return meminfo[key] / 1048576.0

    return {
        "sample": {
            "mem_available_mb": _mb("mem_available_bytes"),
            "anon_mb": _mb("anon_bytes"),
            "cached_mb": _mb("cached_bytes"),
            "shmem_mb": _mb("shmem_bytes"),
            "sreclaimable_mb": _mb("sreclaimable_bytes"),
            "sunreclaim_mb": _mb("sunreclaim_bytes"),
            "psi_available": psi["available"],
            "psi_full_avg10": psi["full_avg10"],
            "psi_full_total_us": psi["full_total_us"],
        },
        "watched_processes": watched,
        "watched_process_rss_mb": {
            label: (value / 1048576.0 if value >= 0 else -1.0)
            for label, value in rss.items()
        },
        # 采样本身会占内存；这个成本必须可见，不能悄悄改变被测对象。
        "single_tick_cost_us": cost_us,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """打印当次内存上限与 swap 状态；运行实验前落盘用。

    论文环境约束是 WSL2 的有效内存上限，因此每次运行都要记，不能只写宿主机容量。
    """

    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--watch",
        nargs="*",
        default=list(default_forge_patterns()),
        help="按命令行关键字观察的进程（默认命中 Forge 服务）",
    )
    args = parser.parse_args(argv)

    payload = {"host_memory_state": collect_host_memory_state()}
    payload.update(probe_once(args.watch))
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - 手动诊断入口
    raise SystemExit(main())
