"""主机内存分层、PSI 与逐进程记账的解析与汇总门禁。"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pytest

from experiments.elastic_residency import host_memory
from experiments.elastic_residency.resource_sampler import ResourceSampler

MIB = 1024 * 1024

# 取自实际 WSL2（内核 6.6.87.2-microsoft-standard-WSL2）的字段样本。
_MEMINFO = """\
MemTotal:       15968388 kB
MemFree:        15286324 kB
MemAvailable:   15279440 kB
Buffers:           41548 kB
Cached:           132444 kB
SwapCached:            0 kB
Active(anon):      40960 kB
AnonPages:         61488 kB
Shmem:              2968 kB
SReclaimable:      16012 kB
SUnreclaim:        90376 kB
SwapTotal:       4194304 kB
SwapFree:        4194304 kB
HugePages_Total:       0
"""

_PSI_IDLE = """\
some avg10=0.00 avg60=0.00 avg300=0.00 total=0
full avg10=0.00 avg60=0.00 avg300=0.00 total=0
"""

_PSI_PRESSURED = """\
some avg10=1.23 avg60=0.50 avg300=0.10 total=456789
full avg10=0.41 avg60=0.02 avg300=0.00 total=123456
"""


class _FakeProcess:
    def __init__(self, pid: int, name: str, cmdline: tuple[str, ...]) -> None:
        self.info = {"pid": pid, "name": name, "cmdline": cmdline}


class _FakeGpuProbe:
    """本测试只验主机内存，不依赖真实 NVML（Windows 上不可用）。"""

    backend_name = "direct_nvml"

    def sample(self, _pid: int):
        return {"available": False, "backend": self.backend_name}


def _row(
    *,
    monotonic_ns: int,
    available_mib: float,
    swap_mib: float,
    anon_mib: float = 1000.0,
    cached_mib: float = 500.0,
    shmem_mib: float = 10.0,
    sreclaimable_mib: float = 20.0,
    sunreclaim_mib: float = 30.0,
    psi_some_us: int = 0,
    psi_full_us: int = 0,
    psi_full_avg10: float = 0.0,
    watched: dict[str, int] | None = None,
) -> dict[str, Any]:
    return {
        "monotonic_ns": monotonic_ns,
        "system_available_bytes": int(available_mib * MIB),
        "system_swap_used_bytes": int(swap_mib * MIB),
        "system_anon_bytes": int(anon_mib * MIB),
        "system_file_cached_bytes": int(cached_mib * MIB),
        "system_shmem_bytes": int(shmem_mib * MIB),
        "system_sreclaimable_bytes": int(sreclaimable_mib * MIB),
        "system_sunreclaim_bytes": int(sunreclaim_mib * MIB),
        "memory_pressure_some_total_us": psi_some_us,
        "memory_pressure_full_total_us": psi_full_us,
        "memory_pressure_full_avg10": psi_full_avg10,
        "watched_process_rss_bytes": watched or {},
    }


def test_parse_meminfo_converts_kb_to_bytes_and_ignores_unknown_fields() -> None:
    parsed = host_memory.parse_meminfo(_MEMINFO)

    assert parsed["mem_total_bytes"] == 15968388 * 1024
    assert parsed["mem_available_bytes"] == 15279440 * 1024
    assert parsed["anon_bytes"] == 61488 * 1024
    assert parsed["cached_bytes"] == 132444 * 1024
    assert parsed["shmem_bytes"] == 2968 * 1024
    assert parsed["sreclaimable_bytes"] == 16012 * 1024
    assert parsed["sunreclaim_bytes"] == 90376 * 1024
    # 不在映射表里的字段不参与，也不报错
    assert "Active(anon)" not in parsed


def test_parse_meminfo_missing_or_malformed_fields_stay_zero() -> None:
    parsed = host_memory.parse_meminfo("MemTotal: 1024 kB\nCached: 乱码 kB\n")

    assert parsed["mem_total_bytes"] == 1024 * 1024
    assert parsed["cached_bytes"] == 0
    # 字段缺失一律 0，不猜也不抛
    assert parsed["anon_bytes"] == 0
    assert set(parsed) == set(host_memory.MEMINFO_KEYS.values())


def test_parse_psi_memory_reads_both_scopes_and_stays_honest_when_absent() -> None:
    pressured = host_memory.parse_psi_memory(_PSI_PRESSURED)

    assert pressured["available"] is True
    assert pressured["some_avg10"] == 1.23
    assert pressured["some_total_us"] == 456789
    assert pressured["full_avg10"] == 0.41
    assert pressured["full_total_us"] == 123456

    idle = host_memory.parse_psi_memory(_PSI_IDLE)
    assert idle["available"] is True
    assert idle["some_total_us"] == 0

    # 平台不支持时显式标记不可用，不用 0 冒充"无压力"
    for absent in (None, ""):
        missing = host_memory.parse_psi_memory(absent)
        assert missing["available"] is False
        assert missing["full_avg10"] == 0.0


def test_collect_host_memory_state_records_cap_and_swap_state(tmp_path: Path) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(_MEMINFO, encoding="utf-8")
    swaps = tmp_path / "swaps"
    swaps.write_text(
        "Filename\t\t\t\tType\t\tSize\t\tUsed\t\tPriority\n"
        "/dev/sdc                                partition\t4194304\t\t0\t\t-2\n",
        encoding="utf-8",
    )
    swappiness = tmp_path / "swappiness"
    swappiness.write_text("10\n", encoding="utf-8")

    state = host_memory.collect_host_memory_state(
        meminfo_path=meminfo,
        psi_path=tmp_path / "missing_pressure",
        swappiness_path=swappiness,
        min_free_path=tmp_path / "missing_min_free",
        swaps_path=swaps,
    )

    assert state["mem_total_bytes"] == 15968388 * 1024
    assert state["swap_total_bytes"] == 4194304 * 1024
    assert state["vm_swappiness"] == 10
    assert state["swap_devices"] == [
        {
            "filename": "/dev/sdc",
            "type": "partition",
            "size_bytes": 4194304 * 1024,
            "used_bytes": 0,
            "priority": -2,
        }
    ]
    # 缺失的伪文件记录为不可用/无值，而不是伪造默认值
    assert state["psi_available"] is False
    assert state["vm_min_free_kbytes"] is None


def test_resolve_watched_processes_matches_command_line_keywords() -> None:
    processes = [
        _FakeProcess(101, "python", ("python", "launch.py", "--api")),
        _FakeProcess(102, "python", ("python", "-m", "experiments.foo")),
        _FakeProcess(103, "webui.sh", ("bash", "webui.sh")),
    ]

    matched = host_memory.resolve_watched_processes(
        ("launch.py", "webui"), process_iter=processes
    )

    assert matched == {"python#101": 101, "webui.sh#103": 103}
    assert host_memory.resolve_watched_processes((), process_iter=processes) == {}


def test_watched_process_rss_marks_unobservable_as_negative_one() -> None:
    # 已退出/不存在的 pid 必须与"占用 0"区分开
    assert host_memory.watched_process_rss({"gone#999999999": 999999999}) == {
        "gone#999999999": -1
    }
    own = host_memory.watched_process_rss({f"self#{os.getpid()}": os.getpid()})
    assert own[f"self#{os.getpid()}"] > 0


def test_summarize_host_memory_pins_swap_onset_composition() -> None:
    rows = [
        _row(monotonic_ns=0, available_mib=3000.0, swap_mib=0.0, anon_mib=4000.0),
        _row(monotonic_ns=1_000_000_000, available_mib=2900.0, swap_mib=0.2),
        _row(
            monotonic_ns=2_000_000_000,
            available_mib=2308.7,
            swap_mib=40.5,
            anon_mib=5200.0,
            cached_mib=1800.0,
            psi_full_avg10=0.41,
        ),
        _row(
            monotonic_ns=3_000_000_000,
            available_mib=2394.4,
            swap_mib=405.66,
            anon_mib=5300.0,
            psi_some_us=456789,
            psi_full_us=123456,
            watched={"python#101": 3 * MIB},
        ),
    ]

    summary = host_memory.summarize_host_memory(rows)

    assert summary["sample_count"] == 4
    assert summary["available_min_mb"] == pytest.approx(2308.7, abs=0.1)
    assert summary["available_min_elapsed_s"] == 2.0
    assert summary["swap_start_mb"] == 0.0
    assert summary["swap_peak_mb"] == pytest.approx(405.66, abs=0.01)
    # 关键归因：swap 开始增长时 available 仍有 2.3 GiB，不是"内存耗尽"
    onset = summary["swap_onset"]
    assert onset is not None
    assert onset["index"] == 2
    assert onset["available_mb"] == pytest.approx(2308.7, abs=0.1)
    assert onset["anon_mb"] == pytest.approx(5200.0, abs=0.1)
    assert onset["cached_mb"] == pytest.approx(1800.0, abs=0.1)
    assert onset["memory_pressure_full_avg10"] == 0.41
    assert summary["memory_pressure_some_stall_ms"] == pytest.approx(456.789)
    assert summary["memory_pressure_full_stall_ms"] == pytest.approx(123.456)
    assert summary["watched_process_rss_peak_mb"] == {"python#101": 3.0}


def test_summarize_host_memory_without_swap_growth_has_no_onset() -> None:
    rows = [
        _row(monotonic_ns=0, available_mib=3000.0, swap_mib=1.26),
        _row(monotonic_ns=1_000_000_000, available_mib=2950.0, swap_mib=1.50),
    ]

    summary = host_memory.summarize_host_memory(rows)

    assert summary["swap_onset"] is None
    assert summary["swap_growth_mb"] == pytest.approx(0.24, abs=0.001)


def test_summarize_host_memory_empty_timeline_is_explicit() -> None:
    summary = host_memory.summarize_host_memory([])

    assert summary["sample_count"] == 0
    assert summary["swap_onset"] is None
    assert "notes" in summary


def test_probe_once_exposes_sampling_cost_and_main_prints_state(capsys) -> None:
    result = host_memory.probe_once(("__no_such_process_keyword__",))

    assert result["watched_processes"] == {}
    assert result["watched_process_rss_mb"] == {}
    assert result["sample"]["mem_available_mb"] >= 0
    # 采样成本必须可见：它自己也会占内存，不能悄悄改变被测对象
    assert result["single_tick_cost_us"] > 0

    assert host_memory.main([]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["host_memory_state"]["mem_total_bytes"] >= 0
    assert "swap_devices" in payload["host_memory_state"]


def test_resource_sampler_records_layers_psi_and_watched_process(tmp_path) -> None:
    sampler = ResourceSampler(
        interval_s=0.02,
        gpu_interval_s=0.04,
        gpu_probe=_FakeGpuProbe(),
        watch_processes={f"self#{os.getpid()}": os.getpid()},
    )
    sampler.start()
    time.sleep(0.1)
    summary = sampler.stop()

    assert summary["sample_count"] >= 3
    assert summary["host_memory_watched_processes"] == {
        f"self#{os.getpid()}": os.getpid()
    }
    host = summary["host_memory"]
    assert host["sample_count"] == summary["sample_count"]
    # 本平台 psutil 一定可读自身 RSS；分层字段在无 /proc 的平台退化为 0 而非报错
    assert host["watched_process_rss_peak_mb"][f"self#{os.getpid()}"] > 0
    assert all(sample.system_anon_bytes >= 0 for sample in sampler.samples)
    assert all(sample.watched_process_rss_bytes for sample in sampler.samples)
