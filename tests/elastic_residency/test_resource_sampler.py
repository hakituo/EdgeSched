import asyncio
import os
import time

from experiments.elastic_residency.contracts import ExperimentArtifactWriter
from experiments.elastic_residency.resource_sampler import (
    EventLoopJitterProbe,
    NvidiaSmiProbe,
    ResourceSampler,
)


def _fake_nvidia(command: list[str]) -> str:
    if any("query-gpu" in part for part in command):
        return "512, 8151, 7, 42.5, 55"
    return "999999, 100\n"


def test_resource_sampler_writes_resource_and_jitter_timeline(tmp_path) -> None:
    writer = ExperimentArtifactWriter(tmp_path, "sampler_run")
    sampler = ResourceSampler(
        interval_s=0.02,
        gpu_interval_s=0.04,
        writer=writer,
        gpu_probe=NvidiaSmiProbe(_fake_nvidia),
    )
    sampler.start()
    time.sleep(0.12)
    summary = sampler.stop()

    assert summary["sample_count"] >= 3
    assert summary["jitter_max_ms"] >= 0
    assert any(sample.gpu_available for sample in sampler.samples)
    assert any(sample.gpu_card_total_mb == 8151 for sample in sampler.samples)
    assert summary["gpu_sample_count"] >= 1
    assert writer.layout.timeline.read_text().count("\n") == len(sampler.samples)


def test_event_loop_jitter_probe_collects_deadline_error() -> None:
    summary = asyncio.run(EventLoopJitterProbe(0.005).run_for(0.04))
    assert summary["sample_count"] >= 3
    assert summary["max_ms"] >= 0


def test_resource_sampler_rejects_empty_explicit_thread_affinity() -> None:
    try:
        ResourceSampler(thread_affinity=[])
    except ValueError as exc:
        assert "thread_affinity" in str(exc)
    else:
        raise AssertionError("显式空 affinity 应被拒绝")


def test_resource_sampler_reports_current_thread_affinity_on_linux() -> None:
    if not hasattr(os, "sched_getaffinity"):
        return
    current = sorted(os.sched_getaffinity(0))
    sampler = ResourceSampler(
        interval_s=0.01,
        gpu_interval_s=0.02,
        gpu_probe=NvidiaSmiProbe(_fake_nvidia),
        thread_affinity=current[:1],
    )
    sampler.start()
    time.sleep(0.04)
    summary = sampler.stop()

    assert summary["sampler_thread_affinity"] == current[:1]
    assert summary["gpu_thread_affinity"] == current[:1]
