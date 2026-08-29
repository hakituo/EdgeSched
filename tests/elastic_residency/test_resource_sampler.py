import asyncio
import os
import time

from experiments.elastic_residency.contracts import ExperimentArtifactWriter
from experiments.elastic_residency.resource_sampler import (
    EventLoopJitterProbe,
    NvidiaSmiProbe,
    ResourceSampler,
    _advance_periodic_deadline,
)


def _fake_nvidia(command: list[str]) -> str:
    if any("query-gpu" in part for part in command):
        return "512, 8151, 7, 42.5, 55"
    return "999999, 100\n"


class _FakeDirectNvmlProbe:
    backend_name = "direct_nvml"

    def sample(self, _pid: int):
        return {
            "available": True,
            "backend": self.backend_name,
            "card_used_mb": 512.0,
            "card_total_mb": 8151.0,
            "utilization_percent": 7.0,
            "power_w": 42.5,
            "temperature_c": 55.0,
            "process_used_mb": None,
        }


def test_gpu_deadline_skips_missed_ticks_without_adding_full_interval() -> None:
    assert _advance_periodic_deadline(0.0, 0.25, 0.20) == 0.25
    assert _advance_periodic_deadline(0.0, 0.25, 0.624) == 0.75


def test_resource_sampler_writes_resource_and_jitter_timeline(tmp_path) -> None:
    writer = ExperimentArtifactWriter(tmp_path, "sampler_run")
    sampler = ResourceSampler(
        interval_s=0.02,
        gpu_interval_s=0.04,
        writer=writer,
        gpu_probe=NvidiaSmiProbe(_fake_nvidia),
    )
    gpu_ready = sampler.start(wait_for_gpu_timeout_s=0.2)
    time.sleep(0.12)
    summary = sampler.stop()

    assert summary["sample_count"] >= 3
    assert summary["jitter_max_ms"] >= 0
    assert any(sample.gpu_available for sample in sampler.samples)
    assert any(sample.gpu_card_total_mb == 8151 for sample in sampler.samples)
    assert summary["gpu_sample_count"] >= 1
    assert gpu_ready is True
    assert summary["gpu_ready_before_timeline"] is True
    assert summary["gpu_unavailable_timeline_sample_count"] == 0
    assert summary["gpu_max_sample_age_ms"] >= 0
    assert summary["gpu_actual_gap_max_ms"] >= 0
    assert summary["gpu_probe_duration_max_ms"] >= 0
    assert summary["cpu_warmup_samples_discarded"] == 1
    assert summary["cpu_valid_sample_count"] == summary["sample_count"] - 1
    assert sampler.samples[0].cpu_sample_valid is False
    assert all(sample.cpu_sample_valid for sample in sampler.samples[1:])
    assert writer.layout.timeline.read_text().count("\n") == len(sampler.samples)


def test_resource_sampler_records_direct_nvml_without_cli(tmp_path) -> None:
    sampler = ResourceSampler(
        interval_s=0.01,
        gpu_interval_s=0.02,
        writer=ExperimentArtifactWriter(tmp_path, "direct_nvml"),
        gpu_probe=_FakeDirectNvmlProbe(),
    )
    sampler.start(wait_for_gpu_timeout_s=0.1)
    time.sleep(0.05)
    summary = sampler.stop()

    assert summary["gpu_telemetry_backend"] == "direct_nvml"
    assert summary["gpu_cli_subprocess_used"] is False
    assert all(sample.gpu_process_used_mb == 0.0 for sample in sampler.samples)


def test_nvidia_probe_can_skip_unavailable_process_memory_query() -> None:
    commands: list[list[str]] = []

    def _recording_runner(command: list[str]) -> str:
        commands.append(command)
        return _fake_nvidia(command)

    result = NvidiaSmiProbe(
        _recording_runner,
        include_process_memory=False,
    ).sample(os.getpid())

    assert result["available"] is True
    assert result["process_used_mb"] == 0.0
    assert result["process_memory_queried"] is False
    assert len(commands) == 1
    assert any("query-gpu" in part for part in commands[0])


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


def test_resource_sampler_reports_gpu_warmup_failure() -> None:
    sampler = ResourceSampler(
        interval_s=0.01,
        gpu_interval_s=0.02,
        gpu_probe=NvidiaSmiProbe(lambda _command: ""),
    )
    gpu_ready = sampler.start(wait_for_gpu_timeout_s=0.03)
    time.sleep(0.03)
    summary = sampler.stop()

    assert gpu_ready is False
    assert summary["gpu_ready_before_timeline"] is False
    assert summary["gpu_available_timeline_sample_count"] == 0
