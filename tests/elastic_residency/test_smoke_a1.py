from types import SimpleNamespace

from experiments.elastic_residency.contracts import ExperimentArtifactWriter
from experiments.elastic_residency.smoke_a1 import (
    _acceptance_checks,
    _source_git_evidence,
)


def _acceptance_args() -> SimpleNamespace:
    return SimpleNamespace(
        max_gpu_sample_age_ms=500.0,
        max_resource_jitter_p95_ms=5.0,
        max_resource_jitter_max_ms=10.0,
        max_event_loop_jitter_p95_ms=5.0,
        max_event_loop_jitter_max_ms=10.0,
    )


def _resource_summary() -> dict[str, object]:
    return {
        "gpu_telemetry_backend": "direct_nvml",
        "gpu_cli_subprocess_used": False,
        "sample_count": 40,
        "cpu_valid_sample_count": 39,
        "gpu_ready_before_timeline": True,
        "gpu_unavailable_timeline_sample_count": 0,
        "gpu_max_sample_age_ms": 300.0,
        "jitter_p95_ms": 0.4,
        "jitter_max_ms": 0.8,
        "sampler_thread_affinity": [28, 29, 30, 31],
        "gpu_thread_affinity": [28, 29, 30, 31],
        "samples": [SimpleNamespace(system_swap_used_bytes=0)],
    }


def test_a1_acceptance_requires_gpu_and_jitter_gates() -> None:
    resource = _resource_summary()
    checks = _acceptance_checks(
        args=_acceptance_args(),
        topology=SimpleNamespace(physical_cores=tuple(range(16))),
        resource_summary=resource,
        event_loop_summary={"sample_count": 200, "p95_ms": 2.0, "max_ms": 4.0},
        allocator_summary={
            "active_lease_count": 0,
            "free_physical_core_count": 16,
        },
        worker_actual_affinity=(0, 2),
        worker_expected_affinity=(0, 2),
        system_expected_affinity=(28, 29, 30, 31),
        git_provenance_complete=True,
    )
    assert all(checks.values())

    resource["gpu_cli_subprocess_used"] = True
    resource["gpu_ready_before_timeline"] = False
    resource["jitter_p95_ms"] = 6.0
    failed = _acceptance_checks(
        args=_acceptance_args(),
        topology=SimpleNamespace(physical_cores=tuple(range(16))),
        resource_summary=resource,
        event_loop_summary={"sample_count": 200, "p95_ms": 2.0, "max_ms": 4.0},
        allocator_summary={
            "active_lease_count": 0,
            "free_physical_core_count": 16,
        },
        worker_actual_affinity=(0, 2),
        worker_expected_affinity=(0, 2),
        system_expected_affinity=(28, 29, 30, 31),
        git_provenance_complete=True,
    )
    assert failed["gpu_telemetry_is_direct_nvml"] is False
    assert failed["gpu_probe_ready_before_timeline"] is False
    assert failed["resource_jitter_p95_within_limit"] is False
    assert not all(failed.values())


def test_dirty_source_requires_and_copies_snapshot(tmp_path) -> None:
    writer = ExperimentArtifactWriter(tmp_path, "dirty_source")
    snapshot = tmp_path / "source.patch"
    snapshot.write_bytes(b"reconstructable source snapshot")
    args = SimpleNamespace(
        source_git_commit="a" * 40,
        source_git_dirty="true",
        source_git_status_entry_count=2,
        source_git_snapshot=str(snapshot),
        source_git_snapshot_sha256=None,
    )
    evidence, complete = _source_git_evidence(args, writer)

    assert complete is True
    assert evidence["snapshot"] == "source_git_snapshot.patch"
    assert len(evidence["snapshot_sha256"]) == 64
    assert (writer.layout.root / evidence["snapshot"]).read_bytes() == snapshot.read_bytes()


def test_clean_source_requires_zero_status_entries(tmp_path) -> None:
    writer = ExperimentArtifactWriter(tmp_path, "clean_source")
    args = SimpleNamespace(
        source_git_commit="b" * 40,
        source_git_dirty="false",
        source_git_status_entry_count=0,
        source_git_snapshot=None,
        source_git_snapshot_sha256=None,
    )
    _, complete = _source_git_evidence(args, writer)
    assert complete is True

    args.source_git_status_entry_count = 1
    _, complete = _source_git_evidence(args, writer)
    assert complete is False
