#!/usr/bin/env python3
"""A1 基础设施真实 smoke：写产物、分核、资源与事件循环 jitter。"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.elastic_residency.contracts import (  # noqa: E402
    BackendEvidence,
    ExperimentArtifactWriter,
    ExperimentEvent,
    ExperimentRunConfig,
)
from experiments.elastic_residency.cpu_topology import (  # noqa: E402
    CPUTopology,
    PhysicalCoreAllocator,
    process_affinity_scope,
)
from experiments.elastic_residency.resource_sampler import (  # noqa: E402
    EventLoopJitterProbe,
    ResourceSampler,
)


def _iso_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _runtime_git_state() -> dict[str, object]:
    def run(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        return completed.stdout.strip() if completed.returncode == 0 else ""

    commit = run("rev-parse", "HEAD")
    if not commit:
        return {
            "available": False,
            "commit": None,
            "dirty": None,
            "status_entry_count": None,
            "reason": "runtime_copy_has_no_git_metadata",
        }
    status_lines = run("status", "--porcelain").splitlines()
    return {
        "available": True,
        "commit": commit,
        "dirty": bool(status_lines),
        "status_entry_count": len(status_lines),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_git_evidence(
    args: argparse.Namespace,
    writer: ExperimentArtifactWriter,
) -> tuple[dict[str, object], bool]:
    dirty = (
        args.source_git_dirty == "true"
        if args.source_git_dirty is not None
        else None
    )
    evidence: dict[str, object] = {
        "commit": args.source_git_commit or None,
        "dirty": dirty,
        "status_entry_count": args.source_git_status_entry_count,
        "snapshot": None,
        "snapshot_sha256": None,
    }
    snapshot_complete = False
    if args.source_git_snapshot:
        source = Path(args.source_git_snapshot).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"源码快照不存在: {source}")
        actual_sha256 = _sha256_file(source)
        if (
            args.source_git_snapshot_sha256
            and actual_sha256 != args.source_git_snapshot_sha256.lower()
        ):
            raise ValueError("源码快照 SHA-256 与命令行声明不一致")
        suffix = source.suffix if source.suffix else ".bin"
        destination = writer.layout.root / f"source_git_snapshot{suffix}"
        if source != destination.resolve():
            shutil.copyfile(source, destination)
        evidence["snapshot"] = destination.name
        evidence["snapshot_sha256"] = actual_sha256
        snapshot_complete = True

    clean_complete = (
        dirty is False and args.source_git_status_entry_count == 0
    )
    dirty_complete = (
        dirty is True
        and bool(args.source_git_status_entry_count)
        and snapshot_complete
    )
    complete = bool(args.source_git_commit) and (
        clean_complete or dirty_complete
    )
    return evidence, complete


def _acceptance_checks(
    *,
    args: argparse.Namespace,
    topology: CPUTopology,
    resource_summary: dict[str, object],
    event_loop_summary: dict[str, float | int],
    allocator_summary: dict[str, object],
    worker_actual_affinity: tuple[int, ...],
    worker_expected_affinity: tuple[int, ...],
    system_expected_affinity: tuple[int, ...],
    git_provenance_complete: bool,
) -> dict[str, bool]:
    linux_affinity = hasattr(os, "sched_setaffinity")
    return {
        "resource_samples_sufficient": resource_summary["sample_count"] >= 2,
        "cpu_samples_after_warmup_sufficient": (
            resource_summary["cpu_valid_sample_count"] >= 1
        ),
        "event_loop_samples_sufficient": event_loop_summary["sample_count"] >= 2,
        "gpu_probe_ready_before_timeline": bool(
            resource_summary["gpu_ready_before_timeline"]
        ),
        "gpu_present_for_every_timeline_sample": (
            resource_summary["gpu_unavailable_timeline_sample_count"] == 0
        ),
        "gpu_sample_age_within_limit": (
            resource_summary["gpu_max_sample_age_ms"]
            <= args.max_gpu_sample_age_ms
        ),
        "resource_jitter_p95_within_limit": (
            resource_summary["jitter_p95_ms"]
            <= args.max_resource_jitter_p95_ms
        ),
        "resource_jitter_max_within_limit": (
            resource_summary["jitter_max_ms"]
            <= args.max_resource_jitter_max_ms
        ),
        "event_loop_jitter_p95_within_limit": (
            event_loop_summary["p95_ms"]
            <= args.max_event_loop_jitter_p95_ms
        ),
        "event_loop_jitter_max_within_limit": (
            event_loop_summary["max_ms"]
            <= args.max_event_loop_jitter_max_ms
        ),
        "swap_unused": all(
            sample.system_swap_used_bytes == 0
            for sample in resource_summary["samples"]
        ),
        "all_leases_released": allocator_summary["active_lease_count"] == 0,
        "all_physical_cores_returned": (
            allocator_summary["free_physical_core_count"]
            == len(topology.physical_cores)
        ),
        "source_git_provenance_complete": git_provenance_complete,
        "worker_affinity_matches": (
            not linux_affinity
            or set(worker_actual_affinity) == set(worker_expected_affinity)
        ),
        "sampler_affinity_matches": (
            not linux_affinity
            or set(resource_summary["sampler_thread_affinity"])
            == set(system_expected_affinity)
        ),
        "gpu_sampler_affinity_matches": (
            not linux_affinity
            or set(resource_summary["gpu_thread_affinity"])
            == set(system_expected_affinity)
        ),
    }


async def run(args: argparse.Namespace) -> Path:
    topology = CPUTopology.discover()
    if len(topology.physical_cores) < args.system_cores + args.worker_cores:
        raise RuntimeError("物理核心不足，无法执行 A1 smoke")
    allocator = PhysicalCoreAllocator(topology)
    writer = ExperimentArtifactWriter(args.output_root, args.run_id)
    source_git, git_provenance_complete = _source_git_evidence(args, writer)
    config = ExperimentRunConfig(
        experiment_name="elastic_residency_a1_smoke",
        run_id=args.run_id,
        sample_interval_s=args.sample_interval,
        gpu_sample_interval_s=args.gpu_sample_interval,
        backends=(
            BackendEvidence(
                component="control_plane",
                backend_kind="real_os_resource_probe",
                is_real=True,
            ),
        ),
        parameters={
            "duration_s": args.duration,
            "system_cores": args.system_cores,
            "worker_cores": args.worker_cores,
            "acceptance_thresholds": {
                "max_resource_jitter_p95_ms": args.max_resource_jitter_p95_ms,
                "max_resource_jitter_max_ms": args.max_resource_jitter_max_ms,
                "max_event_loop_jitter_p95_ms": (
                    args.max_event_loop_jitter_p95_ms
                ),
                "max_event_loop_jitter_max_ms": (
                    args.max_event_loop_jitter_max_ms
                ),
                "max_gpu_sample_age_ms": args.max_gpu_sample_age_ms,
                "gpu_warmup_timeout_s": args.gpu_warmup_timeout,
            },
        },
    )
    writer.initialize(
        config,
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pid": os.getpid(),
            "cpu_topology": topology.to_dict(),
            "git": {
                "runtime_copy": _runtime_git_state(),
                "source_copy": source_git,
            },
            "models": [],
        },
    )
    system_lease = allocator.reserve_system(args.system_cores)
    worker_lease = allocator.acquire("a1_worker", args.worker_cores)
    writer.append_event(
        ExperimentEvent(
            name="a1_start",
            monotonic_ns=time.monotonic_ns(),
            wall_time_iso=_iso_now(),
            phase="a1",
            attributes={
                "system_affinity": system_lease.affinity,
                "worker_affinity": worker_lease.affinity,
            },
        )
    )
    with process_affinity_scope(worker_lease.affinity) as actual_affinity:
        sampler = ResourceSampler(
            interval_s=args.sample_interval,
            gpu_interval_s=args.gpu_sample_interval,
            writer=writer,
            thread_affinity=system_lease.affinity,
        )
        sampler.start(wait_for_gpu_timeout_s=args.gpu_warmup_timeout)
        event_loop_summary = await EventLoopJitterProbe(
            interval_s=min(0.01, args.sample_interval)
        ).run_for(args.duration)
        resource_summary = sampler.stop()
    worker_lease.release()
    system_lease.release()
    allocator_summary = allocator.snapshot()
    resource_summary["samples"] = list(sampler.samples)
    writer.append_event(
        ExperimentEvent(
            name="a1_finish",
            monotonic_ns=time.monotonic_ns(),
            wall_time_iso=_iso_now(),
            phase="a1",
            attributes={"allocator": allocator_summary},
        )
    )
    acceptance_checks = _acceptance_checks(
        args=args,
        topology=topology,
        resource_summary=resource_summary,
        event_loop_summary=event_loop_summary,
        allocator_summary=allocator_summary,
        worker_actual_affinity=actual_affinity,
        worker_expected_affinity=worker_lease.affinity,
        system_expected_affinity=system_lease.affinity,
        git_provenance_complete=git_provenance_complete,
    )
    resource_summary.pop("samples")
    summary = {
        "resource_sampler": resource_summary,
        "event_loop_jitter": event_loop_summary,
        "allocator_after_release": allocator_summary,
        "worker_actual_affinity": list(actual_affinity),
        "git_provenance_complete": git_provenance_complete,
        "acceptance_checks": acceptance_checks,
        "passed": all(acceptance_checks.values()),
    }
    writer.append_iteration(
        {
            "iteration": 1,
            "duration_s": args.duration,
            **summary,
        }
    )
    writer.write_summary(summary)
    writer.write_summary_csv(
        {
            "run_id": args.run_id,
            "passed": summary["passed"],
            "duration_s": args.duration,
            "resource_sample_count": resource_summary["sample_count"],
            "resource_jitter_p95_ms": resource_summary["jitter_p95_ms"],
            "resource_jitter_max_ms": resource_summary["jitter_max_ms"],
            "event_loop_sample_count": event_loop_summary["sample_count"],
            "event_loop_jitter_p95_ms": event_loop_summary["p95_ms"],
            "event_loop_jitter_max_ms": event_loop_summary["max_ms"],
            "worker_actual_affinity": list(actual_affinity),
            "sampler_thread_affinity": resource_summary[
                "sampler_thread_affinity"
            ],
            "gpu_thread_affinity": resource_summary["gpu_thread_affinity"],
            "free_physical_core_count": allocator_summary[
                "free_physical_core_count"
            ],
            "source_git_commit": args.source_git_commit,
            "source_git_dirty": source_git["dirty"],
            "source_git_snapshot_sha256": source_git["snapshot_sha256"],
        }
    )
    return writer.layout.root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "experiment_results" / "elastic_residency" / "a1"),
    )
    parser.add_argument(
        "--run-id",
        default=f"a1_smoke_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument("--duration", type=float, default=0.5)
    parser.add_argument("--sample-interval", type=float, default=0.05)
    parser.add_argument("--gpu-sample-interval", type=float, default=0.25)
    parser.add_argument("--system-cores", type=int, default=2)
    parser.add_argument("--worker-cores", type=int, default=2)
    parser.add_argument("--source-git-commit", default="")
    parser.add_argument("--source-git-dirty", choices=("true", "false"))
    parser.add_argument("--source-git-status-entry-count", type=int)
    parser.add_argument(
        "--source-git-snapshot",
        help="dirty 源码的可还原快照或补丁；会复制进本次运行目录",
    )
    parser.add_argument("--source-git-snapshot-sha256")
    parser.add_argument("--gpu-warmup-timeout", type=float, default=5.0)
    parser.add_argument("--max-gpu-sample-age-ms", type=float, default=500.0)
    parser.add_argument(
        "--max-resource-jitter-p95-ms", type=float, default=5.0
    )
    parser.add_argument(
        "--max-resource-jitter-max-ms", type=float, default=10.0
    )
    parser.add_argument(
        "--max-event-loop-jitter-p95-ms", type=float, default=5.0
    )
    parser.add_argument(
        "--max-event-loop-jitter-max-ms", type=float, default=10.0
    )
    return parser


def main() -> None:
    output = asyncio.run(run(build_parser().parse_args()))
    print(output)


if __name__ == "__main__":
    main()
