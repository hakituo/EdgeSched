"""SM-A 探针的运行环境与 provenance 采集。

只做只读采集：Git provenance、WSL/Python 环境、direct NVML 整卡口径与
运行参数校验。正式 GPU 遥测复用 ``DirectNvmlProbe``，不启动
``nvidia-smi`` 子进程。
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]

from ..contracts import BackendEvidence, ExperimentRunConfig  # noqa: E402

MPS_NOTE = (
    "MPS 下 NVML 可能把客户端活动归到 MPS server；整卡 direct NVML 只表示整卡"
    "口径，不得称为单个 TTS/LLM/Forge 客户端的精确显存。客户端显存另行记录 "
    "PyTorch allocated/reserved。"
)


class SmCapabilityProbeError(RuntimeError):
    """SM-A 探针运行参数或前置条件不合法。"""


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def runtime_git_state() -> dict[str, Any]:
    def run(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
            shell=False,
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_git_evidence(
    args: argparse.Namespace,
    artifact_root: Path,
) -> tuple[dict[str, Any], bool]:
    """按 A1/C2 约定解析 Windows 源仓库 provenance。"""

    dirty = (
        args.source_git_dirty == "true" if args.source_git_dirty is not None else None
    )
    evidence: dict[str, Any] = {
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
        actual_sha256 = sha256_file(source)
        if (
            args.source_git_snapshot_sha256
            and actual_sha256 != args.source_git_snapshot_sha256.lower()
        ):
            raise ValueError("源码快照 SHA-256 与命令行声明不一致")
        suffix = source.suffix if source.suffix else ".bin"
        destination = artifact_root / f"source_git_snapshot{suffix}"
        if source != destination.resolve():
            shutil.copyfile(source, destination)
        evidence["snapshot"] = destination.name
        evidence["snapshot_sha256"] = actual_sha256
        snapshot_complete = True

    clean_complete = dirty is False and args.source_git_status_entry_count == 0
    dirty_complete = (
        dirty is True and bool(args.source_git_status_entry_count) and snapshot_complete
    )
    complete = bool(args.source_git_commit) and (clean_complete or dirty_complete)
    return evidence, complete


def _read_first_line(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip() or None
    except OSError:
        return None


def wsl_info() -> dict[str, Any]:
    proc_version = _read_first_line(Path("/proc/version")) or ""
    osrelease = _read_first_line(Path("/proc/sys/kernel/osrelease")) or ""
    detected = "microsoft" in proc_version.lower() or "wsl" in osrelease.lower()
    return {
        "detected": detected,
        "proc_version": proc_version or None,
        "kernel_release": osrelease or None,
        "platform": platform.platform(),
        "source": "proc_version_and_osrelease",
    }


def python_environment() -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "executable": sys.executable,
        "prefix": sys.prefix,
        "base_prefix": sys.base_prefix,
        "virtual_env": os.environ.get("VIRTUAL_ENV"),
        "in_virtualenv": sys.prefix != sys.base_prefix,
    }


def gpu_environment() -> dict[str, Any]:
    """整卡 direct NVML 采样；不启动任何 CLI 子进程。"""

    from ..gpu_telemetry import (
        DirectNvmlProbe,
        get_device_uuid,
        get_system_driver_version,
    )

    driver_version, driver_reason = get_system_driver_version()
    uuid, uuid_reason = get_device_uuid()
    backend = DirectNvmlProbe.backend_name
    try:
        sample = DirectNvmlProbe().sample(os.getpid())
    except Exception as exc:  # noqa: BLE001 - 遥测失败必须记录而不是回退 CLI
        sample = {
            "available": False,
            "backend": backend,
            "unavailable_reason": f"nvml_unavailable:{exc}",
        }
    return {
        "telemetry_backend": backend,
        "gpu_cli_subprocess_used": False,
        "nvml_sample": sample,
        "driver_version": driver_version,
        "driver_version_source": driver_reason or ("nvml" if driver_version else None),
        "driver_version_unavailable_reason": None if driver_version else driver_reason,
        "device_uuid": uuid,
        "device_uuid_unavailable_reason": None if uuid else uuid_reason,
        "note": MPS_NOTE,
    }


def build_run_config(
    *,
    run_id: str,
    args: argparse.Namespace,
    percentages: Sequence[int],
    binary_available: bool,
) -> ExperimentRunConfig:
    """构造统一运行配置头。"""

    return ExperimentRunConfig(
        experiment_name="elastic_residency_sm_capability",
        run_id=run_id,
        seed=args.seed,
        backends=(
            BackendEvidence(
                component="gpu_compute_share",
                backend_kind=(
                    "cuda_mps_client" if binary_available else "cuda_unmanaged_only"
                ),
                is_real=True,
                fallback_occurred=False,
            ),
            BackendEvidence(
                component="gpu_telemetry",
                backend_kind="direct_nvml",
                is_real=True,
                fallback_occurred=False,
            ),
        ),
        parameters={
            "percentages": list(percentages),
            "matrix_size": args.matrix_size,
            "iterations": args.iterations,
            "client_hold_seconds": args.hold_seconds,
            "client_timeout_s": args.client_timeout,
            "sm_estimate_rounding": args.sm_estimate_rounding,
            "mps_binary": args.mps_binary,
            "daemon_mode": "per_configuration",
            "mps_explicit_start_server": bool(
                getattr(args, "mps_start_server", False)
            ),
            "mps_control_channel": "interactive_stdin",
            "gpu_telemetry_backend": "direct_nvml",
            "gpu_cli_subprocess_allowed": False,
            "green_context_probe_only": True,
        },
    )


def build_manifest(
    *,
    args: argparse.Namespace,
    gpu_environment: Mapping[str, Any],
    baseline_gpu: Mapping[str, Any],
    binary_info: Mapping[str, Any],
    green_context: Mapping[str, Any],
    source_git: Mapping[str, Any],
) -> dict[str, Any]:
    """构造运行 manifest，包含 GPU、WSL、Python 环境与 Git provenance。"""

    return {
        "python": python_environment(),
        "platform": platform.platform(),
        "host_os": os.name,
        "wsl": wsl_info(),
        "gpu": dict(gpu_environment),
        "gpu_client_baseline": dict(baseline_gpu),
        "mps_binary": dict(binary_info),
        "green_context": dict(green_context),
        "git": {
            "runtime_copy": runtime_git_state(),
            "source_copy": dict(source_git),
        },
        "run_arguments": {
            "percentages": list(args.percentages),
            "matrix_size": args.matrix_size,
            "iterations": args.iterations,
            "hold_seconds": args.hold_seconds,
            "mps_binary": args.mps_binary,
        },
        "pid": os.getpid(),
    }


def parse_percentages(values: Sequence[int]) -> tuple[int, ...]:
    """校验并去重请求比例，保持原始顺序。"""

    if not values:
        raise SmCapabilityProbeError("至少需要一个 percentage")
    seen: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise SmCapabilityProbeError(f"非法 percentage: {value!r}")
        if not 1 <= value <= 100:
            raise SmCapabilityProbeError(f"percentage 必须位于 [1, 100]: {value}")
        if value in seen:
            raise SmCapabilityProbeError(f"重复的 percentage: {value}")
        seen.append(value)
    return tuple(seen)
