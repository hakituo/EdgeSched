"""MPS daemon 日志归档：清理前把证据复制进 run 目录。"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

from .mps_commands import MpsRuntimeError


def collect_log_sources(daemon_stdout: Path, daemon_stderr: Path, log_dir: Path) -> dict[str, Path]:
    """收集需要归档的日志文件：控制进程输出 + 私有 log 目录内容。"""

    sources = {
        "daemon_stdout.log": daemon_stdout,
        "daemon_stderr.log": daemon_stderr,
    }
    if log_dir.is_dir():
        for item in sorted(log_dir.iterdir()):
            sources[f"log_{item.name}"] = item
    return sources


def archive_daemon_logs(
    *,
    label: str,
    daemon_stdout: Path,
    daemon_stderr: Path,
    log_dir: Path,
    destination_root: Path,
) -> dict[str, str]:
    """把 daemon 日志复制到 run 目录；返回 ``{归档名: 目标路径}``。"""

    destination_root.mkdir(parents=True, exist_ok=True)
    archived: dict[str, str] = {}
    for name, source in collect_log_sources(
        daemon_stdout, daemon_stderr, log_dir
    ).items():
        if not source.is_file():
            continue
        target = destination_root / f"{label}_{name}"
        shutil.copyfile(source, target)
        archived[name] = str(target)
    return archived


def residual_paths(controller_pid_file: Path, private_root: Path | None) -> list[str]:
    """清理后仍残存的文件系统痕迹名称列表。"""

    residual: list[str] = []
    if controller_pid_file.exists():
        residual.append("controller_pid_file")
    if private_root is not None and private_root.exists():
        residual.append("private_root")
    return residual


def terminate_process_if_alive(process: Any) -> None:
    """结束仍存活的 control daemon 进程；无法退出时抛 ``MpsRuntimeError``。"""

    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.kill()
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            raise MpsRuntimeError("MPS 控制进程无法退出") from None


def close_streams(*handles: Any) -> None:
    """关闭 daemon 输出流，使日志缓冲落盘后再归档。"""

    for handle in handles:
        try:
            if handle is not None and not handle.closed:
                handle.close()
        except Exception:  # noqa: BLE001 - 关闭失败不应中断清理
            continue


def build_daemon_snapshot(handle: Any) -> dict[str, Any]:
    """把 daemon 句柄整理成可写入 JSONL 的证据快照。"""

    from .mps_commands import CLIENT_ENV_KEYS

    return {
        "label": handle.label,
        "run_id": handle.run_id,
        "binary": handle.binary_path,
        "private_root": str(handle.root),
        "pipe_directory": str(handle.pipe_dir),
        "log_directory": str(handle.log_dir),
        "controller_pid_file": str(handle.controller_pid_file),
        "controller_readiness": (
            handle.readiness.to_dict() if handle.readiness else None
        ),
        "post_quit_evidence": dict(handle.post_quit_evidence),
        "daemon_pid": handle.process.pid if handle.process else None,
        "daemon_returncode": handle.process.poll() if handle.process else None,
        "env_keys_injected": sorted(
            key for key in CLIENT_ENV_KEYS if key in handle.env
        ),
        "health_checks": [item.to_dict() for item in handle.health_checks],
        "archived_logs": dict(handle.archived_logs),
        "retained_artifacts": dict(handle.retained_artifacts),
        "cleanup_succeeded": handle.cleanup_succeeded,
        "cleanup_error": handle.cleanup_error,
        "cleanup_commands": [item.to_dict() for item in handle.cleanup_commands],
        "events": list(handle.events),
    }


def handle_log_summary(handle: Any) -> dict[str, Any]:
    """把归档结果整理成可写入事件的摘要。"""

    return {
        "archived_logs": dict(getattr(handle, "archived_logs", {}) or {}),
        "log_directory": str(getattr(handle, "log_dir", "") or ""),
    }
