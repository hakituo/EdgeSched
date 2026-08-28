"""TTS profiling 的可复现环境、拓扑与统计工具。"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        while block := file_obj.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def model_tree_hash(model_path: Path) -> tuple[str, list[dict[str, Any]]]:
    """记录每个模型文件哈希，并对有序清单计算目录哈希。"""

    if model_path.is_file():
        file_hash = sha256_file(model_path)
        records = [
            {
                "path": model_path.name,
                "size": model_path.stat().st_size,
                "sha256": file_hash,
            }
        ]
    else:
        records = []
        for path in sorted(item for item in model_path.rglob("*") if item.is_file()):
            if ".cache" in path.parts:
                continue
            records.append(
                {
                    "path": path.relative_to(model_path).as_posix(),
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest(), records


def _run_command(command: list[str], *, cwd: Path | None = None) -> str:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return completed.stdout.strip()


def collect_git_state(git_root: Path | None) -> dict[str, Any]:
    if git_root is None or not git_root.exists():
        return {
            "available": False,
            "commit": "",
            "dirty": None,
            "dirty_file_count": None,
            "status_sha256": "",
        }
    safe = f"safe.directory={git_root}"
    commit = _run_command(
        ["git", "-c", safe, "rev-parse", "HEAD"], cwd=git_root
    )
    status = _run_command(
        ["git", "-c", safe, "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=git_root,
    )
    status_lines = [line for line in status.splitlines() if line]
    return {
        "available": bool(commit),
        "commit": commit,
        "dirty": bool(status_lines) if commit else None,
        "dirty_file_count": len(status_lines) if commit else None,
        "status_sha256": hashlib.sha256(status.encode()).hexdigest() if commit else "",
    }


def collect_gpu_info() -> dict[str, Any]:
    output = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    if not output:
        return {"available": False, "name": "", "driver_version": "", "memory_total_mb": 0}
    first = output.splitlines()[0]
    parts = [part.strip() for part in first.split(",")]
    if len(parts) != 3:
        return {"available": False, "raw": first}
    return {
        "available": True,
        "name": parts[0],
        "driver_version": parts[1],
        "memory_total_mb": int(parts[2]),
    }


def collect_linux_memory() -> dict[str, int]:
    values: dict[str, int] = {}
    meminfo = Path("/proc/meminfo")
    if not meminfo.is_file():
        return values
    wanted = {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        key, _, remainder = line.partition(":")
        if key not in wanted:
            continue
        amount_kib = int(remainder.strip().split()[0])
        values[f"{key.lower()}_bytes"] = amount_kib * 1024
    return values


def collect_cpu_topology() -> dict[str, Any]:
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for cpu_dir in sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*")):
        cpu_id = int(cpu_dir.name[3:])
        topology = cpu_dir / "topology"
        try:
            core_id = int((topology / "core_id").read_text().strip())
            package_id = int((topology / "physical_package_id").read_text().strip())
        except (FileNotFoundError, ValueError):
            continue
        groups[(package_id, core_id)].append(cpu_id)
    physical = [min(cpus) for _, cpus in sorted(groups.items())]
    siblings = [sorted(cpus) for _, cpus in sorted(groups.items())]
    return {
        "logical_cpu_count": os.cpu_count() or 0,
        "physical_core_count": len(groups),
        "physical_core_representatives": physical,
        "sibling_groups": siblings,
    }


def select_physical_cpu_affinity(core_count: int) -> tuple[int, ...]:
    if core_count < 1:
        raise ValueError("物理核心数必须大于 0")
    representatives = collect_cpu_topology().get("physical_core_representatives", [])
    if len(representatives) < core_count:
        raise ValueError(
            f"只发现 {len(representatives)} 个物理核，无法绑定 {core_count} 个"
        )
    return tuple(representatives[:core_count])


def collect_runtime_platform(git_root: Path | None) -> dict[str, Any]:
    return {
        "gpu": collect_gpu_info(),
        "linux_memory": collect_linux_memory(),
        "cpu_topology": collect_cpu_topology(),
        "git": collect_git_state(git_root),
    }


def percentile(values: Iterable[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def linear_fit(xs: Iterable[float], ys: Iterable[float]) -> dict[str, float]:
    x_values = [float(value) for value in xs]
    y_values = [float(value) for value in ys]
    if len(x_values) != len(y_values) or not x_values:
        raise ValueError("线性拟合要求长度相等且非空")
    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)
    denominator = sum((value - x_mean) ** 2 for value in x_values)
    slope = (
        sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
        / denominator
        if denominator
        else 0.0
    )
    intercept = y_mean - slope * x_mean
    predictions = [intercept + slope * value for value in x_values]
    residual = sum((actual - predicted) ** 2 for actual, predicted in zip(y_values, predictions))
    total = sum((actual - y_mean) ** 2 for actual in y_values)
    r_squared = 1.0 - residual / total if total else 1.0
    return {"intercept": intercept, "slope_per_char": slope, "r_squared": r_squared}
