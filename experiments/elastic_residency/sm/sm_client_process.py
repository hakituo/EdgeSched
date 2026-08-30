"""以独立子进程运行 SM 客户端探针并采集进程证据。

每个静态配置一个全新子进程；父进程在客户端存活期间轮询
``get_client_list``，用来证明客户端确实连接到了 MPS daemon。
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .mps_daemon import MpsDaemonHandle
from .sm_profiles import GpuComputeBackend

CLIENT_MODULE = "experiments.elastic_residency.sm.sm_client_probe"


def client_artifact_dir(artifact_root: Path) -> Path:
    path = artifact_root / "client_stdio"
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_client_argv(
    *,
    python_executable: str,
    phase_id: str,
    expected_backend: GpuComputeBackend,
    expected_percentage: int | None,
    require_env: Sequence[str],
    output_json: Path,
    matrix_size: int,
    iterations: int,
    hold_seconds: float,
) -> list[str]:
    argv = [
        python_executable,
        "-m",
        CLIENT_MODULE,
        "--phase-id",
        phase_id,
        "--expected-backend",
        expected_backend.value,
        "--output-json",
        str(output_json),
        "--matrix-size",
        str(matrix_size),
        "--iterations",
        str(iterations),
        "--hold-seconds",
        str(hold_seconds),
    ]
    if expected_percentage is not None:
        argv += ["--expected-percentage", str(expected_percentage)]
    if require_env:
        argv += ["--require-env", *require_env]
    return argv


def run_client_probe(
    *,
    artifact_root: Path,
    project_root: Path,
    phase_id: str,
    env: Mapping[str, str],
    expected_backend: GpuComputeBackend,
    expected_percentage: int | None,
    require_env: Sequence[str],
    python_executable: str,
    timeout_s: float,
    hold_seconds: float,
    matrix_size: int,
    iterations: int,
    daemon_handle: MpsDaemonHandle | None = None,
    poll_interval_s: float = 0.5,
    max_poll_samples: int = 60,
) -> dict[str, Any]:
    """启动客户端子进程，返回结构化运行证据。"""

    artifacts = client_artifact_dir(artifact_root)
    stdout_path = artifacts / f"{phase_id}.stdout.log"
    stderr_path = artifacts / f"{phase_id}.stderr.log"
    output_json = artifacts / f"{phase_id}.json"
    argv = build_client_argv(
        python_executable=python_executable,
        phase_id=phase_id,
        expected_backend=expected_backend,
        expected_percentage=expected_percentage,
        require_env=require_env,
        output_json=output_json,
        matrix_size=matrix_size,
        iterations=iterations,
        hold_seconds=hold_seconds,
    )

    observed_pids: set[int] = set()
    poll_samples = 0
    timed_out = False
    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_handle:
        process = subprocess.Popen(  # noqa: S603 - 固定参数列表，无 shell
            argv,
            env=dict(env),
            cwd=str(project_root),
            stdout=stdout_handle,
            stderr=stderr_handle,
            stdin=subprocess.DEVNULL,
            shell=False,
        )
        deadline = time.monotonic() + timeout_s
        while process.poll() is None:
            if daemon_handle is not None and poll_samples < max_poll_samples:
                for pid in daemon_handle.query_clients():
                    observed_pids.add(int(pid))
                poll_samples += 1
            if time.monotonic() >= deadline:
                timed_out = True
                process.kill()
                break
            time.sleep(max(0.05, poll_interval_s))
        returncode = process.wait()

    payload: dict[str, Any] | None = None
    parse_error: str | None = None
    if output_json.is_file():
        try:
            payload = json.loads(output_json.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            parse_error = f"{exc.__class__.__name__}: {exc}"
    return {
        "phase_id": phase_id,
        "argv": list(argv),
        "returncode": returncode,
        "timed_out": timed_out,
        "stdout_path": str(stdout_path.relative_to(artifact_root)),
        "stderr_path": str(stderr_path.relative_to(artifact_root)),
        "output_json_path": str(output_json.relative_to(artifact_root)),
        "payload": payload,
        "json_parse_error": parse_error,
        "mps_client_pids_observed": sorted(observed_pids),
        "mps_client_pid_observed": int(process.pid) in observed_pids,
        "mps_client_pid": process.pid,
        "poll_samples": poll_samples,
    }
