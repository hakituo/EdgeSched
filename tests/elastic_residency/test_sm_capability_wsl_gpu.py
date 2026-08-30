"""SM-A 的 WSL2 GPU 集成入口（需要真实 CUDA + MPS，默认不运行）。

Windows 侧没有 CUDA/MPS 时，普通单元测试不能“因为没有后端”而假通过。
本文件必须显式开启：

```bash
export EDGESCHED_RUN_WSL_GPU_TESTS=1
python -m pytest tests/elastic_residency/test_sm_capability_wsl_gpu.py -q
```

若 MPS binary 存在，控制协议实现错误必须让本文件**失败**：
仅生成结构化 ``NO_GO`` 不足以说明实现正确。硬门判定逻辑位于
``sm_wsl_gates.py``，并有独立单元测试覆盖。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.elastic_residency.sm_wsl_gates import (
    assert_affinity_query_present,
    assert_cleanup_closed,
    assert_controller_ready,
    assert_no_implementation_fault,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

pytestmark = [
    pytest.mark.wsl_gpu,
    pytest.mark.skipif(
        os.environ.get("EDGESCHED_RUN_WSL_GPU_TESTS") != "1",
        reason="需要 WSL2 + CUDA + MPS；设置 EDGESCHED_RUN_WSL_GPU_TESTS=1 后运行",
    ),
]


def _run_probe(tmp_path: Path, run_id: str) -> dict[str, object]:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.elastic_residency.sm.sm_capability",
            "--run-id",
            run_id,
            "--output-root",
            str(tmp_path),
            "--percentages",
            "100",
            "75",
            "50",
            "25",
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
        shell=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _records(root: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (root / "iterations.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _mps_available(root: Path) -> bool:
    return bool(_read_json(root / "manifest.json")["mps_binary"]["available"])


def _controlled_client_payloads(root: Path) -> list[dict[str, Any]]:
    return [
        _read_json(path)
        for path in sorted((root / "client_stdio").glob("*.json"))
        if not path.name.startswith("baseline_unmanaged")
    ]


def test_sm_a_probe_writes_complete_evidence(tmp_path: Path) -> None:
    outcome = _run_probe(tmp_path, "sm_a_wsl_gpu_probe")
    root = Path(str(outcome["artifact_root"]))
    for name in (
        "run_config.json",
        "manifest.json",
        "iterations.jsonl",
        "events.jsonl",
        "summary.json",
        "summary.csv",
        "failures.jsonl",
    ):
        assert (root / name).is_file(), name

    manifest = _read_json(root / "manifest.json")
    assert manifest["gpu"]["telemetry_backend"] == "direct_nvml"
    assert manifest["gpu"]["gpu_cli_subprocess_used"] is False
    assert manifest["mps_binary"]["binary"] == "nvidia-cuda-mps-control"

    summary = _read_json(root / "summary.json")
    assert summary["report_complete"] is True
    assert summary["verdict"] in {"GO", "NO_GO"}
    assert len(summary["sm_evidence"]) == 5  # 基线 + 4 个 percentage
    assert summary["physical_total_sms"] is not None


def test_control_protocol_must_work_when_mps_binary_exists(tmp_path: Path) -> None:
    """MPS binary 存在时，白名单以外的任何失败都必须直接判失败。"""

    outcome = _run_probe(tmp_path, "sm_a_wsl_gpu_protocol")
    root = Path(str(outcome["artifact_root"]))
    if not _mps_available(root):
        pytest.skip("当前环境没有 nvidia-cuda-mps-control，无法验证控制协议")
    summary = _read_json(root / "summary.json")
    assert_no_implementation_fault(summary)
    assert summary["fallback_count"] == 0
    assert summary["cleanup_failures"] == []


def test_cleanup_is_closed_when_mps_binary_exists(tmp_path: Path) -> None:
    outcome = _run_probe(tmp_path, "sm_a_wsl_gpu_cleanup")
    root = Path(str(outcome["artifact_root"]))
    if not _mps_available(root):
        pytest.skip("当前环境没有 nvidia-cuda-mps-control，无法验证清理")
    assert_cleanup_closed(_records(root))


def test_affinity_query_evidence_is_present(tmp_path: Path) -> None:
    """只有 MPS binary 存在且受控客户端实际运行时才要求 driver_query。"""

    outcome = _run_probe(tmp_path, "sm_a_wsl_gpu_affinity")
    root = Path(str(outcome["artifact_root"]))
    if not _mps_available(root):
        pytest.skip("当前环境没有 nvidia-cuda-mps-control，无受控客户端运行")
    assert_affinity_query_present(_controlled_client_payloads(root))


def test_controller_readiness_evidence_is_recorded(tmp_path: Path) -> None:
    outcome = _run_probe(tmp_path, "sm_a_wsl_gpu_readiness")
    root = Path(str(outcome["artifact_root"]))
    if not _mps_available(root):
        pytest.skip("当前环境没有 nvidia-cuda-mps-control")
    assert_controller_ready(_records(root))
