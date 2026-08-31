"""LLM+Forge 父进程 artifact 编排测试（不启动真实后端）。"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from experiments.elastic_residency import llm_forge_parent as parent
from experiments.elastic_residency.llm_forge_colocated import build_parser


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_parent_builds_complete_smoke_artifact(tmp_path: Path, monkeypatch) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"gguf")
    screen_path = tmp_path / "screen.json"
    forge_path = tmp_path / "forge.json"
    output = tmp_path / "results"
    _write_json(
        screen_path,
        {
            "gpu_total_mb": 8151.0,
            "g_safe": {
                "value_mb": 501.0,
                "components": {"idle_gpu_baseline_high_water_mb": 487.0},
            },
            "forge": {
                "run_id": "forge_v1",
                "m_d_incremental_estimate_mb": 5484.0,
                "global_idle_high_water_mb": 487.0,
            },
            "per_context": {
                "512": {
                    "m_l_mb": {"K6": 2117.0},
                    "h_d": {
                        "h_layers": 6,
                        "full_feasible": False,
                        "m_d_mb": 5484.0,
                    },
                }
            },
        },
    )
    _write_json(
        forge_path,
        {
            "run_id": "forge_v1",
            "passed": True,
            "acceptance_checks": {"all": True},
            "resource_sampler": {
                "gpu_telemetry_backend": "direct_nvml",
                "gpu_cli_subprocess_used": False,
            },
            "global_idle_high_water_mb": 487.0,
            "forge_resident_baseline_mb": 711.0,
            "m_d_incremental_estimate_mb": 5484.0,
        },
    )
    args = build_parser().parse_args(
        [
            "--llm-model",
            str(model),
            "--screen-analysis",
            str(screen_path),
            "--forge-summary",
            str(forge_path),
            "--candidates",
            "512:K6",
            "--runs",
            "1",
            "--inter-run-cooldown",
            "0",
            "--run-id",
            "parent_smoke",
            "--output-root",
            str(output),
            "--source-git-commit",
            "c" * 40,
            "--source-git-dirty",
            "false",
            "--source-git-status-entry-count",
            "0",
        ]
    )
    monkeypatch.setattr(
        parent,
        "_discover_model_metadata",
        lambda _path: {"block_count": 32, "context_length": 32768},
    )
    monkeypatch.setattr(parent, "sha256_file", lambda _path: "a" * 64)
    monkeypatch.setattr(parent.platform, "platform", lambda: "test-platform")
    monkeypatch.setattr(parent.platform, "python_version", lambda: "3.test")
    monkeypatch.setattr(
        parent,
        "_git_source",
        lambda _args, _root: (
            {"commit": "c" * 40, "dirty": False, "status_entry_count": 0},
            True,
        ),
    )

    def fake_get(url: str, _timeout: float):
        if url.endswith("sd-models"):
            return [{"title": "model.safetensors"}]
        return {"sd_model_checkpoint": "model.safetensors"}

    monkeypatch.setattr(parent, "forge_get", fake_get)
    monkeypatch.setattr(parent, "forge_post", lambda *_args, **_kwargs: {"images": ["x"]})

    def fake_run(command: list[str], **_kwargs):
        child_json = Path(command[command.index("--child-output-json") + 1])
        image = Path(command[command.index("--child-output-image") + 1])
        image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * 2048)
        _write_json(
            child_json,
            {
                "context_tokens": 512,
                "llm_config": "K6",
                "run_index": 1,
                "passed": True,
                "acceptance_checks": {"all": True},
                "child_started_monotonic_ns": 1,
                "child_started_wall_iso": "2026-08-31T00:00:00+00:00",
                "timeline": [],
            },
        )
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(parent.subprocess, "run", fake_run)
    root = parent.run_parent(args)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert summary["passed"] is True
    assert summary["actual_records"] == 1
    assert summary["evidence_scope"] == "llm_forge_colocated_smoke_or_pilot"
    assert manifest["forge"]["active_checkpoint"] == "model.safetensors"
    assert (root / "failures.jsonl").read_text(encoding="utf-8") == ""
