"""LLM+Forge 正式矩阵冻结分析门禁。"""
from __future__ import annotations

from typing import Any

import pytest

from experiments.elastic_residency.analyze_forge_colocated_results import (
    analyze_forge_colocated,
)


def _screen() -> dict[str, Any]:
    contexts = {}
    for context, config, layers, m_l in (
        (512, "K6", 6, 2117.0),
        (2048, "K4", 4, 1899.0),
        (4096, "K4", 4, 1911.0),
    ):
        contexts[str(context)] = {
            "h_d": {"h_layers": layers, "full_feasible": False},
            "m_l_mb": {config: m_l},
        }
    return {"per_context": contexts}


def _tts_freeze() -> dict[str, Any]:
    return {
        "g_safe": {
            "value_mb": 501.0,
            "components": {
                "idle_gpu_baseline_high_water_mb": 487.0,
                "torch_allocator_slack_mb": 501.0,
                "llm_repeat_run_drift_mb": 45.0,
                "colocated_positive_residual_mb": 0.0,
            },
        },
        "h_tts": {
            "512": {
                "T0": {"safe_capacity_headroom_min_mb": 900.0},
                "T3": {"safe_capacity_headroom_min_mb": 300.0},
            }
        },
    }


def _record(context: int, config: str, residual: float = 0.0) -> dict[str, Any]:
    snapshots = {
        "gpu_telemetry_backend": "direct_nvml",
        "gpu_cli_subprocess_used": False,
    }
    llm_m = {512: 2117.0, 2048: 1899.0, 4096: 1911.0}[context]
    p_mb = 487.0 + llm_m + 5484.0 + residual
    return {
        "context_tokens": context,
        "llm_config": config,
        "passed": True,
        "acceptance_checks": {"all": True},
        "backend_kind": "real_llama_cpp_gguf_plus_forge_http_api",
        "fallback_occurred": False,
        "runtime_evidence": {"checks": {"fresh": True, "swap": True}},
        "forge_image": {"size_bytes": 2048, "sha256": "b" * 64},
        "memory_forge_resident": dict(snapshots),
        "memory_b_llm": dict(snapshots),
        "memory_post_forge": dict(snapshots),
        "colocated_metrics": {
            "global_idle_reference_mb": 487.0,
            "b_llm_mb": 2800.0,
            "p_colocated_mb": p_mb,
            "gpu_total_mb": 8151.0,
            "llm_isolated_increment_mb": llm_m,
            "forge_isolated_increment_mb": 5484.0,
            "colocated_positive_residual_mb": residual,
        },
    }


def _summary(residual: float = 0.0) -> dict[str, Any]:
    records = []
    for context, config in ((512, "K6"), (2048, "K4"), (4096, "K4")):
        records.extend(_record(context, config, residual) for _ in range(5))
    for ordinal, record in enumerate(records, start=1):
        record["run_index"] = ((ordinal - 1) % 5) + 1
        record["matrix_execution_ordinal"] = ordinal
    return {
        "passed": True,
        "failure_count": 0,
        "expected_records": 15,
        "actual_records": 15,
        "evidence_scope": "llm_forge_colocated_formal_repeated_matrix",
        "records": records,
    }


def _manifest() -> dict[str, Any]:
    return {
        "run_id": "forge_colocated_test",
        "git_provenance_complete": True,
        "git": {
            "source_copy": {
                "commit": "a" * 40,
                "dirty": False,
                "status_entry_count": 0,
            }
        },
    }


def test_analysis_freezes_h_d_and_preserves_h_tts() -> None:
    result = analyze_forge_colocated(
        screen=_screen(),
        tts_freeze=_tts_freeze(),
        colocated=_summary(),
        manifest=_manifest(),
    )
    assert result["status"] == "g_safe_h_tts_h_d_frozen"
    assert result["g_safe"]["value_mb"] == 501.0
    assert result["h_d"]["512"]["llm_config"] == "K6"
    assert result["h_d"]["4096"]["status"] == "frozen_real_colocated"
    assert result["evidence_validation"]["validated_records"] == 15


def test_analysis_rejects_when_new_residual_breaks_capacity() -> None:
    with pytest.raises(ValueError, match="不可行"):
        analyze_forge_colocated(
            screen=_screen(),
            tts_freeze=_tts_freeze(),
            colocated=_summary(residual=200.0),
            manifest=_manifest(),
        )


def test_analysis_rejects_fallback() -> None:
    summary = _summary()
    summary["records"][0]["fallback_occurred"] = True
    with pytest.raises(ValueError, match="fallback"):
        analyze_forge_colocated(
            screen=_screen(),
            tts_freeze=_tts_freeze(),
            colocated=summary,
            manifest=_manifest(),
        )
