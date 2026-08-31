"""真实共置结果离线复算的纯逻辑门禁。"""

from __future__ import annotations

from typing import Any

import pytest

from experiments.elastic_residency.analyze_colocated_results import (
    analyze_colocated,
)


def _screen() -> dict[str, Any]:
    return {
        "analysis_name": "screen_test",
        "g_safe": {
            "value_mb": 501.2822265625,
            "components": {
                "idle_gpu_baseline_high_water_mb": 487.6953125,
                "torch_allocator_slack_mb": 501.2822265625,
                "llm_repeat_run_drift_mb": 45.0,
                "colocated_positive_residual_mb": None,
            },
        },
        "per_context": {
            "4096": {
                "h_tts": {
                    "T0": {"h_layers": 20, "full_feasible": False},
                    "T3": {"h_layers": 32, "full_feasible": True},
                },
                "h_d": {"h_layers": 4, "full_feasible": False},
            }
        },
    }


def _record(llm_config: str, tts_config: str, p_mb: float) -> dict[str, Any]:
    isolated = 3255.0 if tts_config == "T0" else 1979.0
    b_llm = p_mb - (2252.0 if tts_config == "T0" else 1790.0)
    return {
        "context_tokens": 4096,
        "llm_config": llm_config,
        "tts_config": tts_config,
        "passed": True,
        "backend_kind": "real_llama_cpp_gguf_plus_qwen3_tts",
        "fallback_occurred": False,
        "acceptance_checks": {"all": True},
        "runtime_evidence": {
            "checks": {
                "gpu_periodic_samples_fresh": True,
                "gpu_capacity_snapshots_fresh": True,
                "swap_growth_within_tolerance": True,
            }
        },
        "memory_baseline": {
            "gpu_card_used_mb": 475.6953125,
            "gpu_telemetry_backend": "direct_nvml",
            "gpu_cli_subprocess_used": False,
        },
        "colocated_metrics": {
            "b_llm_mb": b_llm,
            "p_colocated_mb": p_mb,
            "gpu_total_mb": 8151.0,
            "tts_isolated_increment_mb": isolated,
            # 旧公式把绝对峰值再次叠加完整 G_safe，FULL+T3 会误判。
            "capacity_admissible": p_mb + 501.2822265625 <= 8151.0,
        },
    }


def _colocated() -> dict[str, Any]:
    records = [
        *[_record("K20", "T0", 6686.79296875) for _ in range(5)],
        *[_record("FULL", "T3", 7924.79296875) for _ in range(5)],
    ]
    return {
        "passed": True,
        "failure_count": 0,
        "actual_records": 10,
        "expected_records": 10,
        "evidence_scope": "llm_tts_colocated_formal_repeated_matrix",
        "records": records,
    }


def _manifest() -> dict[str, Any]:
    return {
        "run_id": "colocated_test",
        "git_provenance_complete": True,
        "git": {
            "source_copy": {
                "commit": "d1009158ed03a906933106b500d91d34a1d3ad3c",
                "dirty": False,
                "status_entry_count": 0,
            }
        },
    }


def test_reanalysis_corrects_idle_double_count_and_freezes_h_tts() -> None:
    result = analyze_colocated(
        screen=_screen(),
        colocated=_colocated(),
        manifest=_manifest(),
    )

    assert result["status"] == "h_tts_frozen_h_d_pending"
    assert result["g_safe"]["value_mb"] == pytest.approx(501.2822265625)
    assert result["g_safe"]["components"]["colocated_positive_residual_mb"] == 0
    evidence = result["evidence_validation"]
    assert evidence["validated_records"] == 10
    assert evidence["legacy_double_counted_capacity_false_records"] == 5
    assert evidence["rerun_required"] is False

    full = result["candidates"]["C4096_FULL_T3"]
    assert full["all_capacity_admissible"] is True
    assert full["physical_headroom_min_mb"] == pytest.approx(226.20703125)
    assert full["safe_capacity_headroom_min_mb"] == pytest.approx(200.6201171875)
    assert result["h_tts"]["4096"]["T3"]["status"] == "frozen_real_colocated"
    assert result["h_d"]["4096"]["status"] == "pending_real_llm_forge_colocation"


def test_reanalysis_rejects_fallback_record() -> None:
    colocated = _colocated()
    colocated["records"][0]["fallback_occurred"] = True
    with pytest.raises(ValueError, match=r"真实 LLM\+TTS"):
        analyze_colocated(
            screen=_screen(),
            colocated=colocated,
            manifest=_manifest(),
        )
