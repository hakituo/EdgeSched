"""small 9B 离线筛选产物适配与正式共置复算的门禁测试。"""

from __future__ import annotations

from typing import Any

import pytest

from experiments.elastic_residency.analyze_colocated_results import analyze_colocated
from experiments.elastic_residency.small_screen_view import (
    LEGACY,
    SMALL,
    detect_schema,
    load_screen,
    to_legacy_view,
)

IDLE_MB = 952.30859375
B_LLM_MB = 5335.40625
GPU_TOTAL_MB = 8151.0


def _small_screen() -> dict[str, Any]:
    usage = {
        "K18": 3625.09765625,
        "K20": 3871.09765625,
        "K24": 4383.09765625,
        "K30": 5133.09765625,
    }
    return {
        "analysis_name": "small_stage_screen",
        "provisional_g_safe": {
            "value_mb": 915.015625,
            "components": {
                "idle_gpu_baseline_high_water_mb": 915.015625,
                "torch_allocator_slack_mb": 501.2822265625,
                "llm_repeat_run_drift_mb": 0.0,
                "colocated_positive_residual_mb": None,
            },
            "pending_components": ["colocated_positive_residual_mb"],
        },
        "text_llm_tts_gpu_only": {
            "512": {
                "T0": {
                    "h_layers": 20,
                    "full_feasible": False,
                    "budget_for_m_l_mb": 3980.94921875,
                    "m_l_usage_mb": usage,
                },
                "T3": {
                    "h_layers": 30,
                    "full_feasible": False,
                    "budget_for_m_l_mb": 5256.88671875,
                    "m_l_usage_mb": usage,
                },
            }
        },
    }


def _legacy_screen() -> dict[str, Any]:
    return {
        "g_safe": {"value_mb": 501.0, "components": {"torch_allocator_slack_mb": 501.0}},
        "per_context": {
            "4096": {
                "h_tts": {"T0": {"h_layers": 20, "full_feasible": False}},
                "h_d": {"h_layers": 4, "full_feasible": False},
            }
        },
    }


def _record(tts_config: str, p_mb: float, llm_config: str = "K24") -> dict[str, Any]:
    isolated = 3255.03515625 if tts_config == "T0" else 1979.09765625
    return {
        "context_tokens": 512,
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
            "gpu_card_used_mb": IDLE_MB,
            "gpu_telemetry_backend": "direct_nvml",
            "gpu_cli_subprocess_used": False,
        },
        "colocated_metrics": {
            "b_llm_mb": B_LLM_MB,
            "p_colocated_mb": p_mb,
            "gpu_total_mb": GPU_TOTAL_MB,
            "tts_isolated_increment_mb": isolated,
            "capacity_admissible": True,
        },
    }


def _colocated(llm_config: str = "K24") -> dict[str, Any]:
    records = [
        *[_record("T0", 7619.40625, llm_config) for _ in range(5)],
        *[_record("T3", 7483.40625, llm_config) for _ in range(5)],
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
        "run_id": "small_9b_tts_compat_test",
        "git_provenance_complete": True,
        "git": {
            "source_copy": {
                "commit": "e0f1779d2e122d7b915dc8ea4d10b761be589bbc",
                "dirty": False,
                "status_entry_count": 0,
            }
        },
    }


def test_detect_schema_discriminates_small_and_legacy() -> None:
    assert detect_schema(_small_screen()) == SMALL
    assert detect_schema(_legacy_screen()) == LEGACY
    with pytest.raises(ValueError, match="无法识别"):
        detect_schema({"unknown": True})


def test_to_legacy_view_keeps_profiled_configs_and_empty_h_d() -> None:
    view = to_legacy_view(_small_screen())
    assert view["g_safe"]["value_mb"] == 915.015625
    context = view["per_context"]["512"]
    assert context["profiled_llm_configs"] == ["K18", "K20", "K24", "K30"]
    assert context["h_tts"]["T0"]["h_layers"] == 20
    # small screen 没有 H_D 档位，必须保持 None，不能填 0。
    assert context["h_d"] == {"h_layers": None, "full_feasible": False}


def test_load_screen_auto_returns_view_and_schema() -> None:
    screen, schema = load_screen(_small_screen())
    assert schema == SMALL
    assert "per_context" in screen
    legacy = _legacy_screen()
    legacy_screen, legacy_schema = load_screen(legacy)
    assert legacy_schema == LEGACY
    # legacy 结构原样返回，不做映射。
    assert legacy_screen == legacy


def test_small_formal_matrix_freezes_h_tts_and_records_residual() -> None:
    screen, schema = load_screen(_small_screen())
    result = analyze_colocated(
        screen=screen,
        colocated=_colocated(),
        manifest=_manifest(),
        screen_schema=schema,
    )

    assert result["screen_schema"] == SMALL
    assert result["status"] == "h_tts_frozen_h_d_pending"
    # T3 共置增量 2148 大于隔离先验 1979.09765625，正残差必须被登记进 G_safe 分量。
    residual = result["g_safe"]["components"]["colocated_positive_residual_mb"]
    assert residual == pytest.approx(168.90234375)
    assert result["g_safe"]["value_mb"] == pytest.approx(915.015625)
    assert result["h_tts"]["512"]["T3"]["validated_runs"] == 5
    assert result["h_d"]["512"]["screened_h_layers"] is None


def test_small_candidate_may_exceed_screen_boundary_with_flag() -> None:
    screen, schema = load_screen(_small_screen())
    result = analyze_colocated(
        screen=screen,
        colocated=_colocated(),
        manifest=_manifest(),
        screen_schema=schema,
    )
    t0 = result["candidates"]["C512_K24_T0"]
    t3 = result["candidates"]["C512_K24_T3"]
    assert t0["screen_boundary"] == "K20"
    assert t0["measured_beyond_screen_boundary"] is True
    assert t3["screen_boundary"] == "K30"
    assert t3["measured_beyond_screen_boundary"] is True
    assert t0["physical_headroom_min_mb"] == pytest.approx(531.59375)


def test_small_rejects_unprofiled_candidate() -> None:
    screen, schema = load_screen(_small_screen())
    with pytest.raises(ValueError, match="已画像档位"):
        analyze_colocated(
            screen=screen,
            colocated=_colocated(llm_config="K33"),
            manifest=_manifest(),
            screen_schema=schema,
        )


def test_h_tts_keys_disambiguate_multiple_llm_configs() -> None:
    """同一 TTS 配置下多个 llm 档都必须保留，后写不得覆盖先写。"""
    screen, schema = load_screen(_small_screen())
    colocated = _colocated()
    colocated["records"].extend(_record("T0", 5435.41, "K18") for _ in range(5))
    colocated["actual_records"] = colocated["expected_records"] = 15
    result = analyze_colocated(
        screen=screen, colocated=colocated, manifest=_manifest(), screen_schema=schema
    )
    keys = set(result["h_tts"]["512"])
    assert {"T0_at_K24", "T0_at_K18", "T3"} <= keys
