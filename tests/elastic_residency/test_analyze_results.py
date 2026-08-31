"""analyze_results 的纯逻辑测试（不依赖 psutil/torch）。"""

from __future__ import annotations

from typing import Any

from experiments.elastic_residency.analyze_results import (
    _config_layer_count,
    analyze,
    compute_g_safe,
    compute_h,
    extract_forge_profile,
    extract_llm_memory_curve,
    extract_tts_profiles,
    suggest_refinements,
)
from experiments.elastic_residency.analysis_input_merge import (
    merge_llm_curve_summaries,
)


def _c2_summary() -> dict[str, Any]:
    """最小 C2 形状：单上下文、四个配置。"""

    def _config(mean: float) -> dict[str, Any]:
        return {"gpu_delta_mb": {"mean": mean, "max": mean + 4.0, "min": mean}}

    return {
        "run_id": "c2_test",
        "passed": True,
        "by_context_config": {
            "512": {
                "K0": _config(127.0),
                "K16": _config(3373.0),
                "K24": _config(4383.0),
                "FULL": _config(5543.0),
            },
        },
    }


def _c3_summary() -> dict[str, Any]:
    return {
        "run_id": "c3_test",
        "passed": True,
        "by_context_config": {
            "512": {
                "K8": {
                    "gpu_delta_mb": {"mean": 2000.0, "max": 2004.0, "min": 1998.0}
                },
                "K20": {
                    "gpu_delta_mb": {"mean": 3900.0, "max": 3904.0, "min": 3898.0}
                },
            },
        },
    }


def _b2_summary() -> dict[str, Any]:
    return {
        "run_id": "b2_test",
        "by_config": {
            "T0": {
                "max_card_delta_mb": 3284.0,
                "torch_peak_allocated_mb": 2616.7,
                "torch_peak_reserved_mb": 3118.0,
            },
            "T3": {
                "max_card_delta_mb": 1980.0,
                "torch_peak_allocated_mb": 1788.5,
                "torch_peak_reserved_mb": 1816.0,
            },
        },
    }


def _forge_summary() -> dict[str, Any]:
    return {
        "run_id": "forge_test",
        "peak_card_used_mb": {"max": 5904.0},
        "global_idle_high_water_mb": 314.0,
    }


def test_extract_llm_memory_curve_reads_by_context() -> None:
    curve = extract_llm_memory_curve(_c2_summary())
    assert set(curve) == {"512"}
    assert curve["512"]["K24"]["mean"] == 4383.0
    assert curve["512"]["FULL"]["max"] == 5547.0


def test_merge_llm_curves_requires_complete_non_overlapping_inputs() -> None:
    merged = merge_llm_curve_summaries(
        _c2_summary(), _c3_summary(), expected_config_count=6
    )
    assert merged["source_run_ids"] == ["c2_test", "c3_test"]
    assert set(merged["by_context_config"]["512"]) == {
        "K0",
        "K8",
        "K16",
        "K20",
        "K24",
        "FULL",
    }

    incomplete = _c3_summary()
    incomplete["by_context_config"]["512"].pop("K20")
    try:
        merge_llm_curve_summaries(
            _c2_summary(), incomplete, expected_config_count=6
        )
    except ValueError as exc:
        assert "预期 6" in str(exc)
    else:  # pragma: no cover - 防止门禁失效
        raise AssertionError("不完整 16 档输入必须被拒绝")

    duplicate = _c3_summary()
    duplicate["by_context_config"]["512"]["K16"] = duplicate[
        "by_context_config"
    ]["512"]["K8"]
    try:
        merge_llm_curve_summaries(_c2_summary(), duplicate)
    except ValueError as exc:
        assert "重复" in str(exc)
    else:  # pragma: no cover - 防止门禁失效
        raise AssertionError("重复配置必须被拒绝")


def test_extract_tts_profiles_computes_allocator_slack() -> None:
    profiles = extract_tts_profiles(_b2_summary())
    assert profiles["T0"]["max_card_delta_mb"] == 3284.0
    assert abs(profiles["T0"]["allocator_slack_mb"] - 501.3) < 1e-6
    assert abs(profiles["T3"]["allocator_slack_mb"] - 27.5) < 1e-6


def test_extract_forge_profile_modes() -> None:
    reprofiled = extract_forge_profile(_forge_summary(), None, None)
    assert reprofiled["source"] == "forge_reprofile"
    assert abs(reprofiled["m_d_incremental_estimate_mb"] - 5590.0) < 1e-6

    legacy = extract_forge_profile(None, 5904.0, 314.0)
    assert legacy["source"] == "legacy_estimate"
    assert abs(legacy["m_d_incremental_estimate_mb"] - 5590.0) < 1e-6

    missing = extract_forge_profile(None, None, None)
    assert missing["source"] == "missing"
    assert missing["m_d_incremental_estimate_mb"] is None


def test_compute_g_safe_takes_max_of_available_components() -> None:
    curve = extract_llm_memory_curve(_c2_summary())
    profiles = extract_tts_profiles(_b2_summary())
    g_safe = compute_g_safe(curve, profiles, idle_high_water_mb=314.0)
    components = g_safe["components"]
    assert abs(components["torch_allocator_slack_mb"] - 501.3) < 1e-6
    # fixture 中 max = mean + 4，重复运行漂移分量为 4。
    assert components["llm_repeat_run_drift_mb"] == 4.0
    assert components["idle_gpu_baseline_high_water_mb"] == 314.0
    assert components["colocated_positive_residual_mb"] is None
    assert "colocated_positive_residual_mb" in g_safe["pending_components"]
    assert abs(g_safe["value_mb"] - 501.3) < 1e-6


def test_compute_h_layer_semantics() -> None:
    m_l = {
        "K0": {"mean": 100.0, "max": 104.0},
        "K8": {"mean": 2000.0, "max": 2004.0},
        "K16": {"mean": 3400.0, "max": 3404.0},
        "FULL": {"mean": 5600.0, "max": 5604.0},
    }
    # budget = 8151 - 3284 - 501.3 = 4365.7：K16 可行、K8 之上只有 K16。
    result = compute_h(m_l, companion_mb=3284.0, g_safe_mb=501.3, gpu_total_mb=8151.0)
    assert result["h_layers"] == 16
    assert result["full_feasible"] is False

    # companion 很小时 FULL 可行且 H 达到最大已测层。
    result_full = compute_h(m_l, companion_mb=100.0, g_safe_mb=50.0, gpu_total_mb=8151.0)
    assert result_full["h_layers"] == 16
    assert result_full["full_feasible"] is True

    # budget 连 K0 都放不下时 H 为 None。
    result_none = compute_h(m_l, companion_mb=8050.0, g_safe_mb=10.0, gpu_total_mb=8151.0)
    assert result_none["h_layers"] is None

    # max 基准比 mean 更保守。
    m_l_max = {
        "K16": {"mean": 3400.0, "max": 4400.0},
    }
    result_max = compute_h(
        m_l_max,
        companion_mb=3284.0,
        g_safe_mb=501.3,
        gpu_total_mb=8151.0,
        basis="max",
    )
    assert result_max["h_layers"] is None


def test_config_layer_count() -> None:
    assert _config_layer_count("K0") == 0
    assert _config_layer_count("k24") == 24
    assert _config_layer_count("FULL") is None


def test_suggest_refinements_clips_dedups_and_excludes_measured() -> None:
    suggestions = suggest_refinements(
        anchors=[16, 32],
        model_layers=32,
        measured_layers={0, 8, 16, 24, 32},
    )
    by_anchor = suggestions["by_anchor"]
    assert by_anchor["16"] == [12, 14, 18, 20]
    # 32 的 k+2/k+4 被裁剪到 [0,32]，去掉与 16 重复的 28、30 仍保留。
    assert by_anchor["32"] == [28, 30]
    assert suggestions["merged_sorted"] == [12, 14, 18, 20, 28, 30]

    clipped = suggest_refinements(
        anchors=[1],
        model_layers=32,
        measured_layers={1},
    )
    assert clipped["by_anchor"]["1"] == [0, 3, 5]


def test_analyze_end_to_end_with_forge_reprofile() -> None:
    result = analyze(
        c2_summary=_c2_summary(),
        b2_summary=_b2_summary(),
        forge_summary=_forge_summary(),
        legacy_forge_peak_mb=None,
        legacy_forge_idle_mb=None,
        gpu_total_mb=8151.0,
        model_layers=32,
        tts_configs=["T0", "T3"],
    )
    context = result["per_context"]["512"]
    # T0：M_L <= 8151 - 3284 - 501.3 = 4365.7，K24(4383) 不可行 → H=16。
    assert context["h_tts"]["T0"]["h_layers"] == 16
    # T3：M_L <= 8151 - 1980 - 501.3 = 5669.7，fixture 中最大已测层 K24(4383) 可行。
    assert context["h_tts"]["T3"]["h_layers"] == 24
    assert context["h_tts"]["T3"]["full_feasible"] is True
    # H_D：M_L <= 8151 - 5590 - 501.3 = 2059.7，只有 K0(127) 可行。
    assert context["h_d"]["h_layers"] == 0
    assert result["forge"]["source"] == "forge_reprofile"
    # anchors = {16, 24, 0}；排除已测 {0,16,24} 后合并建议。
    assert result["critical_refinement"]["merged_sorted"] == [
        2, 4, 12, 14, 18, 20, 22, 26, 28,
    ]


def test_analyze_marks_h_d_pending_without_forge() -> None:
    result = analyze(
        c2_summary=_c2_summary(),
        b2_summary=_b2_summary(),
        forge_summary=None,
        legacy_forge_peak_mb=None,
        legacy_forge_idle_mb=None,
        gpu_total_mb=8151.0,
        model_layers=32,
        tts_configs=["T0", "T3"],
    )
    context = result["per_context"]["512"]
    assert context["h_d"]["status"] == "pending_forge_reprofile"
    assert result["forge"]["source"] == "missing"
