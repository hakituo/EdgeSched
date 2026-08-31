"""LLM+Forge 候选和容量口径测试。"""

from __future__ import annotations

import base64

import pytest

from experiments.elastic_residency.forge_colocated_capacity import (
    forge_colocated_metrics,
)
from experiments.elastic_residency.llm_forge_plan import (
    LlmForgeCandidate,
    build_execution_plan,
    candidate_profile,
    parse_candidate_specs,
    select_candidates,
    screened_candidates,
)
from experiments.elastic_residency.llm_forge_child import decode_forge_image
from experiments.elastic_residency.llm_forge_parent import _validate_forge_profile


def _screen() -> dict:
    return {
        "gpu_total_mb": 8151.0,
        "g_safe": {
            "value_mb": 501.2822265625,
            "components": {"idle_gpu_baseline_high_water_mb": 487.6953125},
        },
        "per_context": {
            "512": {
                "m_l_mb": {
                    "K4": 1853.09765625,
                    "K6": 2117.09765625,
                    "K8": 2365.09765625,
                },
                "h_d": {"h_layers": 6, "full_feasible": False, "m_d_mb": 5484.0},
            },
            "2048": {
                "m_l_mb": {"K4": 1899.09765625},
                "h_d": {"h_layers": 4, "full_feasible": False, "m_d_mb": 5484.0},
            },
        },
    }


def test_plan_parses_screened_candidates_and_preserves_order() -> None:
    parsed = parse_candidate_specs(["512:K6", "2048:k4"])
    assert parsed == (
        LlmForgeCandidate(512, "K6"),
        LlmForgeCandidate(2048, "K4"),
    )
    assert screened_candidates(_screen()) == parsed
    assert build_execution_plan(parsed, runs=2) == [
        (512, "K6", 1, 1),
        (512, "K6", 2, 2),
        (2048, "K4", 1, 3),
        (2048, "K4", 2, 4),
    ]


def test_plan_rejects_duplicates_and_non_screened_candidate() -> None:
    with pytest.raises(ValueError, match="重复"):
        parse_candidate_specs(["512:K6", "512:k6"])
    with pytest.raises(ValueError, match="不属于"):
        candidate_profile(_screen(), LlmForgeCandidate(512, "K4"))


def test_plan_explicitly_allows_only_profiled_lower_fallback() -> None:
    candidates, boundaries, exploratory = select_candidates(
        _screen(), ["512:K4"], allow_exploratory_fallback=True
    )
    assert candidates == exploratory == (LlmForgeCandidate(512, "K4"),)
    assert boundaries[0] == LlmForgeCandidate(512, "K6")
    profile = candidate_profile(
        _screen(), candidates[0], allow_lower_fallback=True
    )
    assert profile["llm_isolated_increment_mb"] == pytest.approx(1853.09765625)
    with pytest.raises(ValueError, match="边界或其降档"):
        select_candidates(
            _screen(), ["512:K8"], allow_exploratory_fallback=True
        )
    with pytest.raises(ValueError, match="没有候选画像"):
        select_candidates(
            _screen(), ["512:K2"], allow_exploratory_fallback=True
        )


def test_profile_and_capacity_use_global_idle_increment_semantics() -> None:
    profile = candidate_profile(_screen(), LlmForgeCandidate(512, "K6"))
    assert profile["llm_isolated_increment_mb"] == pytest.approx(2117.09765625)
    metrics = forge_colocated_metrics(
        global_idle_reference_mb=profile["global_idle_reference_mb"],
        b_llm_mb=2828.79296875,
        p_colocated_mb=8070.0,
        gpu_total_mb=profile["gpu_total_mb"],
        safety_margin_mb=profile["safety_margin_mb"],
        llm_isolated_increment_mb=profile["llm_isolated_increment_mb"],
        forge_isolated_increment_mb=profile["forge_isolated_increment_mb"],
    )
    assert metrics["colocated_positive_residual_mb"] == pytest.approx(0.0)
    assert metrics["physical_headroom_mb"] == pytest.approx(81.0)
    assert metrics["safe_capacity_headroom_mb"] == pytest.approx(67.4130859375)
    assert metrics["capacity_admissible"] is True


def test_capacity_rejects_invalid_baseline_order() -> None:
    with pytest.raises(ValueError, match="顺序"):
        forge_colocated_metrics(
            global_idle_reference_mb=500.0,
            b_llm_mb=400.0,
            p_colocated_mb=8000.0,
            gpu_total_mb=8151.0,
            safety_margin_mb=501.0,
            llm_isolated_increment_mb=2000.0,
            forge_isolated_increment_mb=5484.0,
        )


def test_decode_forge_image_rejects_non_image() -> None:
    png = b"\x89PNG\r\n\x1a\n" + b"x" * 2048
    assert decode_forge_image(base64.b64encode(png).decode("ascii")) == png
    with pytest.raises(ValueError, match="PNG/JPEG"):
        decode_forge_image(base64.b64encode(b"x" * 2048).decode("ascii"))


def test_forge_profile_requires_same_direct_nvml_run() -> None:
    screen = {
        "forge": {
            "run_id": "forge_v1",
            "m_d_incremental_estimate_mb": 5484.0,
            "global_idle_high_water_mb": 487.0,
        }
    }
    summary = {
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
    }
    profile = _validate_forge_profile(screen, summary)
    assert profile["forge_resident_baseline_mb"] == 711.0
    summary["resource_sampler"]["gpu_cli_subprocess_used"] = True
    with pytest.raises(ValueError, match="direct NVML"):
        _validate_forge_profile(screen, summary)
