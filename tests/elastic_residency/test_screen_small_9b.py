"""阶段容量筛选的无模型边界检查。"""

import pytest

from experiments.elastic_residency.screen_small_9b import calculate, validate_summary


@pytest.mark.parametrize("nested", [False, True])
def test_provenance_schema_supports_forge_without_ignoring_failures(nested):
    s = {"passed": True}
    if nested:
        s["acceptance_checks"] = {"git_provenance_complete": True}
    else:
        s["git_provenance_complete"] = True
    validate_summary(s)
    s["passed"] = False
    with pytest.raises(ValueError):
        validate_summary(s)
    with pytest.raises(ValueError):
        validate_summary({"passed": True})


def test_joint_vision_is_not_double_counted_and_diffusion_stays_diagnostic():
    curve = {"512": {"K0": {"min": 100, "mean": 100, "max": 100},
                     "FULL": {"min": 5000, "mean": 5000, "max": 5000}}}
    tts = {"T3": {"max_card_delta_mb": 1500, "allocator_slack_mb": 100}}
    result = calculate(curve, tts, 4000, 6000, 500, gpu_total=8151)
    assert result["full_c512_vision_tts_gpu_only"]["T3"]["estimated_gpu_slack_mib"] == 151
    assert result["full_c512_four_component_gpu_slack_mib"]["T3"] == -3849
    assert result["text_llm_tts_gpu_only"]["512"]["T3"]["full_feasible"]
    assert "no automatic rerun" in result["diffusion_host_gate"]
    assert "colocated_positive_residual_mb" in result["provisional_g_safe"]["pending_components"]


def test_observed_baseline_is_not_omitted_from_screen():
    curve = {"512": {"FULL": {"min": 5000, "mean": 5000, "max": 5000}}}
    tts = {"T0": {"max_card_delta_mb": 2500, "allocator_slack_mb": 100}}
    result = calculate(curve, tts, 4000, 6000, 1000)
    assert result["provisional_g_safe"]["value_mb"] == 1000
    assert not result["text_llm_tts_gpu_only"]["512"]["T0"]["full_feasible"]
