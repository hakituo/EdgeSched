"""S0 协议冻结测试：07 模块的待确认项必须逐条落值且自洽。"""

from experiments.elastic_residency.s0_pipeline import config as s0_config


def test_freeze_status_no_longer_pending_for_input_items() -> None:
    status = s0_config.FREEZE_STATUS
    assert status["input_and_generation"] == "frozen"
    assert status["sentence_splitting"] == "frozen"
    assert status["startup_policy"] == "frozen"
    assert status["comparison_rule"] == "frozen"
    # 设备侧必须显式保持 pending，不能被顺手标成通过
    assert status["actual_playback_slo"] == "pending_device_validation"
    # 首句就绪只报告不设绝对阈值，避免预填未经证据的数值
    assert status["first_sentence_ready_slo"] == "frozen_as_reporting_only"


def test_frozen_protocol_covers_all_required_rows() -> None:
    protocol = s0_config.FROZEN_PROTOCOL
    assert protocol["version"] == s0_config.S0_PROTOCOL_VERSION
    for key in (
        "input_and_generation",
        "sentence_splitting",
        "startup_policy",
        "repetitions",
        "output_difference_rule",
        "comparison_rule",
        "failure_handling",
    ):
        assert key in protocol, key


def test_repetition_plan_matches_frozen_placements() -> None:
    protocol = s0_config.FROZEN_PROTOCOL
    repeats = protocol["repetitions"]
    assert repeats["formal_placements"] == list(s0_config.S0_PLACEMENTS)
    assert repeats["formal_total_runs"] == repeats["runs_per_placement"] * len(
        s0_config.S0_PLACEMENTS
    )
    assert repeats["paired_order"] == "rotate"


def test_comparison_baselines_are_frozen_placements() -> None:
    rule = s0_config.FROZEN_PROTOCOL["comparison_rule"]
    assert rule["baseline_non_mixed"] in s0_config.S0_PLACEMENTS
    assert rule["reference_same_llm"] in s0_config.S0_PLACEMENTS
    assert 0 < rule["relative_tolerance"] < 1


def test_startup_policy_scope_is_explicit() -> None:
    rule = s0_config.FROZEN_PROTOCOL["startup_policy"]
    assert rule["formal"] == s0_config.DEFAULT_STARTUP_POLICY
    assert set(rule["comparison"]) == set(s0_config.S0_STARTUP_POLICIES)
    assert s0_config.DEFAULT_STARTUP_POLICY == s0_config.STARTUP_FIRST_SENTENCE


def test_failure_handling_forbids_silent_drop() -> None:
    handling = s0_config.FROZEN_PROTOCOL["failure_handling"]
    assert handling["fail_fast_default"] is True
    assert "不得静默删除" in handling["no_silent_drop"]
