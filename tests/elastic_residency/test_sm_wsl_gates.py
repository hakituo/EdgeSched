"""SM-A WSL 硬门的纯单元测试：证明硬门真的会拦截缺陷，而不是永远放行。"""

from __future__ import annotations

from typing import Any

import pytest

from tests.elastic_residency.sm_wsl_gates import (
    assert_affinity_query_present,
    assert_cleanup_closed,
    assert_controller_ready,
    assert_no_implementation_fault,
    is_environment_limitation,
)


def _summary(reasons: list[str], failures: list[str | None]) -> dict[str, Any]:
    return {
        "unsupported_reasons": reasons,
        "sm_evidence": [
            {"phase_id": f"mps_cap_{index}", "failure": failure}
            for index, failure in enumerate(failures)
        ],
    }


def _record(**daemon: Any) -> dict[str, Any]:
    return {"phase_id": "mps_cap_50", "daemon": daemon}


# -------------------- 环境白名单 --------------------


@pytest.mark.parametrize(
    "reason",
    [
        "mps_binary_unavailable:binary_not_found_on_path",
        "effective_sm_count_not_queryable_from_client_context",
        "effective_sm_count_not_verified_for_at_least_two_percentages",
        "cuCtxGetExecAffinity_rc_224",
    ],
)
def test_environment_limitations_are_whitelisted(reason: str) -> None:
    assert is_environment_limitation(reason) is True
    assert_no_implementation_fault(_summary([reason], [None]))


@pytest.mark.parametrize(
    "reason",
    [
        "mps_daemon_start_failed:boom",
        "client_status:cuda_device_unavailable",
        "mps_client_pid_not_observed_in_get_client_list",
        "some_completely_unknown_failure_mode",
    ],
)
def test_any_non_whitelisted_reason_fails_the_gate(reason: str) -> None:
    with pytest.raises(AssertionError, match="非环境限制的失败"):
        assert_no_implementation_fault(_summary([reason], [None]))


def test_unknown_per_item_failure_fails_the_gate() -> None:
    """不能只匹配已知 marker：未知 failure 同样必须失败。"""

    with pytest.raises(AssertionError, match="非环境限制的失败"):
        assert_no_implementation_fault(
            _summary([], ["totally_new_failure_mode_x"])
        )


def test_whitelisted_per_item_failure_is_allowed() -> None:
    summary = _summary([], ["effective_sm_count_not_queryable_from_client_context"])
    assert_no_implementation_fault(summary)


# -------------------- 清理闭合 --------------------


def test_closed_cleanup_passes() -> None:
    assert_cleanup_closed(
        [
            _record(
                cleanup_succeeded=True,
                post_quit_evidence={"reached_quiescent": True},
                residual_artifacts=[],
            )
        ]
    )


def test_missing_residual_artifacts_field_fails() -> None:
    with pytest.raises(AssertionError, match="缺少 residual_artifacts"):
        assert_cleanup_closed(
            [
                _record(
                    cleanup_succeeded=True,
                    post_quit_evidence={"reached_quiescent": True},
                )
            ]
        )


def test_non_empty_residual_artifacts_fails() -> None:
    with pytest.raises(AssertionError, match="residual="):
        assert_cleanup_closed(
            [
                _record(
                    cleanup_succeeded=True,
                    post_quit_evidence={"reached_quiescent": True},
                    residual_artifacts=["controller_pid_file"],
                )
            ]
        )


def test_cleanup_not_succeeded_fails() -> None:
    with pytest.raises(AssertionError, match="cleanup_succeeded"):
        assert_cleanup_closed(
            [
                _record(
                    cleanup_succeeded=False,
                    post_quit_evidence={"reached_quiescent": True},
                    residual_artifacts=[],
                )
            ]
        )


def test_quiescent_not_reached_fails() -> None:
    with pytest.raises(AssertionError, match="post_quit_evidence"):
        assert_cleanup_closed(
            [
                _record(
                    cleanup_succeeded=True,
                    post_quit_evidence={"reached_quiescent": False},
                    residual_artifacts=[],
                )
            ]
        )


def test_missing_daemon_snapshot_fails() -> None:
    with pytest.raises(AssertionError, match="缺少 daemon 快照"):
        assert_cleanup_closed([{"phase_id": "mps_cap_50", "daemon": {}}])


# -------------------- driver_query --------------------


def _payload(driver_query: Any) -> dict[str, Any]:
    return {"sm_measurement": {"driver_query": driver_query}}


def test_present_driver_query_passes() -> None:
    assert_affinity_query_present(
        [_payload({"requested_affinity_type": 1, "cu_query_returncode": 224})]
    )


def test_missing_driver_query_fails() -> None:
    with pytest.raises(AssertionError, match="缺少 sm_measurement.driver_query"):
        assert_affinity_query_present([{"sm_measurement": {}}])


def test_missing_sm_measurement_fails() -> None:
    with pytest.raises(AssertionError, match="缺少 sm_measurement"):
        assert_affinity_query_present([{}])


def test_wrong_affinity_type_fails() -> None:
    with pytest.raises(AssertionError, match="requested_affinity_type"):
        assert_affinity_query_present([_payload({"requested_affinity_type": 0})])


def test_bad_returncode_type_fails() -> None:
    with pytest.raises(AssertionError, match="cu_query_returncode"):
        assert_affinity_query_present([_payload({"cu_query_returncode": "224"})])


def test_empty_payload_list_fails() -> None:
    with pytest.raises(AssertionError, match="缺少受控客户端 payload"):
        assert_affinity_query_present([])


# -------------------- controller 就绪 --------------------


def test_controller_ready_passes() -> None:
    assert_controller_ready(
        [_record(controller_readiness={"ready": True, "pid": 4242})]
    )


def test_missing_controller_readiness_fails() -> None:
    with pytest.raises(AssertionError, match="缺少 controller_readiness"):
        assert_controller_ready([_record()])


def test_controller_not_ready_fails() -> None:
    with pytest.raises(AssertionError, match="controller 未就绪"):
        assert_controller_ready([_record(controller_readiness={"ready": False})])
