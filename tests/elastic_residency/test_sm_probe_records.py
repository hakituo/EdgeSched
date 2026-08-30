"""SM-A 逐配置记录构造的单元测试（不依赖 GPU/CUDA/MPS）。"""

from __future__ import annotations

import pytest

from experiments.elastic_residency.sm.sm_probe_env import (
    SmCapabilityProbeError,
    parse_percentages,
)
from experiments.elastic_residency.sm.sm_probe_records import (
    build_baseline_result,
    build_mps_cap_result,
    build_unsupported_result,
)
from experiments.elastic_residency.sm.sm_profiles import GpuComputeBackend
from tests.elastic_residency.sm_payload_builders import (
    PHYSICAL_TOTAL_SMS,
    client_payload,
    client_run,
)


def _mps_result(**overrides) -> object:
    params = dict(
        phase_id="mps_cap_50",
        percentage=50,
        client=client_run(client_payload(effective_sm_count=24)),
        daemon_snapshot={"label": "mps_cap_50"},
        cleanup_succeeded=True,
        sm_estimate_rounding="ceil",
        physical_total_sms=PHYSICAL_TOTAL_SMS,
    )
    params.update(overrides)
    return build_mps_cap_result(**params)


def test_parse_percentages_validates_range_and_duplicates() -> None:
    assert parse_percentages([100, 75, 50, 25]) == (100, 75, 50, 25)
    for invalid in ([0], [101], [50, 50], []):
        with pytest.raises(SmCapabilityProbeError):
            parse_percentages(invalid)


def test_mps_cap_result_is_verified_only_with_consistent_evidence() -> None:
    result = _mps_result()
    assert result.failure is None
    assert result.profile.verified is True
    assert result.profile.effective_sm_count == 24
    assert result.profile.requested_sm_count == 24
    assert result.fallback_detected is False
    assert result.profile.sm_count_verified is True


def test_mps_client_limited_property_does_not_override_physical_total() -> None:
    """物理 36 SM、25% 客户端看到 9 SM：物理总量仍必须是 36。"""

    result = build_mps_cap_result(
        phase_id="mps_cap_25",
        percentage=25,
        client=client_run(
            client_payload(
                context_visible_sm_count=9,
                effective_sm_count=9,
                requested_percentage=25,
            )
        ),
        daemon_snapshot={},
        cleanup_succeeded=True,
        sm_estimate_rounding="ceil",
        physical_total_sms=36,
    )
    assert result.profile.total_physical_sms == 36
    assert result.profile.requested_sm_count == 9  # ceil(36 * 25 / 100)
    assert result.profile.effective_sm_count == 9
    assert result.capability.details["context_visible_sm_count"] == 9
    assert result.capability.details["physical_total_sms"] == 36
    assert result.failure is None


def test_context_visible_sm_cannot_exceed_physical_total() -> None:
    result = _mps_result(
        client=client_run(
            client_payload(context_visible_sm_count=64, effective_sm_count=24)
        )
    )
    assert "context_visible_sm_count_exceeds_physical_total_sms" in (
        result.failure or ""
    )
    assert result.profile.total_physical_sms == PHYSICAL_TOTAL_SMS


def test_effective_sm_cannot_exceed_physical_total() -> None:
    result = _mps_result(
        client=client_run(
            client_payload(context_visible_sm_count=48, effective_sm_count=96)
        )
    )
    assert "effective_sm_count_exceeds_physical_total_sms" in (result.failure or "")


def test_missing_environment_is_reported_as_fallback_not_success() -> None:
    result = _mps_result(
        client=client_run(client_payload(configuration_applied=False))
    )
    assert result.fallback_detected is True
    assert result.profile.verified is False
    assert result.failure is not None


def test_corrupt_json_and_timeout_produce_failures() -> None:
    corrupt = _mps_result(client=client_run(None, json_parse_error="JSONDecodeError"))
    assert "client_json_parse_error" in (corrupt.failure or "")
    timed_out = _mps_result(client=client_run(client_payload(), timed_out=True))
    assert "client_timeout" in (timed_out.failure or "")


def test_workload_and_context_order_failures_are_recorded() -> None:
    workload = _mps_result(
        client=client_run(client_payload(workload_passed=False))
    )
    assert "cuda_workload_not_passed" in (workload.failure or "")
    context = _mps_result(client=client_run(client_payload(context_created=False)))
    assert "cuda_context_not_created_after_configuration" in (context.failure or "")


def test_unqueried_sm_count_is_not_faked() -> None:
    result = _mps_result(client=client_run(client_payload(effective_sm_count=None)))
    assert "effective_sm_count_not_queryable" in (result.failure or "")
    assert result.profile.effective_sm_count is None
    assert result.profile.exclusive_sm_demand == 0


def test_unsupported_result_is_fail_closed() -> None:
    result = build_unsupported_result(
        phase_id="mps_cap_75",
        percentage=75,
        failure_reason="mps_binary_unavailable:binary_not_found_on_path",
        total_physical_sms=PHYSICAL_TOTAL_SMS,
        binary={"available": False},
    )
    assert result.profile.backend is GpuComputeBackend.UNSUPPORTED
    assert result.profile.verified is False
    assert result.profile.effective_sm_count is None
    assert result.fallback_detected is False


def test_baseline_visible_sm_becomes_physical_total() -> None:
    result = build_baseline_result(
        client=client_run(client_payload(), mps_pid_observed=False)
    )
    assert result.failure is None
    assert result.profile.backend is GpuComputeBackend.UNMANAGED
    assert result.profile.requested_percentage is None
    assert result.profile.total_physical_sms == PHYSICAL_TOTAL_SMS
    assert result.capability.details["note"].startswith("不受控共享基线")
