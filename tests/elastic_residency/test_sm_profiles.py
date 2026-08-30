"""GPU 计算份额数据契约的单元测试（不依赖 GPU/CUDA/MPS）。"""

from __future__ import annotations

import pytest

from experiments.elastic_residency.sm.sm_partition_rules import (
    GpuPartitionCapability,
    GpuPartitionProbeResult,
    assert_exclusive_sm_budget,
    supports_exclusive_sm_budget,
    validate_exclusive_sm_budget,
)
from experiments.elastic_residency.sm.sm_profiles import (
    GpuComputeBackend,
    GpuComputeShareCapability,
    SmProfileError,
    backend_capability,
    coerce_backend,
    estimate_requested_sm_count,
)
from experiments.elastic_residency.sm.sm_share_profile import (
    GpuShareProfile,
    unsupported_profile,
)

TOTAL_SMS = 48


def _profile(**overrides) -> GpuShareProfile:
    params = dict(
        phase_id="tts",
        backend=GpuComputeBackend.MPS_CAP,
        capability=backend_capability(GpuComputeBackend.MPS_CAP),
        requested_percentage=50,
        total_physical_sms=TOTAL_SMS,
    )
    params.update(overrides)
    return GpuShareProfile(**params)


def test_backend_enum_values_and_capability_mapping() -> None:
    assert coerce_backend("mps_cap") is GpuComputeBackend.MPS_CAP
    assert backend_capability(GpuComputeBackend.UNMANAGED) is (
        GpuComputeShareCapability.UNMANAGED
    )
    assert backend_capability(GpuComputeBackend.MPS_CAP) is (
        GpuComputeShareCapability.CAP_LIMITABLE
    )
    assert backend_capability(GpuComputeBackend.MPS_STATIC) is (
        GpuComputeShareCapability.SM_PARTITIONABLE
    )
    assert backend_capability(GpuComputeBackend.GREEN_CONTEXT) is (
        GpuComputeShareCapability.SM_PARTITIONABLE
    )
    assert backend_capability(GpuComputeBackend.UNSUPPORTED) is (
        GpuComputeShareCapability.UNMANAGED
    )


def test_profile_serialization_keeps_backend_and_capability_strings() -> None:
    payload = _profile(requested_percentage=75).to_dict()
    assert payload["backend"] == "MPS_CAP"
    assert payload["capability"] == "CAP_LIMITABLE"
    assert payload["requested_percentage"] == 75
    assert payload["is_spatial_partition"] is False
    assert payload["spatial_partition_verified"] is False
    assert payload["exclusive_sm_demand"] == 0


def test_capability_must_match_backend() -> None:
    with pytest.raises(SmProfileError, match="capability 与 backend 不一致"):
        _profile(capability=GpuComputeShareCapability.SM_PARTITIONABLE)


@pytest.mark.parametrize("percentage", [0, -1, 101, 150])
def test_percentage_range_is_validated(percentage: int) -> None:
    with pytest.raises(SmProfileError):
        _profile(requested_percentage=percentage)


def test_percentage_backends_require_explicit_percentage() -> None:
    with pytest.raises(SmProfileError, match="必须显式提供 requested_percentage"):
        _profile(requested_percentage=None)


def test_unmanaged_profile_does_not_require_percentage() -> None:
    profile = GpuShareProfile(
        phase_id="baseline",
        backend=GpuComputeBackend.UNMANAGED,
        capability=backend_capability(GpuComputeBackend.UNMANAGED),
        total_physical_sms=TOTAL_SMS,
    )
    assert profile.requested_percentage is None
    assert profile.exclusive_sm_demand == 0


@pytest.mark.parametrize(
    ("percentage", "rounding", "expected"),
    [
        (100, "ceil", 48),
        (50, "ceil", 24),
        (50, "floor", 24),
        (50, "round", 24),
        (25, "ceil", 12),
        (75, "ceil", 36),
        (33, "ceil", 16),
        (33, "floor", 15),
        (33, "round", 16),
        (1, "ceil", 1),
        (1, "floor", 1),
    ],
)
def test_requested_sm_rounding_boundaries(
    percentage: int, rounding: str, expected: int
) -> None:
    assert (
        estimate_requested_sm_count(TOTAL_SMS, percentage, rounding=rounding)
        == expected
    )


def test_estimate_returns_none_when_total_unknown() -> None:
    assert estimate_requested_sm_count(None, 50) is None
    assert estimate_requested_sm_count(TOTAL_SMS, None) is None


def test_requested_and_effective_sm_are_distinct_fields() -> None:
    profile = _profile(
        requested_percentage=25,
        requested_sm_count=12,
        effective_sm_count=16,
        verified=True,
        verification_method="cuda_driver_cuCtxGetExecAffinity",
        cuda_context_created_after_configuration=True,
    )
    assert profile.requested_sm_count == 12
    assert profile.effective_sm_count == 16
    assert profile.sm_count_verified is True


def test_effective_sm_cannot_exceed_total_or_be_faked_with_zero() -> None:
    with pytest.raises(SmProfileError, match="禁止用 0 假装已测量"):
        _profile(effective_sm_count=0)
    with pytest.raises(SmProfileError, match="必须 <= 48"):
        _profile(effective_sm_count=49, verified=False)


def test_effective_sm_below_minimum_partition_is_rejected() -> None:
    with pytest.raises(SmProfileError, match="最小分区粒度"):
        _profile(minimum_partition_sms=8, effective_sm_count=4)


def test_memory_limit_is_modelled_separately_from_sm() -> None:
    profile = _profile(
        requested_gpu_memory_limit_mb=1024.0,
        effective_gpu_memory_limit_mb=1536.0,
    )
    assert profile.requested_gpu_memory_limit_mb == 1024.0
    assert profile.effective_gpu_memory_limit_mb == 1536.0
    assert profile.effective_sm_count is None
    with pytest.raises(SmProfileError, match="必须大于 0"):
        _profile(effective_gpu_memory_limit_mb=0)


def test_mps_cap_never_counts_toward_exclusive_partition_budget() -> None:
    cap_profile = _profile(
        phase_id="tts_cap",
        effective_sm_count=48,
        verified=True,
        verification_method="mps_client_subprocess",
        cuda_context_created_after_configuration=True,
    )
    assert cap_profile.sm_count_verified is True
    assert supports_exclusive_sm_budget(cap_profile) is False
    assert cap_profile.exclusive_sm_demand == 0
    verdict = validate_exclusive_sm_budget([cap_profile, cap_profile], TOTAL_SMS)
    assert verdict["within_budget"] is True
    assert verdict["demand_sm"] == 0
    assert verdict["excluded_phase_ids"] == ["tts_cap", "tts_cap"]


def _static_partition_profile(effective: int) -> GpuShareProfile:
    return GpuShareProfile(
        phase_id=f"phase_{effective}",
        backend=GpuComputeBackend.MPS_STATIC,
        capability=backend_capability(GpuComputeBackend.MPS_STATIC),
        requested_percentage=50,
        requested_sm_count=effective,
        effective_sm_count=effective,
        total_physical_sms=TOTAL_SMS,
        minimum_partition_sms=8,
        cuda_context_created_after_configuration=True,
        verification_method="cuda_driver_cuCtxGetExecAffinity",
        verified=True,
    )


def test_only_verified_spatial_partitions_consume_budget() -> None:
    first = _static_partition_profile(16)
    second = _static_partition_profile(24)
    assert supports_exclusive_sm_budget(first) is True
    verdict = validate_exclusive_sm_budget([first, second, _profile()], TOTAL_SMS)
    assert verdict["demand_sm"] == 40
    assert verdict["within_budget"] is True
    assert verdict["participant_phase_ids"] == ["phase_16", "phase_24"]
    assert verdict["excluded_phase_ids"] == ["tts"]

    with pytest.raises(SmProfileError, match="空间分区 SM 预算越界"):
        assert_exclusive_sm_budget([first, second, _static_partition_profile(16)], TOTAL_SMS)


def test_budget_is_not_judged_when_total_unknown() -> None:
    verdict = validate_exclusive_sm_budget([_static_partition_profile(16)], None)
    assert verdict["judged"] is False
    assert verdict["reason"] == "total_physical_sms_unknown"
    assert verdict["within_budget"] is None


def test_verified_requires_context_order_and_method() -> None:
    with pytest.raises(SmProfileError, match="verification_method"):
        _profile(verified=True)
    with pytest.raises(SmProfileError, match="CUDA context"):
        _profile(
            verified=True,
            verification_method="mps_client_subprocess",
            cuda_context_created_after_configuration=None,
        )


def test_sm_partitionable_requires_measured_sm_count() -> None:
    with pytest.raises(SmProfileError, match="客户端实测 SM 数"):
        GpuShareProfile(
            phase_id="gc",
            backend=GpuComputeBackend.GREEN_CONTEXT,
            capability=backend_capability(GpuComputeBackend.GREEN_CONTEXT),
            requested_percentage=50,
            total_physical_sms=TOTAL_SMS,
            cuda_context_created_after_configuration=True,
            verification_method="cuda_driver_cuCtxGetExecAffinity",
            verified=True,
        )


def test_unsupported_profile_is_fail_closed() -> None:
    profile = unsupported_profile(
        "tts", "mps_binary_unavailable:binary_not_found_on_path", requested_percentage=75
    )
    assert profile.backend is GpuComputeBackend.UNSUPPORTED
    assert profile.verified is False
    assert profile.effective_sm_count is None
    assert profile.exclusive_sm_demand == 0
    with pytest.raises(SmProfileError):
        unsupported_profile("tts", "   ")


def test_probe_result_serializes_nested_profile() -> None:
    profile = _profile()
    capability = GpuPartitionCapability(
        backend=GpuComputeBackend.MPS_CAP,
        capability=backend_capability(GpuComputeBackend.MPS_CAP),
        backend_available=True,
        total_physical_sms=TOTAL_SMS,
    )
    result = GpuPartitionProbeResult(
        phase_id="tts",
        profile=profile,
        capability=capability,
        client_returncode=0,
        cleanup_succeeded=True,
    )
    payload = result.to_dict()
    assert payload["profile"]["backend"] == "MPS_CAP"
    assert payload["capability"]["backend_available"] is True
    assert payload["cleanup_succeeded"] is True
