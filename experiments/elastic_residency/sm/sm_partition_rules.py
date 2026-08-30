"""GPU 分区能力记录、逐配置探测结果与独占 SM 预算规则。

核心规则：只有 ``MPS_STATIC`` / ``GREEN_CONTEXT`` 且验证成功的 profile 才能
参与“SM 数之和 <= 物理 SM”的独占分区准入判断；``MPS_CAP`` 与 ``UNMANAGED``
一律不参与，因为它们不形成可验证的独占 SM 保留。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..contracts import to_jsonable
from .sm_profiles import (
    GpuComputeBackend,
    GpuComputeShareCapability,
    SmProfileError,
    backend_capability,
    coerce_backend,
    coerce_capability,
    is_int,
    set_frozen_fields,
    validate_optional_bool,
    validate_optional_int,
)
from .sm_share_profile import GpuShareProfile

SM_PROFILE_SCHEMA_VERSION = "edgesched.sm_profiles.v1"


def supports_exclusive_sm_budget(profile: GpuShareProfile) -> bool:
    """该 profile 是否可以计入独占分区准入判断。"""

    return profile.spatial_partition_verified


def validate_exclusive_sm_budget(
    profiles: Sequence[GpuShareProfile],
    total_physical_sms: int | None,
) -> dict[str, Any]:
    """校验“空间分区 SM 数之和不超过物理 SM”。

    ``total_physical_sms`` 未知时返回 ``judged=False``，而不是假装通过。
    """

    participants = [
        profile for profile in profiles if supports_exclusive_sm_budget(profile)
    ]
    demand = sum(profile.exclusive_sm_demand for profile in participants)
    excluded = [
        profile.phase_id
        for profile in profiles
        if not supports_exclusive_sm_budget(profile)
    ]
    if total_physical_sms is None:
        return {
            "judged": False,
            "reason": "total_physical_sms_unknown",
            "demand_sm": demand,
            "total_physical_sms": None,
            "participant_phase_ids": [p.phase_id for p in participants],
            "excluded_phase_ids": excluded,
            "within_budget": None,
        }
    if not is_int(total_physical_sms) or total_physical_sms < 1:
        raise SmProfileError("total_physical_sms 必须是 >= 1 的整数")
    return {
        "judged": True,
        "reason": None,
        "demand_sm": demand,
        "total_physical_sms": int(total_physical_sms),
        "participant_phase_ids": [p.phase_id for p in participants],
        "excluded_phase_ids": excluded,
        "within_budget": demand <= int(total_physical_sms),
    }


def assert_exclusive_sm_budget(
    profiles: Sequence[GpuShareProfile],
    total_physical_sms: int,
) -> None:
    """显存与 SM 是两类预算；该函数只校验 SM 空间分区预算。"""

    verdict = validate_exclusive_sm_budget(profiles, total_physical_sms)
    if not verdict["within_budget"]:
        raise SmProfileError(
            "空间分区 SM 预算越界: "
            f"{verdict['demand_sm']} > {verdict['total_physical_sms']}"
        )


@dataclass(frozen=True)
class GpuPartitionCapability:
    """一次能力探测得到的后端能力边界。"""

    backend: GpuComputeBackend
    capability: GpuComputeShareCapability
    backend_available: bool | None
    total_physical_sms: int | None = None
    minimum_partition_sms: int | None = None
    requested_percentage: int | None = None
    effective_sm_count: int | None = None
    dynamic_reconfiguration_supported: bool | None = None
    cuda_context_created_after_configuration: bool | None = None
    verification_method: str = ""
    failure_reason: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SM_PROFILE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        backend = coerce_backend(self.backend)
        capability = coerce_capability(self.capability)
        if capability is not backend_capability(backend):
            raise SmProfileError(
                f"capability 与 backend 不一致: {backend.value} 只能声明 "
                f"{backend_capability(backend).value}"
            )
        set_frozen_fields(
            self,
            backend=backend,
            capability=capability,
            backend_available=validate_optional_bool(
                "backend_available", self.backend_available
            ),
            total_physical_sms=validate_optional_int(
                "total_physical_sms", self.total_physical_sms
            ),
            minimum_partition_sms=validate_optional_int(
                "minimum_partition_sms", self.minimum_partition_sms
            ),
            effective_sm_count=validate_optional_int(
                "effective_sm_count", self.effective_sm_count
            ),
            dynamic_reconfiguration_supported=validate_optional_bool(
                "dynamic_reconfiguration_supported",
                self.dynamic_reconfiguration_supported,
            ),
            cuda_context_created_after_configuration=validate_optional_bool(
                "cuda_context_created_after_configuration",
                self.cuda_context_created_after_configuration,
            ),
            verification_method=str(self.verification_method or ""),
            details=dict(self.details or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True)
class GpuPartitionProbeResult:
    """单个静态配置的完整探测记录，用于逐配置 JSONL 证据。"""

    phase_id: str
    profile: GpuShareProfile
    capability: GpuPartitionCapability
    client_returncode: int | None = None
    client_stdout_path: str | None = None
    client_stderr_path: str | None = None
    daemon: Mapping[str, Any] = field(default_factory=dict)
    cleanup_succeeded: bool | None = None
    fallback_detected: bool = False
    failure: str | None = None
    wall_time_iso: str = ""
    schema_version: str = SM_PROFILE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.phase_id, str) or not self.phase_id.strip():
            raise SmProfileError("phase_id 不能为空")
        if not isinstance(self.profile, GpuShareProfile):
            raise SmProfileError("profile 必须是 GpuShareProfile")
        if not isinstance(self.capability, GpuPartitionCapability):
            raise SmProfileError("capability 必须是 GpuPartitionCapability")
        if not isinstance(self.fallback_detected, bool):
            raise SmProfileError("fallback_detected 必须是布尔值")
        set_frozen_fields(
            self,
            cleanup_succeeded=validate_optional_bool(
                "cleanup_succeeded", self.cleanup_succeeded
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = to_jsonable(self)
        payload["profile"] = self.profile.to_dict()
        payload["capability"] = self.capability.to_dict()
        return payload
