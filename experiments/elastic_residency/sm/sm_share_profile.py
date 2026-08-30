"""单个阶段的 GPU 计算份额 profile（``GpuShareProfile``）。

与 ``sm_profiles`` 的枚举/校验原语分开，保证单文件规模可控。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from ..contracts import to_jsonable
from .sm_profiles import (
    GpuComputeBackend,
    GpuComputeShareCapability,
    PERCENTAGE_BACKENDS,
    SPATIAL_PARTITION_BACKENDS,
    SmProfileError,
    backend_capability,
    coerce_backend,
    coerce_capability,
    set_frozen_fields,
    validate_optional_bool,
    validate_optional_int,
    validate_optional_positive_float,
    validate_percentage,
)

SM_PROFILE_SCHEMA_VERSION = "edgesched.sm_profiles.v1"


@dataclass(frozen=True)
class GpuShareProfile:
    """单个阶段在某个静态 GPU 计算份额配置下的完整画像。"""

    phase_id: str
    backend: GpuComputeBackend
    capability: GpuComputeShareCapability
    requested_percentage: int | None = None
    requested_sm_count: int | None = None
    effective_sm_count: int | None = None
    total_physical_sms: int | None = None
    minimum_partition_sms: int | None = None
    requested_gpu_memory_limit_mb: float | None = None
    effective_gpu_memory_limit_mb: float | None = None
    cuda_context_created_after_configuration: bool | None = None
    verification_method: str = ""
    verified: bool = False
    failure_reason: str | None = None
    schema_version: str = SM_PROFILE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        backend = coerce_backend(self.backend)
        capability = coerce_capability(self.capability)
        expected = backend_capability(backend)
        if capability is not expected:
            raise SmProfileError(
                f"capability 与 backend 不一致: {backend.value} 只能声明 "
                f"{expected.value}，实际为 {capability.value}"
            )
        if not isinstance(self.phase_id, str) or not self.phase_id.strip():
            raise SmProfileError("phase_id 不能为空")
        if not isinstance(self.verified, bool):
            raise SmProfileError("verified 必须是布尔值")

        percentage = validate_percentage(
            "requested_percentage", self.requested_percentage
        )
        if backend.value in PERCENTAGE_BACKENDS and percentage is None:
            raise SmProfileError(f"{backend.value} 必须显式提供 requested_percentage")

        total = validate_optional_int("total_physical_sms", self.total_physical_sms)
        minimum = validate_optional_int(
            "minimum_partition_sms", self.minimum_partition_sms
        )
        if total is not None and minimum is not None and minimum > total:
            raise SmProfileError("minimum_partition_sms 不能大于 total_physical_sms")

        requested_sm = validate_optional_int(
            "requested_sm_count", self.requested_sm_count, maximum=total
        )
        effective_sm = validate_optional_int(
            "effective_sm_count", self.effective_sm_count, maximum=total
        )
        if minimum is not None and effective_sm is not None and effective_sm < minimum:
            raise SmProfileError(
                "effective_sm_count 小于后端要求的最小分区粒度 "
                f"({effective_sm} < {minimum})"
            )

        if self.failure_reason is not None and (
            not isinstance(self.failure_reason, str) or not self.failure_reason.strip()
        ):
            raise SmProfileError("failure_reason 必须是非空字符串或 None")
        if self.verified:
            self._validate_verified_state(backend, capability, effective_sm)

        set_frozen_fields(
            self,
            backend=backend,
            capability=capability,
            requested_percentage=percentage,
            requested_sm_count=requested_sm,
            effective_sm_count=effective_sm,
            total_physical_sms=total,
            minimum_partition_sms=minimum,
            requested_gpu_memory_limit_mb=validate_optional_positive_float(
                "requested_gpu_memory_limit_mb", self.requested_gpu_memory_limit_mb
            ),
            effective_gpu_memory_limit_mb=validate_optional_positive_float(
                "effective_gpu_memory_limit_mb", self.effective_gpu_memory_limit_mb
            ),
            cuda_context_created_after_configuration=validate_optional_bool(
                "cuda_context_created_after_configuration",
                self.cuda_context_created_after_configuration,
            ),
            verification_method=str(self.verification_method or ""),
        )

    def _validate_verified_state(
        self,
        backend: GpuComputeBackend,
        capability: GpuComputeShareCapability,
        effective_sm: int | None,
    ) -> None:
        if self.failure_reason is not None:
            raise SmProfileError("存在 failure_reason 时不能标记 verified=True")
        if backend is GpuComputeBackend.UNSUPPORTED:
            raise SmProfileError("UNSUPPORTED 配置不能标记 verified=True")
        if not str(self.verification_method or "").strip():
            raise SmProfileError("verified=True 必须给出 verification_method")
        if self.cuda_context_created_after_configuration is not True:
            raise SmProfileError(
                "verified=True 必须确认 CUDA context 在配置注入之后创建"
            )
        if capability is GpuComputeShareCapability.SM_PARTITIONABLE and (
            effective_sm is None
        ):
            raise SmProfileError(
                "SM_PARTITIONABLE 配置只有通过客户端实测 SM 数后才能标记 verified"
            )

    @property
    def is_spatial_partition(self) -> bool:
        """后端是否声称形成离散空间分区。"""

        return self.backend.value in SPATIAL_PARTITION_BACKENDS

    @property
    def spatial_partition_verified(self) -> bool:
        """是否可以按已验证的空间 SM 分区处理。"""

        return (
            self.is_spatial_partition
            and self.verified
            and self.effective_sm_count is not None
        )

    @property
    def sm_count_verified(self) -> bool:
        """有效 SM 数是否来自客户端 context 实测（而不是由比例推算）。"""

        return self.effective_sm_count is not None and self.verified

    @property
    def exclusive_sm_demand(self) -> int:
        """对“SM 数之和 <= 物理 SM”独占准入判断的贡献。

        ``MPS_CAP`` 只是客户端执行资源上限，不形成独占 SM 保留，贡献为 0。
        """

        if self.spatial_partition_verified:
            return int(self.effective_sm_count or 0)
        return 0

    def to_dict(self) -> dict[str, Any]:
        payload = to_jsonable(self)
        payload["is_spatial_partition"] = self.is_spatial_partition
        payload["spatial_partition_verified"] = self.spatial_partition_verified
        payload["exclusive_sm_demand"] = self.exclusive_sm_demand
        payload["sm_count_verified"] = self.sm_count_verified
        return payload


def unsupported_profile(
    phase_id: str,
    failure_reason: str,
    *,
    requested_percentage: int | None = None,
    total_physical_sms: int | None = None,
    minimum_partition_sms: int | None = None,
    requested_sm_count: int | None = None,
    cuda_context_created_after_configuration: bool | None = None,
) -> GpuShareProfile:
    """构造 fail-closed 的 ``UNSUPPORTED`` profile。

    后端不存在、初始化失败或实际份额无法验证时使用；绝不能用它静默退回
    普通 CUDA 共享。
    """

    if not failure_reason.strip():
        raise SmProfileError("failure_reason 不能为空")
    return GpuShareProfile(
        phase_id=phase_id,
        backend=GpuComputeBackend.UNSUPPORTED,
        capability=backend_capability(GpuComputeBackend.UNSUPPORTED),
        requested_percentage=requested_percentage,
        requested_sm_count=requested_sm_count,
        effective_sm_count=None,
        total_physical_sms=total_physical_sms,
        minimum_partition_sms=minimum_partition_sms,
        cuda_context_created_after_configuration=(
            cuda_context_created_after_configuration
        ),
        verification_method="unsupported_no_measurement",
        verified=False,
        failure_reason=failure_reason,
    )


def backend_value(profile: GpuShareProfile) -> str:
    """返回后端字符串值，便于写入 CSV/JSON。"""

    value = profile.backend
    return value.value if isinstance(value, Enum) else str(value)
