"""GPU 计算份额（SM）的基础枚举、能力映射与字段校验。

本模块只提供类型与校验原语，不启动任何后端、不创建 CUDA context。
``GpuShareProfile`` 位于 ``sm_share_profile``，能力/探测记录位于
``sm_partition_rules``。

语义边界必须与 ``paper/thesis_experiment_guide/06_implementation_and_roadmap.md`` 的 P1-C 一致：

- ``MPS_CAP`` 只是 ``CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`` 客户端执行资源上限，
  不是独占 SM 分区；
- 只有 ``MPS_STATIC`` / ``GREEN_CONTEXT`` 且验证成功后才能按空间分区处理；
- 未测量到的字段保持 ``None``，禁止用 0 假装测量成功。
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any

PERCENTAGE_MIN = 1
PERCENTAGE_MAX = 100

#: 只有这两个后端在验证成功后才能按可验证的空间 SM 分区处理。
SPATIAL_PARTITION_BACKENDS = frozenset({"MPS_STATIC", "GREEN_CONTEXT"})
#: 这三个后端必须显式给出请求比例。
PERCENTAGE_BACKENDS = frozenset({"MPS_CAP", "MPS_STATIC", "GREEN_CONTEXT"})

ROUNDING_MODES = ("ceil", "floor", "round")


class SmProfileError(ValueError):
    """GPU 计算份额数据契约校验失败。"""


class GpuComputeBackend(str, Enum):
    """实际使用（或尝试使用）的 GPU 计算份额后端。"""

    UNMANAGED = "UNMANAGED"
    MPS_CAP = "MPS_CAP"
    MPS_STATIC = "MPS_STATIC"
    GREEN_CONTEXT = "GREEN_CONTEXT"
    UNSUPPORTED = "UNSUPPORTED"


class GpuComputeShareCapability(str, Enum):
    """阶段 + 后端联合声明的计算份额能力（论文中的 ``chi_i``）。"""

    UNMANAGED = "UNMANAGED"
    CAP_LIMITABLE = "CAP_LIMITABLE"
    SM_PARTITIONABLE = "SM_PARTITIONABLE"


def coerce_backend(value: Any) -> GpuComputeBackend:
    """把字符串或枚举统一成 ``GpuComputeBackend``。"""

    if isinstance(value, GpuComputeBackend):
        return value
    if isinstance(value, str):
        try:
            return GpuComputeBackend(value.strip().upper())
        except ValueError as exc:
            raise SmProfileError(f"未知的 GPU 计算份额后端: {value!r}") from exc
    raise SmProfileError(f"后端必须是 GpuComputeBackend 或字符串: {value!r}")


def coerce_capability(value: Any) -> GpuComputeShareCapability:
    """把字符串或枚举统一成 ``GpuComputeShareCapability``。"""

    if isinstance(value, GpuComputeShareCapability):
        return value
    if isinstance(value, str):
        try:
            return GpuComputeShareCapability(value.strip().upper())
        except ValueError as exc:
            raise SmProfileError(f"未知的 GPU 计算份额能力: {value!r}") from exc
    raise SmProfileError(f"能力必须是 GpuComputeShareCapability 或字符串: {value!r}")


def backend_capability(backend: Any) -> GpuComputeShareCapability:
    """后端能提供的能力上限；``MPS_CAP`` 只能得到 ``CAP_LIMITABLE``。"""

    resolved = coerce_backend(backend)
    if resolved.value in SPATIAL_PARTITION_BACKENDS:
        return GpuComputeShareCapability.SM_PARTITIONABLE
    if resolved is GpuComputeBackend.MPS_CAP:
        return GpuComputeShareCapability.CAP_LIMITABLE
    return GpuComputeShareCapability.UNMANAGED


def is_int(value: Any) -> bool:
    """整数判断；``bool`` 不是合法整数。"""

    return isinstance(value, int) and not isinstance(value, bool)


def estimate_requested_sm_count(
    total_physical_sms: int | None,
    percentage: int | None,
    *,
    rounding: str = "ceil",
) -> int | None:
    """由请求比例推算**请求/预期** SM 数。

    该结果只能作为请求值或预期值写入证据，永远不是实测的有效 SM 数。
    ``total_physical_sms`` 未知时返回 ``None``，禁止用 0 代替。
    """

    if rounding not in ROUNDING_MODES:
        raise SmProfileError(f"未知舍入模式: {rounding!r}")
    if total_physical_sms is None:
        return None
    if not is_int(total_physical_sms) or total_physical_sms < 1:
        raise SmProfileError("total_physical_sms 必须是 >= 1 的整数")
    if percentage is None:
        return None
    if not is_int(percentage):
        raise SmProfileError("percentage 必须是整数")
    if not PERCENTAGE_MIN <= percentage <= PERCENTAGE_MAX:
        raise SmProfileError(
            f"percentage 必须位于 [{PERCENTAGE_MIN}, {PERCENTAGE_MAX}]"
        )
    raw = total_physical_sms * percentage / 100.0
    if rounding == "ceil":
        return int(math.ceil(raw))
    if rounding == "floor":
        return max(1, int(math.floor(raw)))
    return max(1, int(math.floor(raw + 0.5)))


def validate_percentage(name: str, value: Any) -> int | None:
    """校验百分比范围；``None`` 表示未请求。"""

    if value is None:
        return None
    if not is_int(value):
        raise SmProfileError(f"{name} 必须是整数")
    if not PERCENTAGE_MIN <= value <= PERCENTAGE_MAX:
        raise SmProfileError(
            f"{name} 必须位于 [{PERCENTAGE_MIN}, {PERCENTAGE_MAX}]"
        )
    return int(value)


def validate_optional_int(
    name: str,
    value: Any,
    *,
    minimum: int = 1,
    maximum: int | None = None,
) -> int | None:
    """校验可选整数；未知必须传 ``None``，禁止用 0 假装已测量。"""

    if value is None:
        return None
    if not is_int(value):
        raise SmProfileError(f"{name} 必须是整数或 None，禁止用 0 假装已测量")
    if value == 0 and minimum >= 1:
        raise SmProfileError(
            f"{name} 为 0 表示未测量，未知时必须填 None，禁止用 0 假装已测量"
        )
    if value < minimum:
        raise SmProfileError(f"{name} 必须 >= {minimum}")
    if maximum is not None and value > maximum:
        raise SmProfileError(f"{name} 必须 <= {maximum}")
    return int(value)


def validate_optional_positive_float(name: str, value: Any) -> float | None:
    """校验可选正浮点数；未知必须传 ``None``。"""

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SmProfileError(f"{name} 必须是数值或 None")
    if not math.isfinite(float(value)):
        raise SmProfileError(f"{name} 必须是有限数值")
    if float(value) <= 0:
        raise SmProfileError(f"{name} 必须大于 0；未知时请填 None")
    return float(value)


def validate_optional_bool(name: str, value: Any) -> bool | None:
    """校验可选布尔；未知必须传 ``None``。"""

    if value is None:
        return None
    if not isinstance(value, bool):
        raise SmProfileError(f"{name} 必须是布尔或 None")
    return value


def set_frozen_fields(instance: Any, **values: Any) -> None:
    """在 ``frozen=True`` dataclass 的 ``__post_init__`` 中写入规范化后的字段。"""

    for name, value in values.items():
        object.__setattr__(instance, name, value)
