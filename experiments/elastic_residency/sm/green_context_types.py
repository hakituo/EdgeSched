"""Green Context 只读探测的类型、符号表与结构体布局自检。

只做类型与常量定义，不调用任何 Driver API。
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
from typing import Any, Mapping

GREEN_CONTEXT_SCHEMA_VERSION = "edgesched.green_context_probe.v1"

CUDA_SUCCESS = 0
CU_DEV_RESOURCE_TYPE_SM = 1

LIBRARY_CANDIDATES = ("libcuda.so.1", "nvcuda.dll", "libcuda.dylib")

#: 进入后续实现所需的驱动 API 符号。stream 符号是
#: ``cuGreenCtxStreamCreate``（不是 ``cuStreamCreateWithGreenCtx``）。
REQUIRED_SYMBOLS: tuple[str, ...] = (
    "cuDeviceGetDevResource",
    "cuDevSmResourceSplitByCount",
    "cuDevResourceGenerateDesc",
    "cuGreenCtxCreate",
    "cuGreenCtxStreamCreate",
)

#: ``CUdevResource`` 布局无法在本仓库内与 CUDA Driver 头文件逐一核对：
#: 联合体大小、``pNext`` 位置与对齐都随版本变化。因此设备资源查询的原始观测值
#: 只能标记为 ``layout_unverified``，不得作为设备能力结论，也不得据此宣称支持。
DEVICE_RESOURCE_LAYOUT_VERIFIED = False
DEVICE_RESOURCE_LAYOUT_REASON = "cuda_driver_header_layout_not_verified"


class CudevSmResource(ctypes.Structure):
    _fields_ = [("type", ctypes.c_uint), ("sm_count", ctypes.c_uint)]


class CudevResourceData(ctypes.Union):
    _fields_ = [
        ("type", ctypes.c_uint),
        ("sm", CudevSmResource),
        ("pad", ctypes.c_char * 256),
    ]


class CudevResource(ctypes.Structure):
    """仅用于**尝试**一次设备资源查询；布局未核对，结果不得作为能力结论。"""

    _fields_ = [("data", CudevResourceData), ("next", ctypes.c_void_p)]


def describe_resource_layout() -> dict[str, Any]:
    """返回结构体布局自检结果，避免靠 padding 猜测宣称可信。"""

    return {
        "struct_size": ctypes.sizeof(CudevResource),
        "type_offset": CudevSmResource.type.offset,
        "sm_count_offset": CudevSmResource.sm_count.offset,
        "layout_verified": DEVICE_RESOURCE_LAYOUT_VERIFIED,
        "reason": DEVICE_RESOURCE_LAYOUT_REASON,
    }


def load_cuda_library(
    candidates: tuple[str, ...] = LIBRARY_CANDIDATES,
) -> tuple[Any | None, str | None]:
    """按顺序尝试加载 CUDA Driver 库；失败时返回原因而不是抛异常。"""

    for candidate in candidates:
        try:
            return ctypes.CDLL(candidate), candidate
        except OSError:
            continue
    return None, None


@dataclass(frozen=True)
class GreenContextCapability:
    """Green Context 只读探测结论。"""

    supported: bool
    status: str
    reasons: tuple[str, ...]
    library: str | None
    symbols: Mapping[str, bool]
    missing_symbols: tuple[str, ...]
    device_resource_query_attempted: bool
    device_sm_count: int | None
    minimum_partition_sms: int | None
    device_sm_count_unverified: int | None = None
    device_resource_layout_verified: bool = False
    gate_conditions: Mapping[str, bool | None] = field(default_factory=dict)
    details: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = GREEN_CONTEXT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "supported": self.supported,
            "status": self.status,
            "reasons": list(self.reasons),
            "library": self.library,
            "symbols": dict(self.symbols),
            "missing_symbols": list(self.missing_symbols),
            "device_resource_query_attempted": (
                self.device_resource_query_attempted
            ),
            "device_sm_count": self.device_sm_count,
            "device_sm_count_unverified": self.device_sm_count_unverified,
            "device_resource_layout_verified": (
                self.device_resource_layout_verified
            ),
            "minimum_partition_sms": self.minimum_partition_sms,
            "gate_conditions": dict(self.gate_conditions),
            "details": dict(self.details),
        }
