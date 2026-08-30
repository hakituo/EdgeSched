#!/usr/bin/env python3
"""Green Context 只读能力探测（不创建任何 Green Context）。

当前主环境名为 ``mvp_core-cu128``，不能假设 CUDA 13.1 Runtime Green Context
API 可用。本模块**只做只读探测**：加载 CUDA Driver 库、检查符号是否存在，
并在显式要求时才尝试一次设备 SM 资源查询。它不创建 Green Context、不创建
stream、也不提供任何伪造的 Python wrapper。

进入后续 Green Context 实现必须同时满足（本模块只负责前两项的探测）：

1. 驱动 API 符号与设备资源查询真实成功；
2. 能查询最小 SM 分区粒度；
3. 能创建 Green Context 和对应 stream；
4. 目标框架能够把全部 kernel 提交到该 stream；
5. 能证明没有 default-stream 逃逸。

PyTorch / llama.cpp / Forge 不能证明上述条件时，Green Context 必须保持
“实验性未支持”状态，并给出 ``UNSUPPORTED`` 结论。
"""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import json
import platform
import sys
from pathlib import Path
from typing import Any

from .green_context_types import (
    CUDA_SUCCESS,
    CU_DEV_RESOURCE_TYPE_SM,
    DEVICE_RESOURCE_LAYOUT_REASON,
    DEVICE_RESOURCE_LAYOUT_VERIFIED,
    CudevResource,
    GreenContextCapability,
    LIBRARY_CANDIDATES,
    REQUIRED_SYMBOLS,
    describe_resource_layout,
    load_cuda_library,
)


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _gate_conditions(
    *,
    symbols_complete: bool,
    device_query: bool | None,
) -> dict[str, bool | None]:
    return {
        "driver_api_symbols": symbols_complete,
        "device_resource_query": (
            bool(device_query) and DEVICE_RESOURCE_LAYOUT_VERIFIED
        ),
        "device_resource_layout_verified": DEVICE_RESOURCE_LAYOUT_VERIFIED,
        "min_partition_granularity": False,
        "green_context_creation": False,
        "framework_stream_binding": False,
        "no_default_stream_escape": False,
    }


def probe_green_context(
    *,
    attempt_device_resource: bool = False,
    library_candidates: tuple[str, ...] = LIBRARY_CANDIDATES,
) -> GreenContextCapability:
    """执行一次只读能力探测。

    默认只做符号检查，不初始化 CUDA，也不创建任何对象。
    """

    library, library_name = load_cuda_library(library_candidates)
    if library is None:
        return GreenContextCapability(
            supported=False,
            status="unsupported_cuda_driver_library_unavailable",
            reasons=("cuda_driver_library_unavailable",),
            library=None,
            symbols={name: False for name in REQUIRED_SYMBOLS},
            missing_symbols=REQUIRED_SYMBOLS,
            device_resource_query_attempted=False,
            device_sm_count=None,
            minimum_partition_sms=None,
            gate_conditions=_gate_conditions(
                symbols_complete=False, device_query=None
            ),
            details={"library_candidates": list(library_candidates)},
        )

    symbols = {name: hasattr(library, name) for name in REQUIRED_SYMBOLS}
    missing = tuple(name for name, present in symbols.items() if not present)
    reasons: list[str] = []
    if missing:
        reasons.append("required_driver_api_symbols_missing")

    observed: int | None = None
    query_reason: str | None = None
    details: dict[str, Any] = {"library_candidates": list(library_candidates)}

    if attempt_device_resource and not missing:
        observed, query_reason = _attempt_device_resource(library)
        details["device_resource_query_reason"] = query_reason
        details["device_resource_layout"] = describe_resource_layout()
        if observed is None:
            reasons.append("device_resource_query_failed")
    else:
        reasons.append("device_resource_query_not_attempted_read_only_probe")
    # 布局未与 CUDA 头文件核对，原始观测值不能升级成设备能力结论。
    reasons.append(DEVICE_RESOURCE_LAYOUT_REASON)
    reasons.append("min_partition_granularity_unverified")
    # 后续三项条件需要真实创建 Green Context / stream 并证明框架全部 kernel
    # 都提交到该 stream。本轮一律不做，因此永远保持 False。
    reasons.append("green_context_creation_not_attempted")
    reasons.append("framework_stream_binding_unproven")
    reasons.append("default_stream_escape_unproven")

    return GreenContextCapability(
        supported=False,
        status="unsupported_experimental_read_only",
        reasons=tuple(reasons),
        library=library_name,
        symbols=symbols,
        missing_symbols=missing,
        device_resource_query_attempted=bool(attempt_device_resource),
        device_sm_count=None,
        device_sm_count_unverified=observed,
        device_resource_layout_verified=DEVICE_RESOURCE_LAYOUT_VERIFIED,
        minimum_partition_sms=None,
        gate_conditions=_gate_conditions(
            symbols_complete=not missing,
            device_query=observed is not None,
        ),
        details=details,
    )


def _attempt_device_resource(library: Any) -> tuple[int | None, str | None]:
    """尝试一次设备 SM 资源查询，返回**未验证布局**的原始观测值。

    只调用 ``cuInit`` / ``cuDeviceGet`` / ``cuDeviceGetDevResource``，不创建
    Green Context 或 stream。``CUdevResource`` 布局未与 CUDA 头文件核对，
    因此返回值只能记为 ``device_sm_count_unverified``，不能作为设备能力结论。
    任何异常都必须被捕获并转成原因，绝不让只读探针崩溃。
    """

    try:
        cu_init = getattr(library, "cuInit", None)
        cu_device_get = getattr(library, "cuDeviceGet", None)
        cu_dev_resource = getattr(library, "cuDeviceGetDevResource", None)
        if cu_init is None or cu_device_get is None or cu_dev_resource is None:
            return None, "device_resource_symbols_unavailable"
        cu_init.argtypes = [ctypes.c_uint]
        cu_init.restype = ctypes.c_int
        cu_device_get.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        cu_device_get.restype = ctypes.c_int
        cu_dev_resource.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(CudevResource),
            ctypes.c_uint,
        ]
        cu_dev_resource.restype = ctypes.c_int

        init_rc = int(cu_init(0))
        if init_rc != CUDA_SUCCESS:
            return None, f"cuInit_rc_{init_rc}"
        device = ctypes.c_int(0)
        device_rc = int(cu_device_get(ctypes.byref(device), 0))
        if device_rc != CUDA_SUCCESS:
            return None, f"cuDeviceGet_rc_{device_rc}"
        resource = CudevResource()
        resource_rc = int(
            cu_dev_resource(
                device.value, ctypes.byref(resource), CU_DEV_RESOURCE_TYPE_SM
            )
        )
        if resource_rc != CUDA_SUCCESS:
            return None, f"cuDeviceGetDevResource_rc_{resource_rc}"
        sm_count = int(resource.data.sm.sm_count)
        if not 1 <= sm_count <= 4096:
            return None, "device_resource_value_implausible_for_unverified_layout"
        return sm_count, DEVICE_RESOURCE_LAYOUT_REASON
    except Exception as exc:  # noqa: BLE001 - 只读探针不得崩溃
        return None, f"device_resource_query_exception:{exc.__class__.__name__}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", default="")
    parser.add_argument(
        "--attempt-device-resource",
        action="store_true",
        help="额外执行一次设备 SM 资源只读查询（仍不创建 Green Context）",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    capability = probe_green_context(
        attempt_device_resource=args.attempt_device_resource
    )
    payload = {
        **capability.to_dict(),
        "wall_time_iso": _utc_now(),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    if args.output_json:
        Path(args.output_json).write_text(encoded + "\n", encoding="utf-8")
    sys.stdout.write(encoded + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
