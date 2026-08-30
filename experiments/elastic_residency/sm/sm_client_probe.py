#!/usr/bin/env python3
"""GPU 计算份额客户端探针（必须作为独立进程运行）。

执行顺序是不可协商的：

1. 先捕获并确认 SM/MPS 环境变量已经存在；
2. **之后**才初始化 CUDA / 导入 torch 的 CUDA 后端；
3. 读取物理 GPU 信息（UUID、名称、CUDA/driver、compute capability）；
4. 读取客户端实际看到的 multiprocessor count；
5. 通过 CUDA Driver API 查询当前 context 的
   ``CU_EXEC_AFFINITY_TYPE_SM_COUNT``；
6. 执行短 CUDA workload 并同步，证明该 context 能真实提交并完成任务；
7. 向 stdout 输出**单个**结构化 JSON。

``effective_sm_count`` 只能来自客户端 CUDA context 的实际查询。
``round(total_sms * percentage / 100)`` 一类的公式只生成请求/预期值，
永远不是实测结果；未查询到时保持 ``None``，禁止用 0 伪装。
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import sys
import time
import uuid
from pathlib import Path
from typing import Any

# 关键：环境变量快照必须在任何 CUDA/torch 导入之前完成，因此本模块
# 只允许在此处导入标准库，torch 一律在函数内部延迟导入。
ENVIRONMENT_SNAPSHOT: dict[str, str] = dict(os.environ)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.elastic_residency.sm.cuda_exec_affinity import (  # noqa: E402
    query_context_exec_affinity_sm_count,
    resolve_driver_version,
)
from experiments.elastic_residency.sm.mps_commands import (  # noqa: E402
    ENV_ACTIVE_THREAD_PERCENTAGE,
)
from experiments.elastic_residency.sm.sm_client_workload import (  # noqa: E402
    run_workload,
)
from experiments.elastic_residency.sm.sm_profiles import (  # noqa: E402
    GpuComputeBackend,
)

CLIENT_PROBE_SCHEMA_VERSION = "edgesched.sm_client_probe.v1"

#: ``multi_processor_count`` 是 context 可见值，不是物理总量。
CONTEXT_VISIBLE_SM_SOURCE = (
    "torch.cuda.get_device_properties(0).multi_processor_count"
)


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _device_uuid(torch_module: Any, properties: Any) -> str | None:
    """尽量把设备 UUID 转成字符串；无法可靠转换时返回 ``None``。"""

    getter = getattr(torch_module.cuda, "get_device_uuid", None)
    if getter is not None:
        try:
            text = str(getter(0)).strip()
        except Exception:  # noqa: BLE001 - 环境诊断字段，失败保持 None
            text = ""
        if text:
            return text
    raw = getattr(properties, "uuid", None)
    if raw is None:
        return None
    for extractor in (lambda: str(uuid.UUID(bytes=bytes(raw))), lambda: str(raw)):
        try:
            text = extractor().strip()
        except Exception:  # noqa: BLE001 - 结构体布局未知时保持 None
            continue
        if text and "object at 0x" not in text:
            return text
    return None


def _json_fallback(value: Any) -> str:
    """避免单个不可序列化字段让整份证据丢失。"""

    return f"<unserializable:{type(value).__name__}>"


def _empty_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": CLIENT_PROBE_SCHEMA_VERSION,
        "status": "failed",
        "phase_id": args.phase_id,
        "pid": os.getpid(),
        "wall_time_iso": _utc_now(),
        "python": platform.python_version(),
        "expected_backend": args.expected_backend,
        "expected_percentage": args.expected_percentage,
        "environment": {},
        "cuda": {
            "torch_version": None,
            "cuda_runtime_version": None,
            "cuda_available": None,
            "initialized_at_import": None,
            "initialized_after_probe": None,
            "context_created_after_configuration": None,
        },
        "gpu": {
            "uuid": None,
            "name": None,
            # 注意：MPS active-thread percentage 生效后，该属性反映**当前 context
            # 可见**的 SM 数，可能是受限值，不是设备物理 SM 总数。
            "context_visible_sm_count": None,
            "context_visible_sm_count_source": (
                CONTEXT_VISIBLE_SM_SOURCE
            ),
            "compute_capability": None,
            "driver_version": None,
            "driver_version_source": None,
        },
        "sm_measurement": {
            "context_visible_sm_count": None,
            "context_effective_sm_count": None,
            "context_sm_count_source": None,
            "context_sm_count_unavailable_reason": None,
            "effective_sm_count": None,
            "sm_count_verified": False,
            "verification_method": "unverified",
        },
        "workload": None,
        "failure_reason": None,
    }


def _check_environment(
    args: argparse.Namespace,
    payload: dict[str, Any],
) -> tuple[dict[str, str], list[str], int | None]:
    required = tuple(args.require_env or ())
    values = {key: ENVIRONMENT_SNAPSHOT.get(key) for key in required}
    missing = sorted(key for key, value in values.items() if value in (None, ""))
    payload["environment"] = {
        "captured_before_cuda_init": True,
        "required_keys": list(required),
        "values": values,
        "missing": missing,
        "configuration_applied": not missing,
    }
    if args.expected_percentage is not None:
        return values, missing, int(args.expected_percentage)
    raw = ENVIRONMENT_SNAPSHOT.get(ENV_ACTIVE_THREAD_PERCENTAGE)
    return values, missing, (int(raw) if raw not in (None, "") else None)


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    payload = _empty_payload(args)
    _values, missing_env, requested_percentage = _check_environment(args, payload)
    if missing_env:
        payload["failure_reason"] = (
            "required_environment_missing_before_cuda_init:" + ",".join(missing_env)
        )
        return payload

    try:
        import torch  # noqa: PLC0415 - 必须晚于环境变量确认
    except Exception as exc:  # noqa: BLE001 - 导入失败也要输出结构化结果
        payload["failure_reason"] = f"torch_import_failed:{exc.__class__.__name__}"
        return payload

    initialized_at_import = bool(torch.cuda.is_initialized())
    payload["cuda"]["torch_version"] = str(torch.__version__)
    payload["cuda"]["cuda_runtime_version"] = str(
        getattr(torch.version, "cuda", "") or ""
    )
    payload["cuda"]["initialized_at_import"] = initialized_at_import
    if not torch.cuda.is_available():
        payload["failure_reason"] = "cuda_device_unavailable"
        return payload
    payload["cuda"]["cuda_available"] = True

    properties = torch.cuda.get_device_properties(0)
    # MPS 生效后该属性可能反映受限后的可用数量，因此只记为 context 可见值；
    # 物理 SM 总数由不受控 CUDA 基线提供，这里绝不冒充物理总量。
    context_visible_sm_count = (
        int(getattr(properties, "multi_processor_count", 0)) or None
    )
    compute_capability = (
        f"{int(properties.major)}.{int(properties.minor)}"
        if getattr(properties, "major", None) is not None
        else None
    )
    driver_version, driver_source = resolve_driver_version()
    torch.cuda.init()
    context_created = not initialized_at_import and bool(torch.cuda.is_initialized())
    payload["cuda"]["initialized_after_probe"] = bool(torch.cuda.is_initialized())
    payload["cuda"]["context_created_after_configuration"] = context_created
    payload["gpu"].update(
        uuid=_device_uuid(torch, properties),
        name=str(getattr(properties, "name", "") or None),
        context_visible_sm_count=context_visible_sm_count,
        compute_capability=compute_capability,
        driver_version=driver_version,
        driver_version_source=driver_source,
    )

    # 在当前线程真实提交一次，确保 torch 的 context 对 CUDA Driver API 可见，
    # 否则 cuCtxGetExecAffinity 会因为缺少 current context 返回 201。
    warmup = torch.zeros(1, device=torch.device("cuda:0"))
    warmup.add_(1.0)
    torch.cuda.synchronize()
    del warmup

    affinity = query_context_exec_affinity_sm_count()
    effective_sm_count = affinity["context_effective_sm_count"]
    payload["sm_measurement"].update(
        context_visible_sm_count=context_visible_sm_count,
        context_effective_sm_count=effective_sm_count,
        context_sm_count_source=affinity["context_sm_count_source"],
        context_sm_count_unavailable_reason=affinity[
            "context_sm_count_unavailable_reason"
        ],
        effective_sm_count=effective_sm_count,
        requested_percentage_echo=requested_percentage,
        sm_count_verified=effective_sm_count is not None,
        verification_method=affinity["context_sm_count_source"] or "no_context_sm_query",
        driver_query=affinity,
    )

    workload = run_workload(
        torch, matrix_size=args.matrix_size, iterations=args.iterations
    )
    payload["workload"] = workload
    if args.hold_seconds > 0:
        # 保持 context 存活，让父进程有机会在 MPS client list 中观察到本 PID。
        time.sleep(args.hold_seconds)

    if not workload["passed"]:
        payload["failure_reason"] = "cuda_workload_output_not_finite"
        return payload
    if context_created is not True:
        payload["failure_reason"] = "cuda_context_not_created_after_configuration"
        return payload
    payload["status"] = "ok"
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--phase-id", default="client")
    parser.add_argument(
        "--expected-backend",
        default=GpuComputeBackend.UNMANAGED.value,
        choices=tuple(item.value for item in GpuComputeBackend),
    )
    parser.add_argument("--expected-percentage", type=int)
    parser.add_argument(
        "--require-env",
        nargs="*",
        default=[],
        help="必须在 CUDA 初始化之前存在的环境变量名",
    )
    parser.add_argument("--matrix-size", type=int, default=2048)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=0.0,
        help="workload 后保持 context 存活的秒数，便于父进程观察 MPS client",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = run_probe(args)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
        default=_json_fallback,
    )
    if args.output_json:
        Path(args.output_json).write_text(encoded + "\n", encoding="utf-8")
    sys.stdout.write(encoded + "\n")
    return 0 if payload["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
