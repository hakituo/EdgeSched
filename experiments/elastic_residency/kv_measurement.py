"""KV 显存测量语义与 llama.cpp 原生接口采集。

本模块把以下五个互不相同的概念显式拆开，任何实验记录不得混用：

1. GPU 预分配 KV buffer（容量，来自 llama.cpp 初始化日志）；
2. 当前序列的原生 position 边界与 llama-cpp-python 已评估 token 数；
3. 单序列序列化状态大小（``llama_state_seq_get_size``）；
4. 完整 ``save_state()`` 迁移负载（含 logits、embedding 与 KV memory）；
5. 整卡显存增量（direct NVML，隔离环境差值估计，不是精确进程显存）。

不可用或缺失的字段必须保留 ``None`` 或直接失败，禁止填 0 冒充。
"""

from __future__ import annotations

import ctypes
import re
import threading
from typing import Any, Callable, Sequence


BYTES_PER_MIB = 1024.0 * 1024.0

KV_MEASUREMENT_SEMANTICS: dict[str, str] = {
    "full_state_serialized_bytes": (
        "llama_state_get_size(ctx)，等于 llm.save_state().llama_state_size；"
        "包含 logits、embedding 与 KV memory 的完整序列化迁移负载，"
        "不是纯 GPU KV 显存。"
    ),
    "kv_sequence_state_bytes": (
        "llama_state_seq_get_size(ctx, seq_id)：单序列状态序列化大小，"
        "包含该序列 KV 与相关运行状态；不等于 GPU 预分配 KV buffer。"
    ),
    "kv_position_min": (
        "llama_memory_seq_pos_min(mem, seq_id)：当前缓存中该序列的最小 token "
        "position；无任何 token 时为 -1。统一 KV memory 下它可能只反映"
        "显式关联到 seq_id 的边界，不能据此推算 token 数。"
    ),
    "kv_position_max": (
        "llama_memory_seq_pos_max(mem, seq_id)：当前缓存中该序列的最大 token "
        "position；无任何 token 时为 -1。"
    ),
    "kv_tokens_used": (
        "llama-cpp-python llm.n_tokens：本实验单序列、无 context shift 条件下"
        "已评估并保留在上下文中的 token 数，是使用量而不是容量。"
    ),
    "kv_gpu_buffer_mb": (
        "llama.cpp 初始化日志中 CUDA 设备（CUDA0 等）预分配 KV buffer 之和；"
        "由 n_ctx 决定的容量，与已用 token 数无关；不含 CUDA_Host。"
    ),
    "kv_cpu_buffer_mb": (
        "llama.cpp 初始化日志中 CPU/CUDA_Host 宿主侧预分配 KV buffer 之和；"
        "CPU 容量不计入 GPU 卡上显存。"
    ),
    "compute_gpu_buffer_mb": (
        "llama.cpp 初始化日志中 CUDA 设备（CUDA0 等）compute buffer 之和；"
        "不含 CUDA_Host。"
    ),
    "compute_cpu_buffer_mb": (
        "llama.cpp 初始化日志中 CPU/CUDA_Host 宿主侧 compute buffer 之和。"
    ),
    "kv_buffer_mb_by_device": (
        "按设备名分列的 KV buffer 容量（MiB），含 CUDA_Host 等宿主侧缓冲；"
        "设备未出现时该键不存在，不填 0。"
    ),
    "compute_buffer_mb_by_device": (
        "按设备名分列的 compute buffer 容量（MiB）；设备未出现时该键不存在。"
    ),
    "kv_state_bytes": (
        "兼容字段：与 full_state_serialized_bytes 相同的完整序列化状态大小，"
        "历史曲线用该名承载同一语义。"
    ),
    "gpu_card_delta_from_baseline_mb": (
        "direct NVML 整卡显存相对加载前基线的差值；仅在隔离环境下才可"
        "近似为该负载的显存占用，不是精确进程显存。"
    ),
}


_BUFFER_LINE_PATTERN = re.compile(
    r"(?P<device>\S+)\s+(?P<kind>KV|compute)\s+buffer\s+size\s*=\s*"
    r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>[KMG]i?B|B)?"
)

_UNIT_TO_BYTES: dict[str | None, float] = {
    None: 1.0,
    "B": 1.0,
    "KiB": 1024.0,
    "MiB": 1024.0**2,
    "GiB": 1024.0**3,
    "KB": 1000.0,
    "MB": 1000.0**2,
    "GB": 1000.0**3,
}

_GPU_DEVICE_PATTERN = re.compile(r"^CUDA\d+$")
_CPU_DEVICE_NAME = "CPU"


def position_span_from_bounds(pos_min: int, pos_max: int) -> int:
    """计算原生 position 边界跨度；该跨度不等于 KV 已用 token 数。"""

    if pos_min is None or pos_max is None:
        raise ValueError("position 不可用时不得计算边界跨度")
    if pos_min < 0 or pos_max < pos_min:
        return 0
    return pos_max - pos_min + 1


def parse_llama_buffer_lines(lines: Sequence[str]) -> dict[str, Any]:
    """解析 llama.cpp 初始化日志中的 KV/compute buffer 尺寸行。

    返回按设备分列的容量（MiB）与四个汇总字段；对应设备未在日志出现时
    汇总字段为 ``None``，绝不填 0 冒充。
    """

    kv_by_device: dict[str, float] = {}
    compute_by_device: dict[str, float] = {}
    matched_lines: list[str] = []
    for line in lines:
        match = _BUFFER_LINE_PATTERN.search(line)
        if match is None:
            continue
        value_bytes = float(match.group("value")) * _UNIT_TO_BYTES.get(
            match.group("unit"), 1.0
        )
        device = match.group("device")
        target = kv_by_device if match.group("kind") == "KV" else compute_by_device
        target[device] = target.get(device, 0.0) + value_bytes / BYTES_PER_MIB
        matched_lines.append(line.rstrip("\n"))
    return {
        "kv_buffer_mb_by_device": kv_by_device,
        "compute_buffer_mb_by_device": compute_by_device,
        "kv_gpu_buffer_mb": _summarize_devices(kv_by_device),
        "kv_cpu_buffer_mb": _summarize_host_devices(kv_by_device),
        "compute_gpu_buffer_mb": _summarize_devices(compute_by_device),
        "compute_cpu_buffer_mb": _summarize_host_devices(compute_by_device),
        "matched_line_count": len(matched_lines),
    }


def _summarize_devices(by_device: dict[str, float]) -> float | None:
    values = [
        value
        for device, value in by_device.items()
        if _GPU_DEVICE_PATTERN.match(device)
    ]
    return sum(values) if values else None


def _summarize_host_devices(by_device: dict[str, float]) -> float | None:
    values = [
        value
        for device, value in by_device.items()
        if device == _CPU_DEVICE_NAME or device.endswith("_Host")
    ]
    return sum(values) if values else None


def buffer_report_acceptance(report: dict[str, Any] | None) -> dict[str, bool]:
    """校验 buffer 报告是否成功捕获了必需的容量字段。"""

    if not isinstance(report, dict):
        return {
            "kv_buffer_size_captured": False,
            "compute_buffer_size_captured": False,
        }
    return {
        "kv_buffer_size_captured": (
            report.get("kv_gpu_buffer_mb") is not None
            or report.get("kv_cpu_buffer_mb") is not None
        ),
        "compute_buffer_size_captured": (
            report.get("compute_gpu_buffer_mb") is not None
            or report.get("compute_cpu_buffer_mb") is not None
        ),
    }


def capture_llama_kv_state(
    llm: Any,
    *,
    seq_id: int = 0,
    api: Any | None = None,
) -> dict[str, Any]:
    """在推理计时窗口之外查询 llama.cpp 原生 KV/状态接口。

    任何查询不可用都直接抛错，不得填 0 冒充。
    """

    if api is None:
        import llama_cpp as api  # noqa: PLC0415

    context = getattr(llm, "_ctx", None)
    ctx_handle = getattr(context, "ctx", None)
    if ctx_handle is None:
        raise RuntimeError("无法获取 llama.cpp 上下文句柄，KV 状态查询失败")
    memory = getattr(context, "memory", None)
    if not memory:
        get_memory = getattr(api, "llama_get_memory", None)
        if get_memory is None:
            raise RuntimeError("llama.cpp 绑定缺少 llama_get_memory，无法查询 KV memory")
        memory = get_memory(ctx_handle)
    if not memory:
        raise RuntimeError("llama_get_memory 返回空 KV memory，禁止继续测量")
    full_state = api.llama_state_get_size(ctx_handle)
    seq_state = api.llama_state_seq_get_size(ctx_handle, seq_id)
    pos_min = api.llama_memory_seq_pos_min(memory, seq_id)
    pos_max = api.llama_memory_seq_pos_max(memory, seq_id)
    evaluated_tokens = getattr(llm, "n_tokens", None)
    if (
        full_state is None
        or seq_state is None
        or pos_min is None
        or pos_max is None
        or evaluated_tokens is None
    ):
        raise RuntimeError("llama.cpp 返回不可用的 KV 查询结果，禁止填 0 冒充")
    full_state = int(full_state)
    seq_state = int(seq_state)
    pos_min = int(pos_min)
    pos_max = int(pos_max)
    evaluated_tokens = int(evaluated_tokens)
    if evaluated_tokens < 0:
        raise RuntimeError("llm.n_tokens 为负数，KV token 使用量不可用")
    return {
        "seq_id": int(seq_id),
        "full_state_serialized_bytes": full_state,
        "kv_sequence_state_bytes": seq_state,
        "kv_position_min": pos_min,
        "kv_position_max": pos_max,
        "kv_tokens_used": evaluated_tokens,
        "kv_tokens_used_source": "llama_cpp_python.n_tokens",
        "kv_position_span": position_span_from_bounds(pos_min, pos_max),
    }


class LlamaInitLogCapture:
    """临时接管 llama.cpp 日志回调，采集初始化日志中的 buffer 尺寸。

    只在 ``with`` 语句内收集日志，不向 stderr 打印；退出时恢复
    llama-cpp-python 默认回调。不改变任何模型初始化参数。
    """

    def __init__(
        self,
        *,
        log_set: Callable[..., None] | None = None,
    ) -> None:
        self._custom_log_set = log_set
        self._lines: list[str] = []
        self._active = False
        self._lock = threading.Lock()
        self._callback: Any = None

    def __enter__(self) -> "LlamaInitLogCapture":
        import llama_cpp  # noqa: PLC0415

        log_set = self._custom_log_set or llama_cpp.llama_log_set

        @llama_cpp.llama_log_callback
        def _capture_log(level: int, text: bytes, user_data: Any) -> None:
            if not self._active:
                return
            with self._lock:
                self._lines.append(text.decode("utf-8", errors="replace"))

        self._callback = _capture_log
        log_set(self._callback, ctypes.c_void_p(0))
        self._active = True
        return self

    def __exit__(self, *_: object) -> None:
        self._active = False
        if self._custom_log_set is not None:
            self._custom_log_set(None, None)
            return
        import llama_cpp  # noqa: PLC0415

        try:
            from llama_cpp import _logger  # noqa: PLC0415

            llama_cpp.llama_log_set(
                _logger.llama_log_callback, ctypes.c_void_p(0)
            )
        except (ImportError, AttributeError):
            # 恢复默认回调不可用时退回 libllama 默认 stderr 行为。
            llama_cpp.llama_log_set(None, ctypes.c_void_p(0))

    @property
    def lines(self) -> list[str]:
        with self._lock:
            return list(self._lines)

    def buffer_report(self) -> dict[str, Any]:
        return parse_llama_buffer_lines(self.lines)
