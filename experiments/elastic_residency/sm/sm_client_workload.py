"""客户端探针使用的短 CUDA workload。

目的不是性能测量，而是证明当前 context 能真实提交并完成任务，
并顺带记录 PyTorch allocator 的 allocated/reserved 作为框架内显存证据。
"""

from __future__ import annotations

import time
from typing import Any

BYTES_PER_MIB = 1024.0 * 1024.0


def run_workload(
    torch_module: Any,
    *,
    matrix_size: int,
    iterations: int,
) -> dict[str, Any]:
    """在目标 context 上执行短 CUDA workload 并同步。"""

    device = torch_module.device("cuda:0")
    torch_module.cuda.reset_peak_memory_stats()
    left = torch_module.randn(
        matrix_size, matrix_size, device=device, dtype=torch_module.float32
    )
    right = torch_module.randn(
        matrix_size, matrix_size, device=device, dtype=torch_module.float32
    )
    torch_module.cuda.synchronize()
    started = time.perf_counter()
    output = None
    for _ in range(max(1, iterations)):
        output = left @ right
    torch_module.cuda.synchronize()
    elapsed_s = time.perf_counter() - started
    finite = bool(torch_module.isfinite(output).all().item())
    checksum = float(output.float().sum().item())
    del left, right
    return {
        "matrix_size": int(matrix_size),
        "iterations": int(max(1, iterations)),
        "elapsed_s": elapsed_s,
        "synchronize_ok": True,
        "output_finite": finite,
        "output_checksum": checksum,
        "passed": finite,
        "max_memory_allocated_mb": (
            torch_module.cuda.max_memory_allocated() / BYTES_PER_MIB
        ),
        "max_memory_reserved_mb": (
            torch_module.cuda.max_memory_reserved() / BYTES_PER_MIB
        ),
        "memory_allocated_mb": torch_module.cuda.memory_allocated() / BYTES_PER_MIB,
        "memory_reserved_mb": torch_module.cuda.memory_reserved() / BYTES_PER_MIB,
    }
