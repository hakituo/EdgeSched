"""SM-A 测试的共享 payload 构造（不是 pytest 收集对象）。"""

from __future__ import annotations

from typing import Any

PHYSICAL_TOTAL_SMS = 48


def client_payload(
    *,
    status: str = "ok",
    configuration_applied: bool = True,
    context_created: bool | None = True,
    context_visible_sm_count: int | None = PHYSICAL_TOTAL_SMS,
    effective_sm_count: int | None = None,
    workload_passed: bool = True,
    requested_percentage: int | None = 50,
    failure_reason: str | None = None,
) -> dict[str, Any]:
    """构造与 ``sm_client_probe`` 输出结构一致的客户端结果。

    ``context_visible_sm_count`` 是客户端自己看到的（可能被 MPS cap 限制的）
    SM 数，**不是**物理 SM 总数；物理总量由不受控基线单独提供。
    """

    return {
        "schema_version": "edgesched.sm_client_probe.v1",
        "status": status,
        "phase_id": "mps_cap_50",
        "pid": 1234,
        "expected_backend": "MPS_CAP",
        "expected_percentage": requested_percentage,
        "environment": {
            "captured_before_cuda_init": True,
            "required_keys": [],
            "values": {},
            "missing": [] if configuration_applied else ["CUDA_MPS_PIPE_DIRECTORY"],
            "configuration_applied": configuration_applied,
        },
        "cuda": {
            "torch_version": "2.6.0",
            "cuda_runtime_version": "12.8",
            "cuda_available": True,
            "initialized_at_import": False,
            "initialized_after_probe": True,
            "context_created_after_configuration": context_created,
        },
        "gpu": {
            "uuid": "GPU-abc",
            "name": "NVIDIA GeForce RTX 5070 Laptop GPU",
            "context_visible_sm_count": context_visible_sm_count,
            "context_visible_sm_count_source": (
                "torch.cuda.get_device_properties(0).multi_processor_count"
            ),
            "compute_capability": "12.0",
            "driver_version": "581.29",
            "driver_version_source": "nvml",
        },
        "sm_measurement": {
            "context_visible_sm_count": context_visible_sm_count,
            "context_effective_sm_count": effective_sm_count,
            "context_sm_count_source": (
                "cuda_driver_cuCtxGetExecAffinity"
                if effective_sm_count is not None
                else None
            ),
            "context_sm_count_unavailable_reason": (
                None
                if effective_sm_count is not None
                else "cuCtxGetExecAffinity_symbol_unavailable"
            ),
            "effective_sm_count": effective_sm_count,
            "requested_percentage_echo": requested_percentage,
            "sm_count_verified": effective_sm_count is not None,
            "verification_method": "cuda_driver_cuCtxGetExecAffinity",
            "driver_query": {},
        },
        "workload": {
            "matrix_size": 2048,
            "iterations": 2,
            "elapsed_s": 0.05,
            "synchronize_ok": True,
            "output_finite": True,
            "output_checksum": 1.5,
            "passed": workload_passed,
            "max_memory_allocated_mb": 48.0,
            "max_memory_reserved_mb": 64.0,
            "memory_allocated_mb": 32.0,
            "memory_reserved_mb": 64.0,
        },
        "failure_reason": failure_reason,
    }


def client_run(
    payload: dict[str, Any] | None = None,
    *,
    mps_pid_observed: bool = True,
    returncode: int = 0,
    json_parse_error: str | None = None,
    timed_out: bool = False,
) -> dict[str, Any]:
    """构造 ``run_client_probe`` 的返回值。"""

    return {
        "phase_id": "mps_cap_50",
        "argv": [],
        "returncode": returncode,
        "timed_out": timed_out,
        "stdout_path": "client_stdio/mps_cap_50.stdout.log",
        "stderr_path": "client_stdio/mps_cap_50.stderr.log",
        "output_json_path": "client_stdio/mps_cap_50.json",
        "payload": payload,
        "json_parse_error": json_parse_error,
        "mps_client_pids_observed": [4321] if mps_pid_observed else [],
        "mps_client_pid_observed": mps_pid_observed,
        "mps_client_pid": 4321,
        "poll_samples": 3,
    }
