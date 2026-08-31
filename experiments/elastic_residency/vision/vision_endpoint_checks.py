"""端点单元的验收门禁装配（纯计算，可离线单测）。

把"测量证据"翻译成一组布尔硬门：任一为 False 即整轮失败。
门禁分四类：模型/驻留状态一致性、推理与 KV 完整性、运行期遥测有效性、
以及视觉专属的 projector 真实执行证据。
"""

from __future__ import annotations

from typing import Any

from experiments.elastic_residency.kv_measurement import buffer_report_acceptance
from experiments.elastic_residency.vision.vision_evidence import (
    output_text_checks,
    projector_execution_checks,
)


def build_acceptance_checks(
    *,
    uses_vision: bool,
    phase: dict[str, Any],
    buffer_report: dict[str, Any],
    runtime_evidence: dict[str, Any],
    resource_summary: dict[str, Any],
    capacity_snapshots: list[dict[str, Any]],
    allocator_after_release: dict[str, Any],
    physical_core_count: int,
    actual_layers: int,
    expected_layers: int,
    generation_tokens: int,
    expected_state_tokens: int,
    gpu_ready: bool,
    actual_affinity: tuple[int, ...],
    llm_affinity: tuple[int, ...],
    system_affinity: tuple[int, ...],
    expected_visual_tokens: int | None,
) -> dict[str, bool]:
    """装配单轮端点的全部硬门禁。"""

    kv_state = phase["kv_native_state"]
    inference = phase["inference"]
    checks: dict[str, bool] = {
        "model_layer_count_matches": actual_layers == expected_layers,
        "residency_state_matches_projector_state": (
            uses_vision == (phase["vision"] is not None)
        ),
        "gpu_ready_before_timeline": bool(gpu_ready),
        "gpu_present_for_every_timeline_sample": (
            resource_summary["gpu_unavailable_timeline_sample_count"] == 0
        ),
        "prefill_and_decode_positive": (
            inference["prefill_tokens_per_s"] > 0
            and inference["decode_tokens_per_s"] > 0
        ),
        "token_gap_count_exact": (
            inference["token_gap_ms"]["count"] == generation_tokens - 1
        ),
        "kv_tokens_match_executed_positions": (
            kv_state["kv_tokens_used"] == expected_state_tokens
        ),
        "kv_sequence_state_bytes_positive": kv_state["kv_sequence_state_bytes"] > 0,
        "full_state_serialized_bytes_positive": (
            kv_state["full_state_serialized_bytes"] > 0
        ),
        "kv_state_roundtrip_succeeded": bool(phase["kv_roundtrip"]["succeeded"]),
        **buffer_report_acceptance(buffer_report),
        **output_text_checks(
            output_text=phase["output"]["output_text"],
            generated_token_count=inference["generated_token_count"],
            expected_token_count=generation_tokens,
        ),
        **runtime_evidence["checks"],
        "gpu_telemetry_is_direct_nvml": (
            resource_summary["gpu_telemetry_backend"] == "direct_nvml"
            and not resource_summary["gpu_cli_subprocess_used"]
            and all(
                item["gpu_telemetry_backend"] == "direct_nvml"
                and not item["gpu_cli_subprocess_used"]
                for item in capacity_snapshots
            )
        ),
        "worker_affinity_matches": set(actual_affinity) == set(llm_affinity),
        "sampler_affinity_matches": (
            set(resource_summary["sampler_thread_affinity"]) == set(system_affinity)
        ),
        "gpu_sampler_affinity_matches": (
            set(resource_summary["gpu_thread_affinity"]) == set(system_affinity)
        ),
        "all_leases_released": allocator_after_release["active_lease_count"] == 0,
        "all_physical_cores_returned": (
            allocator_after_release["free_physical_core_count"] == physical_core_count
        ),
    }
    vision = phase["vision"]
    # 状态与实际执行不一致时上面的门禁已经失败；这里不再对空 vision 取值，
    # 保证记录仍然完整可读，而不是让子进程崩溃丢失全部证据。
    if uses_vision and vision is not None:
        checks.update(
            projector_execution_checks(
                session_evidence=vision["session"],
                chunk_evidence=vision["chunk_execution"],
                bitmap_evidence=vision["bitmaps"],
                expected_image_count=1,
                expected_visual_tokens=expected_visual_tokens,
            )
        )
        checks["projector_released_after_run"] = bool(
            vision["release"]["projector_freed"]
        )
    return checks
