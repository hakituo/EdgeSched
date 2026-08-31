"""端点运行契约：冻结运行配置与单轮证据记录的装配。

这里只做结构装配，不做任何测量或判定；所有数值都由调用方从真实运行中采集。
把装配集中在一处，避免 run_config、manifest 与 iteration 字段各自漂移。
"""

from __future__ import annotations

import argparse
from typing import Any

from experiments.elastic_residency.contracts import (
    BackendEvidence,
    ExperimentRunConfig,
)
from experiments.elastic_residency.kv_measurement import KV_MEASUREMENT_SEMANTICS
from experiments.elastic_residency.vision.vision_evidence import (
    VISION_MEASUREMENT_SEMANTICS,
)


def build_run_config(
    args: argparse.Namespace,
    *,
    gate: dict[str, Any],
    model_layers: int,
    expected_records: int,
) -> ExperimentRunConfig:
    """冻结本次运行的真实后端证据与参数。"""

    artifacts = gate["artifacts"]
    return ExperimentRunConfig(
        experiment_name="vision_endpoint_smoke_v1",
        run_id=args.run_id,
        seed=args.seed,
        sample_interval_s=args.sample_interval,
        gpu_sample_interval_s=args.gpu_sample_interval,
        backends=(
            BackendEvidence(
                component="llm",
                backend_kind="real_llama_cpp_gguf",
                is_real=True,
                model_path=artifacts["main_model"]["path"],
                model_sha256=artifacts["main_model"]["sha256"] or "",
            ),
            BackendEvidence(
                component="vision_projector",
                backend_kind="real_libmtmd_mmproj",
                is_real=True,
                model_path=artifacts["projector"]["path"],
                model_sha256=artifacts["projector"]["sha256"] or "",
                details={
                    "projector_type": gate["projector_profile"]["projector_type"]
                },
            ),
        ),
        parameters={
            "configs": [name.upper() for name in args.configs],
            "residency_states": list(args.residency_states),
            "runs": args.runs,
            "model_layers": model_layers,
            "context_tokens": args.context_tokens,
            "generation_tokens": args.generation_tokens,
            "n_ctx": args.context_tokens + args.generation_tokens,
            "prompt_mode": args.prompt_mode,
            "vision_eval_mode": args.vision_eval_mode,
            "question": args.question,
            "image_path": artifacts["image"]["path"],
            "image_file_sha256": artifacts["image"]["file_sha256"],
            "image_pixel_payload_sha256": artifacts["image"]["pixel_payload_sha256"],
            "image_max_tokens": args.image_max_tokens,
            "image_min_tokens": args.image_min_tokens,
            "expected_visual_tokens": gate["expected_visual_tokens"],
            "expected_records": expected_records,
            "llm_physical_cores": args.llm_cores,
            "system_physical_cores": args.system_cores,
            "gpu_telemetry_backend": "direct_nvml",
            "gpu_cli_subprocess_allowed": False,
            "fixed_generation_ignores_eos": True,
            "acceptance_thresholds": {
                "max_gpu_sample_age_ms": args.max_gpu_sample_age_ms,
                "max_swap_growth_mb": args.max_swap_growth_mb,
            },
        },
    )


def build_record(
    *,
    args: argparse.Namespace,
    phase: dict[str, Any],
    measurement: dict[str, Any],
) -> dict[str, Any]:
    """把一轮真实测量装配为可复查的 iteration 记录。"""

    uses_vision = phase["vision"] is not None
    checks = measurement["acceptance_checks"]
    buffer_report = measurement["buffer_report"]
    memory = measurement["memory"]
    return {
        "config": args.child_config,
        "residency_state": args.residency_state,
        "run_index": args.run_index,
        "passed": all(checks.values()),
        "acceptance_checks": checks,
        "backend_kind": (
            "real_llama_cpp_gguf_plus_libmtmd_vision"
            if uses_vision
            else "real_llama_cpp_gguf_text_only"
        ),
        "fallback_occurred": False,
        "n_gpu_layers": measurement["n_gpu_layers"],
        "offload_kqv": measurement["offload_kqv"],
        "op_offload": measurement["offload_kqv"],
        "flash_attn": True,
        "model_layers": measurement["actual_layers"],
        "model_architecture": measurement["architecture"],
        "context_tokens": args.context_tokens,
        "generation_tokens": args.generation_tokens,
        "n_ctx": measurement["n_ctx"],
        "vision_eval_mode": args.vision_eval_mode if uses_vision else None,
        "llm_physical_cores": args.llm_cores,
        "llm_logical_affinity": list(measurement["llm_logical_affinity"]),
        "llm_threads": measurement["llm_threads"],
        "system_physical_cores": args.system_cores,
        "system_logical_affinity": list(measurement["system_logical_affinity"]),
        "llm_load_s": measurement["llm_load_s"],
        "unload_s": measurement["unload_s"],
        "prompt": phase["prompt"],
        "inference": phase["inference"],
        "output": phase["output"],
        "kv_native_state": phase["kv_native_state"],
        "kv_roundtrip": phase["kv_roundtrip"],
        "expected_state_tokens": measurement["expected_state_tokens"],
        "kv_gpu_buffer_mb": buffer_report.get("kv_gpu_buffer_mb"),
        "kv_cpu_buffer_mb": buffer_report.get("kv_cpu_buffer_mb"),
        "compute_gpu_buffer_mb": buffer_report.get("compute_gpu_buffer_mb"),
        "compute_cpu_buffer_mb": buffer_report.get("compute_cpu_buffer_mb"),
        "kv_measurement_semantics": KV_MEASUREMENT_SEMANTICS,
        "vision_measurement_semantics": VISION_MEASUREMENT_SEMANTICS,
        "vision": phase["vision"],
        "memory_baseline": memory["baseline"],
        "memory_baseline_rss_bytes": memory["baseline_rss"],
        "memory_after_llm_load": memory["after_llm"],
        "memory_after_llm_load_rss_bytes": memory["after_llm_rss"],
        "memory_after_inference": memory["after_inference"],
        "memory_after_inference_rss_bytes": memory["after_inference_rss"],
        "memory_after_unload": memory["after_unload"],
        "memory_after_unload_rss_bytes": memory["after_unload_rss"],
        "runtime_evidence": measurement["runtime_evidence"],
        "resource_summary": measurement["resource_summary"],
        "resource_by_phase": measurement["resource_by_phase"],
        "phase_intervals_ns": measurement["phase_intervals_ns"],
        "allocator_after_release": measurement["allocator_after_release"],
        "child_started_monotonic_ns": measurement["child_started_monotonic_ns"],
        "child_started_wall_iso": measurement["child_started_wall_iso"],
        "child_finished_monotonic_ns": measurement["child_finished_monotonic_ns"],
        "child_finished_wall_iso": measurement["child_finished_wall_iso"],
        "timeline": measurement["timeline"],
    }
