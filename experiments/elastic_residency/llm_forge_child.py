"""单轮真实 LLM+Forge 共置测量。"""
from __future__ import annotations
import argparse
import base64
import datetime as dt
import gc
import hashlib
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import psutil
from experiments.elastic_residency.colocated_runtime_evidence import (
    assess_colocated_runtime,
)
from experiments.elastic_residency.cpu_topology import (
    CPUTopology,
    PhysicalCoreAllocator,
    process_affinity_scope,
)
from experiments.elastic_residency.forge_b1_reprofile import (
    build_txt2img_payload,
    forge_post,
)
from experiments.elastic_residency.forge_colocated_capacity import (
    forge_colocated_metrics,
)
from experiments.elastic_residency.gpu_telemetry import DirectNvmlProbe
from experiments.elastic_residency.kv_measurement import (
    KV_MEASUREMENT_SEMANTICS,
    LlamaInitLogCapture,
    buffer_report_acceptance,
    capture_llama_kv_state,
)
from experiments.elastic_residency.llm_residency_curve import (
    _build_exact_prompt_tokens,
    _token_ids_sha256,
    resolve_config,
)
from experiments.elastic_residency.resource_sampler import ResourceSampler
from experiments.elastic_residency.tts_gpu_snapshot import nvidia_memory_snapshot

def decode_forge_image(encoded: str) -> bytes:
    """解码 Forge base64 图像并拒绝空/非图像响应。"""
    payload = encoded.split(",", 1)[-1] if encoded.startswith("data:") else encoded
    raw = base64.b64decode(payload, validate=True)
    is_png = raw.startswith(b"\x89PNG\r\n\x1a\n")
    is_jpeg = raw.startswith(b"\xff\xd8\xff")
    if len(raw) < 1024 or not (is_png or is_jpeg):
        raise ValueError("Forge 返回内容不是有效的 PNG/JPEG 图像")
    return raw

def _window_peak(
    samples: list[Any], start_ns: int, end_ns: int, snapshots: list[dict[str, Any]]
) -> float | None:
    values = [
        float(sample.gpu_card_used_mb)
        for sample in samples
        if sample.gpu_available and start_ns <= sample.monotonic_ns <= end_ns
    ]
    values.extend(
        float(snapshot["gpu_card_used_mb"])
        for snapshot in snapshots
        if snapshot.get("gpu_card_used_mb") is not None
    )
    return max(values) if values else None

def run_child(args: argparse.Namespace) -> dict[str, Any]:
    from llama_cpp import Llama
    model_path = Path(args.llm_model).expanduser().resolve()
    image_path = Path(args.child_output_image).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"找不到 GGUF: {model_path}")
    topology = CPUTopology.discover()
    if len(topology.physical_cores) < args.system_cores + args.llm_cores:
        raise RuntimeError("物理核不足以同时分配 LLM worker 与系统采样器")
    allocator = PhysicalCoreAllocator(topology)
    system_lease = allocator.reserve_system(args.system_cores)
    llm_lease = allocator.acquire("llm_forge_colocated", args.llm_cores, use_smt=True)
    process = psutil.Process()
    process.cpu_percent(interval=None)
    n_gpu_layers = resolve_config(args.llm_config, args.model_layers)
    expected_state_tokens = args.context_tokens + args.generation_tokens
    offload_kqv = n_gpu_layers != 0
    sync_probe = DirectNvmlProbe(include_process_memory=False)
    sampler: ResourceSampler | None = None
    llm: Any = None
    child_started_ns = time.monotonic_ns()
    child_started_wall = dt.datetime.now(dt.timezone.utc).isoformat()
    try:
        with process_affinity_scope(llm_lease.affinity) as actual_affinity:
            sampler = ResourceSampler(
                interval_s=args.sample_interval,
                gpu_interval_s=args.gpu_sample_interval,
                gpu_probe=DirectNvmlProbe(include_process_memory=False),
                thread_affinity=system_lease.affinity,
            )
            gpu_ready = sampler.start(wait_for_gpu_timeout_s=args.gpu_warmup_timeout)
            forge_resident_snapshot = nvidia_memory_snapshot(
                os.getpid(), probe=sync_probe
            )
            gpu_total_mb = forge_resident_snapshot["gpu_card_total_mb"]
            forge_resident_mb = forge_resident_snapshot["gpu_card_used_mb"]
            if gpu_total_mb is None or forge_resident_mb is None:
                raise RuntimeError("无法读取 Forge 常驻整卡显存或 GPU 总量")

            load_started = time.perf_counter()
            with LlamaInitLogCapture() as log_capture:
                llm = Llama(
                    model_path=str(model_path),
                    n_gpu_layers=n_gpu_layers,
                    n_ctx=expected_state_tokens,
                    n_batch=min(expected_state_tokens, 512),
                    n_ubatch=min(expected_state_tokens, 512),
                    n_threads=len(llm_lease.affinity),
                    n_threads_batch=len(llm_lease.affinity),
                    seed=args.seed,
                    use_mmap=True,
                    offload_kqv=offload_kqv,
                    op_offload=offload_kqv,
                    flash_attn=True,
                    verbose=False,
                )
            llm_load_s = time.perf_counter() - load_started
            buffer_report = log_capture.buffer_report()
            metadata = dict(llm.metadata)
            architecture = str(metadata.get("general.architecture", ""))
            actual_layers = int(metadata[f"{architecture}.block_count"])
            prompt_tokens = _build_exact_prompt_tokens(llm, expected_state_tokens)
            llm.eval(prompt_tokens)
            kv_state = capture_llama_kv_state(llm)
            b_llm_snapshot = nvidia_memory_snapshot(os.getpid(), probe=sync_probe)
            b_llm_mb = b_llm_snapshot["gpu_card_used_mb"]
            if b_llm_mb is None:
                raise RuntimeError("LLM 共置基线不可用")

            payload = build_txt2img_payload(
                seed=args.forge_seed,
                width=args.width,
                height=args.height,
                steps=args.steps,
                cfg_scale=args.cfg_scale,
                sampler=args.forge_sampler,
            )
            request_started_ns = time.monotonic_ns()
            request_started = time.perf_counter()
            response = forge_post(
                args.forge_url.rstrip("/") + "/sdapi/v1/txt2img",
                payload,
                timeout=args.request_timeout,
            )
            forge_latency_s = time.perf_counter() - request_started
            request_finished_ns = time.monotonic_ns()
            post_forge_snapshot = nvidia_memory_snapshot(os.getpid(), probe=sync_probe)
            images = response.get("images", [])
            if len(images) != 1:
                raise RuntimeError(f"Forge 应返回 1 张图，实际 {len(images)}")
            image_bytes = decode_forge_image(images[0])
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(image_bytes)

            llm.close()
            llm = None
            gc.collect()
            resource_summary = sampler.stop()
            timeline = [asdict(sample) for sample in sampler.samples]
        llm_lease.release()
        system_lease.release()
        allocator_after_release = allocator.snapshot()
    finally:
        if llm is not None:
            llm.close()
        if sampler is not None and sampler._thread is not None and sampler._thread.is_alive():
            sampler.stop()
        llm_lease.release()
        system_lease.release()

    p_colocated_mb = _window_peak(
        sampler.samples,
        request_started_ns,
        request_finished_ns,
        [b_llm_snapshot, post_forge_snapshot],
    )
    if p_colocated_mb is None:
        raise RuntimeError("Forge 请求窗口没有可用 GPU 峰值")
    runtime_evidence = assess_colocated_runtime(
        sampler.samples,
        start_ns=request_started_ns,
        end_ns=request_finished_ns,
        capacity_snapshots=[
            forge_resident_snapshot,
            b_llm_snapshot,
            post_forge_snapshot,
        ],
        max_gpu_sample_age_ms=args.max_gpu_sample_age_ms,
        max_swap_growth_bytes=int(args.max_swap_growth_mb * 1024 * 1024),
    )
    metrics = forge_colocated_metrics(
        global_idle_reference_mb=args.global_idle_reference_mb,
        b_llm_mb=b_llm_mb,
        p_colocated_mb=p_colocated_mb,
        gpu_total_mb=gpu_total_mb,
        safety_margin_mb=args.safety_margin_mb,
        llm_isolated_increment_mb=args.llm_isolated_increment_mb,
        forge_isolated_increment_mb=args.forge_isolated_increment_mb,
    )
    buffer_checks = buffer_report_acceptance(buffer_report)
    acceptance_checks = {
        "model_layer_count_matches": actual_layers == args.model_layers,
        "state_token_count_exact": len(prompt_tokens) == expected_state_tokens,
        "kv_position_bounds_valid": (
            kv_state["kv_position_min"] >= 0
            and kv_state["kv_position_max"] == expected_state_tokens - 1
            and kv_state["kv_tokens_used"] == expected_state_tokens
        ),
        "kv_sequence_state_bytes_positive": kv_state["kv_sequence_state_bytes"] > 0,
        "full_state_serialized_bytes_positive": (
            kv_state["full_state_serialized_bytes"] > 0
        ),
        **buffer_checks,
        "forge_resident_baseline_matches_profile": abs(
            forge_resident_mb - args.expected_forge_resident_baseline_mb
        )
        <= args.max_forge_resident_drift_mb,
        "gpu_total_matches_screen": abs(gpu_total_mb - args.expected_gpu_total_mb) <= 1,
        "forge_returned_single_valid_image": image_path.is_file()
        and len(image_bytes) >= 1024,
        "capacity_admissible": metrics["capacity_admissible"],
        "gpu_ready_before_timeline": gpu_ready,
        "gpu_present_for_every_timeline_sample": (
            resource_summary["gpu_unavailable_timeline_sample_count"] == 0
        ),
        **runtime_evidence["checks"],
        "gpu_telemetry_is_direct_nvml": (
            resource_summary["gpu_telemetry_backend"] == "direct_nvml"
            and not resource_summary["gpu_cli_subprocess_used"]
            and all(
                item["gpu_telemetry_backend"] == "direct_nvml"
                and not item["gpu_cli_subprocess_used"]
                for item in (
                    forge_resident_snapshot,
                    b_llm_snapshot,
                    post_forge_snapshot,
                )
            )
        ),
        "worker_affinity_matches": set(actual_affinity) == set(llm_lease.affinity),
        "sampler_affinity_matches": set(resource_summary["sampler_thread_affinity"])
        == set(system_lease.affinity),
        "gpu_sampler_affinity_matches": set(resource_summary["gpu_thread_affinity"])
        == set(system_lease.affinity),
        "all_leases_released": allocator_after_release["active_lease_count"] == 0,
        "all_physical_cores_returned": (
            allocator_after_release["free_physical_core_count"]
            == len(topology.physical_cores)
        ),
    }
    return {
        "config": f"{args.llm_config}@C{args.context_tokens}+Forge",
        "context_tokens": args.context_tokens,
        "generation_tokens": args.generation_tokens,
        "expected_state_tokens": expected_state_tokens,
        "llm_config": args.llm_config,
        "run_index": args.run_index,
        "n_gpu_layers": n_gpu_layers,
        "passed": all(acceptance_checks.values()),
        "acceptance_checks": acceptance_checks,
        "backend_kind": "real_llama_cpp_gguf_plus_forge_http_api",
        "fallback_occurred": False,
        "kv_state": kv_state,
        "kv_measurement_semantics": KV_MEASUREMENT_SEMANTICS,
        "prompt_tokens_sha256": _token_ids_sha256(prompt_tokens),
        "llm_load_s": llm_load_s,
        "forge_latency_s": forge_latency_s,
        "forge_image": {
            "path": str(image_path),
            "size_bytes": len(image_bytes),
            "sha256": hashlib.sha256(image_bytes).hexdigest(),
        },
        "memory_forge_resident": forge_resident_snapshot,
        "memory_b_llm": b_llm_snapshot,
        "memory_post_forge": post_forge_snapshot,
        "colocated_metrics": metrics,
        "runtime_evidence": runtime_evidence,
        "resource_summary": resource_summary,
        "allocator_after_release": allocator_after_release,
        "llm_logical_affinity": list(actual_affinity),
        "system_logical_affinity": list(system_lease.affinity),
        "phase_intervals_ns": {
            "forge_request": [request_started_ns, request_finished_ns]
        },
        "child_started_monotonic_ns": child_started_ns,
        "child_started_wall_iso": child_started_wall,
        "child_finished_monotonic_ns": time.monotonic_ns(),
        "child_finished_wall_iso": dt.datetime.now(dt.timezone.utc).isoformat(),
        "timeline": timeline,
    }
