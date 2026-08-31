"""单轮真实端点测量子进程：模型加载、驻留状态执行、验收装配。

一个子进程只测一个 `(驻留档位, 驻留状态)` 单元，结束后归还物理核并卸载模型，
避免跨单元的显存与页缓存污染。
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
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
from experiments.elastic_residency.gpu_telemetry import DirectNvmlProbe
from experiments.elastic_residency.kv_measurement import LlamaInitLogCapture
from experiments.elastic_residency.llm_residency_curve import (
    _phase_resource_metrics,
    resolve_config,
)
from experiments.elastic_residency.resource_sampler import ResourceSampler
from experiments.elastic_residency.tts_gpu_snapshot import nvidia_memory_snapshot
from experiments.elastic_residency.vision.vision_endpoint_checks import (
    build_acceptance_checks,
)
from experiments.elastic_residency.vision.vision_endpoint_contracts import (
    build_record,
)
from experiments.elastic_residency.vision.vision_endpoint_phases import (
    VisionPhaseConfig,
    run_text_phase,
    run_vision_phase,
)
from experiments.elastic_residency.vision.vision_evidence import (
    RESIDENCY_TEXT_ONLY_CHAT,
    VISION_STATES,
    build_prompt_evidence,
)


def _residency_uses_vision(state: str) -> bool:
    return state in VISION_STATES


def run_child(args: argparse.Namespace) -> dict[str, Any]:
    """执行一个端点单元并返回完整证据记录。"""

    from llama_cpp import Llama

    state = args.residency_state
    model_path = Path(args.model).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"找不到 GGUF: {model_path}")
    uses_vision = _residency_uses_vision(state)
    mmproj_path = Path(args.mmproj).expanduser().resolve() if args.mmproj else None
    if uses_vision and (mmproj_path is None or not mmproj_path.is_file()):
        raise FileNotFoundError(f"视觉状态需要可用 mmproj: {mmproj_path}")
    image_path = Path(args.image).expanduser().resolve() if args.image else None
    if uses_vision and (image_path is None or not image_path.is_file()):
        raise FileNotFoundError(f"视觉状态需要可用图片: {image_path}")

    topology = CPUTopology.discover()
    required_cores = args.system_cores + args.llm_cores
    if len(topology.physical_cores) < required_cores:
        raise RuntimeError(
            f"需要 {required_cores} 个物理核，实际只有 {len(topology.physical_cores)}"
        )
    n_gpu_layers = resolve_config(args.child_config, args.model_layers)
    offload_kqv = n_gpu_layers != 0
    n_ctx = args.context_tokens + args.generation_tokens
    allocator = PhysicalCoreAllocator(topology)
    system_lease = allocator.reserve_system(args.system_cores)
    llm_lease = allocator.acquire("vision_endpoint", args.llm_cores, use_smt=True)
    process = psutil.Process()
    process.cpu_percent(interval=None)
    sync_probe = DirectNvmlProbe(include_process_memory=False)

    def snapshot() -> tuple[dict[str, Any], int]:
        return (
            nvidia_memory_snapshot(os.getpid(), probe=sync_probe),
            int(process.memory_info().rss),
        )

    baseline_snapshot, baseline_rss = snapshot()
    if baseline_snapshot["gpu_card_used_mb"] is None:
        raise RuntimeError("无法读取整卡显存基线，禁止继续测量")

    sampler: ResourceSampler | None = None
    llm: Any = None
    child_started_ns = time.monotonic_ns()
    child_started_wall = dt.datetime.now(dt.timezone.utc).isoformat()
    capacity_snapshots: list[dict[str, Any]] = [baseline_snapshot]
    try:
        with process_affinity_scope(llm_lease.affinity) as actual_affinity:
            sampler = ResourceSampler(
                interval_s=args.sample_interval,
                gpu_interval_s=args.gpu_sample_interval,
                gpu_probe=DirectNvmlProbe(include_process_memory=False),
                thread_affinity=system_lease.affinity,
            )
            gpu_ready = sampler.start(wait_for_gpu_timeout_s=args.gpu_warmup_timeout)
            load_started_ns = time.monotonic_ns()
            load_started = time.perf_counter()
            with LlamaInitLogCapture() as log_capture:
                llm = Llama(
                    model_path=str(model_path),
                    n_gpu_layers=n_gpu_layers,
                    n_ctx=n_ctx,
                    n_batch=min(args.context_tokens, 512),
                    n_ubatch=min(args.context_tokens, 512),
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
            load_finished_ns = time.monotonic_ns()
            buffer_report = log_capture.buffer_report()
            metadata = dict(llm.metadata)
            architecture = str(metadata.get("general.architecture", ""))
            actual_layers = int(metadata[f"{architecture}.block_count"])
            chat_template = metadata.get("tokenizer.chat_template")
            memory_after_llm, rss_after_llm = snapshot()
            capacity_snapshots.append(memory_after_llm)

            if uses_vision:
                phase = run_vision_phase(
                    llm=llm,
                    config=VisionPhaseConfig(
                        mmproj_path=mmproj_path,
                        image_path=image_path,
                        question=args.question,
                        prompt_mode=args.prompt_mode,
                        eval_mode=args.vision_eval_mode,
                        generation_tokens=args.generation_tokens,
                        use_gpu=state.endswith("_GPU"),
                        n_threads=len(llm_lease.affinity),
                        image_max_tokens=args.image_max_tokens,
                        image_min_tokens=args.image_min_tokens,
                    ),
                    chat_template=chat_template,
                    snapshot=snapshot,
                )
            else:
                chat_prompt = (
                    build_prompt_evidence(
                        mode=args.prompt_mode,
                        question=args.question,
                        media_marker=None,
                        chat_template=chat_template,
                        eos_token=llm.detokenize([llm.token_eos()]).decode(
                            "utf-8", errors="replace"
                        ),
                        bos_token=llm.detokenize([llm.token_bos()]).decode(
                            "utf-8", errors="replace"
                        ),
                    )
                    if state == RESIDENCY_TEXT_ONLY_CHAT
                    else None
                )
                phase = run_text_phase(
                    llm=llm,
                    residency_state=state,
                    context_tokens=args.context_tokens,
                    generation_tokens=args.generation_tokens,
                    prompt_evidence=chat_prompt,
                )
            memory_after_inference, rss_after_inference = snapshot()
            capacity_snapshots.append(memory_after_inference)

            unload_started_ns = time.monotonic_ns()
            unload_started = time.perf_counter()
            llm.close()
            llm = None
            gc.collect()
            unload_s = time.perf_counter() - unload_started
            unload_finished_ns = time.monotonic_ns()
            memory_after_unload, rss_after_unload = snapshot()
            resource_summary = sampler.stop()
            timeline = [asdict(sample) for sample in sampler.samples]
        llm_lease.release()
        system_lease.release()
        allocator_after_release = allocator.snapshot()
    finally:
        if llm is not None:
            llm.close()
        if (
            sampler is not None
            and sampler._thread is not None
            and sampler._thread.is_alive()
        ):
            sampler.stop()
        llm_lease.release()
        system_lease.release()

    phase_intervals_ns = {
        "load": [load_started_ns, load_finished_ns],
        **phase["phase_intervals_ns"],
        "unload": [unload_started_ns, unload_finished_ns],
    }
    resource_by_phase = {
        name: _phase_resource_metrics(sampler.samples, start_ns, end_ns)
        for name, (start_ns, end_ns) in phase_intervals_ns.items()
    }
    inference_interval = phase_intervals_ns["inference"]
    runtime_evidence = assess_colocated_runtime(
        sampler.samples,
        start_ns=inference_interval[0],
        end_ns=inference_interval[1],
        capacity_snapshots=capacity_snapshots,
        max_gpu_sample_age_ms=args.max_gpu_sample_age_ms,
        max_swap_growth_bytes=int(args.max_swap_growth_mb * 1024 * 1024),
    )
    expected_state_tokens = phase["prompt_position_total"] + args.generation_tokens
    acceptance_checks = build_acceptance_checks(
        uses_vision=uses_vision,
        phase=phase,
        buffer_report=buffer_report,
        runtime_evidence=runtime_evidence,
        resource_summary=resource_summary,
        capacity_snapshots=capacity_snapshots,
        allocator_after_release=allocator_after_release,
        physical_core_count=len(topology.physical_cores),
        actual_layers=actual_layers,
        expected_layers=args.model_layers,
        generation_tokens=args.generation_tokens,
        expected_state_tokens=expected_state_tokens,
        gpu_ready=gpu_ready,
        actual_affinity=tuple(actual_affinity),
        llm_affinity=tuple(llm_lease.affinity),
        system_affinity=tuple(system_lease.affinity),
        expected_visual_tokens=args.expected_visual_tokens,
    )
    return build_record(
        args=args,
        phase=phase,
        measurement={
            "acceptance_checks": acceptance_checks,
            "buffer_report": buffer_report,
            "n_gpu_layers": n_gpu_layers,
            "offload_kqv": offload_kqv,
            "actual_layers": actual_layers,
            "architecture": architecture,
            "n_ctx": n_ctx,
            "llm_logical_affinity": tuple(actual_affinity),
            "llm_threads": len(llm_lease.affinity),
            "system_logical_affinity": tuple(system_lease.affinity),
            "llm_load_s": llm_load_s,
            "unload_s": unload_s,
            "expected_state_tokens": expected_state_tokens,
            "memory": {
                "baseline": baseline_snapshot,
                "baseline_rss": baseline_rss,
                "after_llm": memory_after_llm,
                "after_llm_rss": rss_after_llm,
                "after_inference": memory_after_inference,
                "after_inference_rss": rss_after_inference,
                "after_unload": memory_after_unload,
                "after_unload_rss": rss_after_unload,
            },
            "runtime_evidence": runtime_evidence,
            "resource_summary": resource_summary,
            "resource_by_phase": resource_by_phase,
            "phase_intervals_ns": phase_intervals_ns,
            "allocator_after_release": allocator_after_release,
            "child_started_monotonic_ns": child_started_ns,
            "child_started_wall_iso": child_started_wall,
            "child_finished_monotonic_ns": time.monotonic_ns(),
            "child_finished_wall_iso": dt.datetime.now(dt.timezone.utc).isoformat(),
            "timeline": timeline,
        },
    )
