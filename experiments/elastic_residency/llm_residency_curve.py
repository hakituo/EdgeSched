#!/usr/bin/env python3
"""C1/C2：GGUF LLM n_gpu_layers 驻留曲线与 KV 状态 profiler。"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import psutil


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.elastic_residency.contracts import (  # noqa: E402
    BackendEvidence,
    ExperimentArtifactWriter,
    ExperimentEvent,
    ExperimentRunConfig,
    FailureRecord,
    TimelineSample,
)
from experiments.elastic_residency.cpu_topology import (  # noqa: E402
    CPUTopology,
    PhysicalCoreAllocator,
    process_affinity_scope,
)
from experiments.elastic_residency.resource_sampler import (  # noqa: E402
    ResourceSampler,
)
from experiments.elastic_residency.tts_component_offload import (  # noqa: E402
    _git_source,
    _nvidia_memory_snapshot,
)
from experiments.tts.profiling_utils import (  # noqa: E402
    percentile,
    sha256_file,
)


CONFIG_LAYERS = {"K0": 0, "FULL": -1}
PROMPT_SEED_TEXT = (
    "这是用于本地语言模型驻留曲线的固定上下文。"
    "实验会记录预填充速度、逐词元间隔、内存占用和缓存保存恢复时间。"
)


def resolve_config(name: str, model_layers: int) -> int:
    normalized = name.upper()
    if normalized in CONFIG_LAYERS:
        return CONFIG_LAYERS[normalized]
    if normalized.startswith("K") and normalized[1:].isdigit():
        layers = int(normalized[1:])
        if 0 <= layers <= model_layers:
            return layers
    raise ValueError(f"未知或越界的 LLM 驻留配置: {name}")


def _build_exact_prompt_tokens(llm: Any, target_tokens: int) -> list[int]:
    if target_tokens < 2:
        raise ValueError("prompt token 数至少为 2")
    repeated = PROMPT_SEED_TEXT
    tokens = list(llm.tokenize(repeated.encode("utf-8"), add_bos=True))
    while len(tokens) < target_tokens:
        repeated += PROMPT_SEED_TEXT
        tokens = list(llm.tokenize(repeated.encode("utf-8"), add_bos=True))
    result = tokens[:target_tokens]
    if len(result) != target_tokens:
        raise RuntimeError("无法构造精确 token 数的 prompt")
    return result


def _token_ids_sha256(tokens: list[int]) -> str:
    encoded = ",".join(str(token) for token in tokens).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _native_memory_snapshot(
    *,
    process: psutil.Process,
    baseline_card_used_mb: float,
) -> dict[str, Any]:
    nvidia = _nvidia_memory_snapshot(os.getpid())
    card_used_mb = nvidia["gpu_card_used_mb"]
    return {
        "rss_bytes": process.memory_info().rss,
        "system_available_bytes": psutil.virtual_memory().available,
        "system_swap_used_bytes": psutil.swap_memory().used,
        **nvidia,
        "gpu_process_total_estimate_mb": (
            max(0.0, card_used_mb - baseline_card_used_mb)
            if card_used_mb is not None
            else None
        ),
        "gpu_process_total_estimate_method": "card_delta_from_preload_baseline",
    }


def _discover_model_metadata(model_path: Path) -> dict[str, Any]:
    from llama_cpp import Llama

    model = Llama(
        model_path=str(model_path),
        vocab_only=True,
        n_ctx=512,
        verbose=False,
    )
    try:
        metadata = dict(model.metadata)
    finally:
        model.close()
    architecture = str(metadata.get("general.architecture", ""))
    block_key = f"{architecture}.block_count"
    context_key = f"{architecture}.context_length"
    block_count = int(metadata[block_key])
    return {
        "architecture": architecture,
        "block_count": block_count,
        "context_length": int(metadata.get(context_key, 0)),
        "name": metadata.get("general.name"),
        "block_count_key": block_key,
    }


def _decode_fixed_tokens(
    *,
    llm: Any,
    prompt_tokens: list[int],
    generation_tokens: int,
) -> dict[str, Any]:
    if generation_tokens < 2:
        raise ValueError("generation_tokens 至少为 2")
    llm.reset()
    prefill_started_ns = time.monotonic_ns()
    prefill_started = time.perf_counter()
    llm.eval(prompt_tokens)
    prefill_finished = time.perf_counter()
    prefill_finished_ns = time.monotonic_ns()
    prefill_s = prefill_finished - prefill_started

    generated: list[int] = []
    token_times: list[float] = []
    generation_started_ns = time.monotonic_ns()
    first_sample_started = time.perf_counter()
    for _ in range(generation_tokens):
        token = int(llm.sample(top_k=1, top_p=1.0, min_p=0.0, temp=0.0))
        token_times.append(time.perf_counter())
        generated.append(token)
        llm.eval([token])
    generation_finished = time.perf_counter()
    generation_finished_ns = time.monotonic_ns()
    first_sample_s = token_times[0] - first_sample_started
    token_gaps_ms = [
        (right - left) * 1000.0
        for left, right in zip(token_times, token_times[1:])
    ]
    decode_visible_s = token_times[-1] - token_times[0]
    decode_with_final_eval_s = generation_finished - token_times[0]
    return {
        "generated_tokens": generated,
        "generated_token_count": len(generated),
        "prefill_s": prefill_s,
        "prefill_tokens_per_s": len(prompt_tokens) / prefill_s,
        "first_sample_s": first_sample_s,
        "ttft_s": prefill_s + first_sample_s,
        "decode_visible_s": decode_visible_s,
        "decode_with_final_eval_s": decode_with_final_eval_s,
        "decode_tokens_per_s": (
            (len(generated) - 1) / decode_visible_s
            if decode_visible_s > 0
            else 0.0
        ),
        "token_gap_ms": {
            "count": len(token_gaps_ms),
            "mean": statistics.fmean(token_gaps_ms),
            "p50": percentile(token_gaps_ms, 0.50),
            "p95": percentile(token_gaps_ms, 0.95),
            "max": max(token_gaps_ms),
            "values": token_gaps_ms,
        },
        "phase_intervals_ns": {
            "prefill": [prefill_started_ns, prefill_finished_ns],
            "decode": [generation_started_ns, generation_finished_ns],
            "inference": [prefill_started_ns, generation_finished_ns],
        },
    }


def _phase_resource_metrics(
    samples: list[Any],
    start_ns: int,
    end_ns: int,
) -> dict[str, float | int]:
    selected = [
        sample
        for sample in samples
        if start_ns <= sample.monotonic_ns <= end_ns
    ]
    jitters = [sample.scheduler_jitter_ms for sample in selected]
    gpu_ages = [sample.gpu_sample_age_ms for sample in selected]
    return {
        "sample_count": len(selected),
        "jitter_mean_ms": statistics.fmean(jitters) if jitters else 0.0,
        "jitter_p95_ms": percentile(jitters, 0.95),
        "jitter_max_ms": max(jitters, default=0.0),
        "gpu_sample_age_p95_ms": percentile(gpu_ages, 0.95),
        "gpu_sample_age_max_ms": max(gpu_ages, default=0.0),
    }


def run_child(args: argparse.Namespace) -> dict[str, Any]:
    from llama_cpp import Llama

    model_path = Path(args.model).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"找不到 GGUF: {model_path}")
    topology = CPUTopology.discover()
    required_cores = args.system_cores + args.llm_cores
    if len(topology.physical_cores) < required_cores:
        raise RuntimeError(
            f"C1 需要 {required_cores} 个物理核，实际只有 {len(topology.physical_cores)}"
        )
    n_gpu_layers = resolve_config(args.child_config, args.model_layers)
    offload_kqv = n_gpu_layers != 0
    allocator = PhysicalCoreAllocator(topology)
    system_lease = allocator.reserve_system(args.system_cores)
    llm_lease = allocator.acquire(
        "llm_residency_profiler",
        args.llm_cores,
        use_smt=True,
    )
    process = psutil.Process()
    baseline = _nvidia_memory_snapshot(os.getpid())
    baseline_card_used_mb = baseline["gpu_card_used_mb"]
    if baseline_card_used_mb is None:
        raise RuntimeError("C1 无法读取整卡显存基线")

    sampler: ResourceSampler | None = None
    llm: Any = None
    child_started_ns = time.monotonic_ns()
    child_started_wall_iso = dt.datetime.now(dt.timezone.utc).isoformat()
    try:
        with process_affinity_scope(llm_lease.affinity) as actual_affinity:
            sampler = ResourceSampler(
                interval_s=args.sample_interval,
                gpu_interval_s=args.gpu_sample_interval,
                thread_affinity=system_lease.affinity,
            )
            gpu_ready = sampler.start(
                wait_for_gpu_timeout_s=args.gpu_warmup_timeout
            )
            load_started_ns = time.monotonic_ns()
            load_started = time.perf_counter()
            llm = Llama(
                model_path=str(model_path),
                n_gpu_layers=n_gpu_layers,
                n_ctx=args.context_tokens + args.generation_tokens,
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
            load_s = time.perf_counter() - load_started
            load_finished_ns = time.monotonic_ns()
            memory_loaded = _native_memory_snapshot(
                process=process,
                baseline_card_used_mb=baseline_card_used_mb,
            )
            metadata = dict(llm.metadata)
            architecture = str(metadata.get("general.architecture", ""))
            actual_layers = int(metadata[f"{architecture}.block_count"])
            prompt_tokens = _build_exact_prompt_tokens(
                llm, args.context_tokens
            )
            inference = _decode_fixed_tokens(
                llm=llm,
                prompt_tokens=prompt_tokens,
                generation_tokens=args.generation_tokens,
            )
            memory_after_inference = _native_memory_snapshot(
                process=process,
                baseline_card_used_mb=baseline_card_used_mb,
            )
            kv_save_started_ns = time.monotonic_ns()
            save_started = time.perf_counter()
            state = llm.save_state()
            kv_save_s = time.perf_counter() - save_started
            kv_save_finished_ns = time.monotonic_ns()
            state_bytes = int(getattr(state, "llama_state_size", 0))
            expected_state_tokens = args.context_tokens + args.generation_tokens
            llm.reset()
            kv_restore_started_ns = time.monotonic_ns()
            restore_started = time.perf_counter()
            llm.load_state(state)
            kv_restore_s = time.perf_counter() - restore_started
            kv_restore_finished_ns = time.monotonic_ns()
            restored_tokens = int(llm.n_tokens)
            output_text = llm.detokenize(
                inference["generated_tokens"]
            ).decode("utf-8", errors="replace")
            generated_tokens = inference.pop("generated_tokens")
            inference["generated_tokens_sha256"] = _token_ids_sha256(
                generated_tokens
            )
            inference["output_text_sha256"] = hashlib.sha256(
                output_text.encode("utf-8")
            ).hexdigest()
            inference["output_text_preview"] = output_text[:200]
            inference["eos_token_count"] = sum(
                token == llm.token_eos() for token in generated_tokens
            )
            memory_after_restore = _native_memory_snapshot(
                process=process,
                baseline_card_used_mb=baseline_card_used_mb,
            )
            unload_started_ns = time.monotonic_ns()
            unload_started = time.perf_counter()
            llm.close()
            llm = None
            gc.collect()
            unload_s = time.perf_counter() - unload_started
            unload_finished_ns = time.monotonic_ns()
            memory_after_unload = _native_memory_snapshot(
                process=process,
                baseline_card_used_mb=baseline_card_used_mb,
            )
            resource_summary = sampler.stop()
            phase_intervals_ns = {
                "load": [load_started_ns, load_finished_ns],
                **inference.pop("phase_intervals_ns"),
                "kv_save": [kv_save_started_ns, kv_save_finished_ns],
                "kv_restore": [
                    kv_restore_started_ns,
                    kv_restore_finished_ns,
                ],
                "unload": [unload_started_ns, unload_finished_ns],
            }
            resource_by_phase = {
                name: _phase_resource_metrics(sampler.samples, start_ns, end_ns)
                for name, (start_ns, end_ns) in phase_intervals_ns.items()
            }
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

    acceptance_checks = {
        "model_layer_count_matches": actual_layers == args.model_layers,
        "gpu_ready_before_timeline": gpu_ready,
        "gpu_present_for_every_timeline_sample": (
            resource_summary["gpu_unavailable_timeline_sample_count"] == 0
        ),
        "inference_gpu_sample_age_within_limit": (
            resource_by_phase["inference"]["gpu_sample_age_max_ms"]
            <= args.max_gpu_sample_age_ms
        ),
        "inference_resource_jitter_p95_within_limit": (
            resource_by_phase["inference"]["jitter_p95_ms"]
            <= args.max_resource_jitter_p95_ms
        ),
        "inference_resource_jitter_max_within_limit": (
            resource_by_phase["inference"]["jitter_max_ms"]
            <= args.max_resource_jitter_max_ms
        ),
        "prompt_token_count_exact": len(prompt_tokens) == args.context_tokens,
        "generation_token_count_exact": (
            inference["generated_token_count"] == args.generation_tokens
        ),
        "prefill_and_decode_positive": (
            inference["prefill_tokens_per_s"] > 0
            and inference["decode_tokens_per_s"] > 0
        ),
        "token_gap_count_exact": (
            inference["token_gap_ms"]["count"] == args.generation_tokens - 1
        ),
        "kv_state_saved": state_bytes > 0 and kv_save_s > 0,
        "kv_state_restored": (
            kv_restore_s > 0 and restored_tokens == expected_state_tokens
        ),
        "worker_affinity_matches": set(actual_affinity) == set(llm_lease.affinity),
        "sampler_affinity_matches": (
            set(resource_summary["sampler_thread_affinity"])
            == set(system_lease.affinity)
        ),
        "gpu_sampler_affinity_matches": (
            set(resource_summary["gpu_thread_affinity"])
            == set(system_lease.affinity)
        ),
        "swap_unused": all(
            sample["system_swap_used_bytes"] == 0 for sample in timeline
        ),
        "all_leases_released": (
            allocator_after_release["active_lease_count"] == 0
        ),
        "all_physical_cores_returned": (
            allocator_after_release["free_physical_core_count"]
            == len(topology.physical_cores)
        ),
        "gpu_memory_evidence_available": (
            memory_after_inference["gpu_process_total_mb"] is not None
            or memory_after_inference["gpu_process_total_estimate_mb"] is not None
        ),
    }
    return {
        "config": args.child_config,
        "run_index": args.run_index,
        "passed": all(acceptance_checks.values()),
        "acceptance_checks": acceptance_checks,
        "backend_kind": "real_llama_cpp_gguf",
        "fallback_occurred": False,
        "n_gpu_layers": n_gpu_layers,
        "offload_kqv": offload_kqv,
        "op_offload": offload_kqv,
        "flash_attn": True,
        "model_layers": actual_layers,
        "context_tokens": args.context_tokens,
        "generation_tokens": args.generation_tokens,
        "n_ctx": args.context_tokens + args.generation_tokens,
        "llm_physical_cores": args.llm_cores,
        "llm_logical_affinity": list(actual_affinity),
        "llm_threads": len(llm_lease.affinity),
        "system_physical_cores": args.system_cores,
        "system_logical_affinity": list(system_lease.affinity),
        "prompt_tokens_sha256": _token_ids_sha256(prompt_tokens),
        "load_s": load_s,
        "inference": inference,
        "kv_save_s": kv_save_s,
        "kv_restore_s": kv_restore_s,
        "kv_state_bytes": state_bytes,
        "restored_tokens": restored_tokens,
        "expected_state_tokens": expected_state_tokens,
        "memory_baseline": baseline,
        "memory_loaded": memory_loaded,
        "memory_after_inference": memory_after_inference,
        "memory_after_restore": memory_after_restore,
        "memory_after_unload": memory_after_unload,
        "unload_s": unload_s,
        "resource_summary": resource_summary,
        "resource_by_phase": resource_by_phase,
        "phase_intervals_ns": phase_intervals_ns,
        "quality_warnings": {
            "non_inference_jitter_exceeds_inference_limit": (
                resource_summary["jitter_max_ms"]
                > args.max_resource_jitter_max_ms
            ),
            "gpu_process_memory_is_estimated": (
                memory_after_inference["gpu_process_total_mb"] is None
            ),
        },
        "allocator_after_release": allocator_after_release,
        "child_started_monotonic_ns": child_started_ns,
        "child_started_wall_iso": child_started_wall_iso,
        "child_finished_monotonic_ns": time.monotonic_ns(),
        "child_finished_wall_iso": dt.datetime.now(dt.timezone.utc).isoformat(),
        "timeline": timeline,
    }


def _summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = {
        "load_s": [record["load_s"] for record in records],
        "ttft_s": [record["inference"]["ttft_s"] for record in records],
        "prefill_tokens_per_s": [
            record["inference"]["prefill_tokens_per_s"] for record in records
        ],
        "decode_tokens_per_s": [
            record["inference"]["decode_tokens_per_s"] for record in records
        ],
        "token_gap_p95_ms": [
            record["inference"]["token_gap_ms"]["p95"] for record in records
        ],
        "kv_save_s": [record["kv_save_s"] for record in records],
        "kv_restore_s": [record["kv_restore_s"] for record in records],
        "gpu_delta_mb": [
            record["memory_after_inference"]["gpu_process_total_estimate_mb"]
            or 0.0
            for record in records
        ],
        "rss_mb": [
            record["memory_after_inference"]["rss_bytes"] / 1048576.0
            for record in records
        ],
    }
    return {
        "samples": len(records),
        "passed_count": sum(bool(record["passed"]) for record in records),
        "all_passed": all(record["passed"] for record in records),
        **{
            name: {
                "mean": statistics.fmean(values),
                "p95": percentile(values, 0.95),
                "min": min(values),
                "max": max(values),
            }
            for name, values in metrics.items()
        },
    }


def run_parent(args: argparse.Namespace) -> Path:
    model_path = Path(args.model).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"找不到 GGUF: {model_path}")
    metadata = _discover_model_metadata(model_path)
    model_layers = metadata["block_count"]
    config_names = tuple(name.upper() for name in args.configs)
    for name in config_names:
        resolve_config(name, model_layers)
    writer = ExperimentArtifactWriter(args.output_root, args.run_id)
    source_git, git_complete = _git_source(args, writer.layout.root)
    model_sha256 = sha256_file(model_path)
    expected_records = len(config_names) * args.runs
    config = ExperimentRunConfig(
        experiment_name="llm_residency_curve_c1_c2",
        run_id=args.run_id,
        seed=args.seed,
        sample_interval_s=args.sample_interval,
        gpu_sample_interval_s=args.gpu_sample_interval,
        backends=(
            BackendEvidence(
                component="llm",
                backend_kind="real_llama_cpp_gguf",
                is_real=True,
                model_path=str(model_path),
                model_sha256=model_sha256,
            ),
        ),
        parameters={
            "configs": config_names,
            "runs": args.runs,
            "model_layers": model_layers,
            "context_tokens": args.context_tokens,
            "generation_tokens": args.generation_tokens,
            "llm_physical_cores": args.llm_cores,
            "system_physical_cores": args.system_cores,
            "expected_records": expected_records,
            "fixed_generation_ignores_eos": True,
            "acceptance_thresholds": {
                "max_inference_gpu_sample_age_ms": args.max_gpu_sample_age_ms,
                "max_inference_resource_jitter_p95_ms": (
                    args.max_resource_jitter_p95_ms
                ),
                "max_inference_resource_jitter_max_ms": (
                    args.max_resource_jitter_max_ms
                ),
            },
        },
    )
    writer.initialize(
        config,
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "git": {"source_copy": source_git},
            "git_provenance_complete": git_complete,
            "model": {
                "path": str(model_path),
                "size_bytes": model_path.stat().st_size,
                "sha256": model_sha256,
                "metadata": metadata,
            },
        },
    )

    records: list[dict[str, Any]] = []
    for config_name in config_names:
        for run_index in range(1, args.runs + 1):
            child_json = (
                writer.layout.root / f"{config_name}_r{run_index}_child.json"
            )
            child_log = writer.layout.root / f"{config_name}_r{run_index}.log"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--child-config",
                config_name,
                "--run-index",
                str(run_index),
                "--model-layers",
                str(model_layers),
                "--child-output-json",
                str(child_json),
                "--model",
                str(model_path),
                "--context-tokens",
                str(args.context_tokens),
                "--generation-tokens",
                str(args.generation_tokens),
                "--llm-cores",
                str(args.llm_cores),
                "--system-cores",
                str(args.system_cores),
                "--sample-interval",
                str(args.sample_interval),
                "--gpu-sample-interval",
                str(args.gpu_sample_interval),
                "--gpu-warmup-timeout",
                str(args.gpu_warmup_timeout),
                "--max-gpu-sample-age-ms",
                str(args.max_gpu_sample_age_ms),
                "--max-resource-jitter-p95-ms",
                str(args.max_resource_jitter_p95_ms),
                "--max-resource-jitter-max-ms",
                str(args.max_resource_jitter_max_ms),
                "--seed",
                str(args.seed + run_index - 1),
            ]
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=args.child_timeout,
            )
            child_log.write_text(
                completed.stdout + "\n[stderr]\n" + completed.stderr,
                encoding="utf-8",
            )
            if completed.returncode != 0 or not child_json.is_file():
                writer.append_failure(
                    FailureRecord(
                        stage=f"{config_name}_r{run_index}",
                        error_type="ChildProcessFailure",
                        message=f"returncode={completed.returncode}",
                        monotonic_ns=time.monotonic_ns(),
                        attributes={"log": str(child_log)},
                    )
                )
                continue
            record = json.loads(child_json.read_text(encoding="utf-8"))
            timeline = record.pop("timeline", [])
            for sample in timeline:
                writer.append_timeline(TimelineSample(**sample))
            writer.append_event(
                ExperimentEvent(
                    name="llm_residency_child",
                    monotonic_ns=record["child_started_monotonic_ns"],
                    wall_time_iso=record["child_started_wall_iso"],
                    phase=config_name,
                    attributes={
                        "run_index": run_index,
                        "finished_monotonic_ns": record[
                            "child_finished_monotonic_ns"
                        ],
                    },
                )
            )
            writer.append_iteration(record)
            records.append(record)

    by_config = {
        name: _summarize_records(
            [record for record in records if record["config"] == name]
        )
        for name in config_names
        if any(record["config"] == name for record in records)
    }
    passed = (
        git_complete
        and len(records) == expected_records
        and all(record["passed"] for record in records)
    )
    summary = {
        "passed": passed,
        "expected_records": expected_records,
        "actual_records": len(records),
        "git_provenance_complete": git_complete,
        "model_metadata": metadata,
        "by_config": by_config,
        "records": records,
        "evidence_scope": "c1_smoke" if args.runs == 1 else "c2_curve",
    }
    writer.write_summary(summary)
    fields = [
        "config",
        "run_index",
        "passed",
        "n_gpu_layers",
        "load_s",
        "gpu_delta_mb",
        "rss_mb",
        "ttft_s",
        "prefill_tokens_per_s",
        "decode_tokens_per_s",
        "token_gap_p50_ms",
        "token_gap_p95_ms",
        "token_gap_max_ms",
        "kv_save_s",
        "kv_restore_s",
        "kv_state_bytes",
    ]
    with writer.layout.summary_csv.open(
        "w", encoding="utf-8", newline=""
    ) as file_obj:
        csv_writer = csv.DictWriter(file_obj, fieldnames=fields)
        csv_writer.writeheader()
        for record in records:
            gap = record["inference"]["token_gap_ms"]
            csv_writer.writerow(
                {
                    "config": record["config"],
                    "run_index": record["run_index"],
                    "passed": record["passed"],
                    "n_gpu_layers": record["n_gpu_layers"],
                    "load_s": record["load_s"],
                    "gpu_delta_mb": record["memory_after_inference"][
                        "gpu_process_total_estimate_mb"
                    ],
                    "rss_mb": record["memory_after_inference"]["rss_bytes"]
                    / 1048576.0,
                    "ttft_s": record["inference"]["ttft_s"],
                    "prefill_tokens_per_s": record["inference"][
                        "prefill_tokens_per_s"
                    ],
                    "decode_tokens_per_s": record["inference"][
                        "decode_tokens_per_s"
                    ],
                    "token_gap_p50_ms": gap["p50"],
                    "token_gap_p95_ms": gap["p95"],
                    "token_gap_max_ms": gap["max"],
                    "kv_save_s": record["kv_save_s"],
                    "kv_restore_s": record["kv_restore_s"],
                    "kv_state_bytes": record["kv_state_bytes"],
                }
            )
    return writer.layout.root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="/home/leslie/mvp_core/models/llm/Qwen3.5-9B-Q4_K_M.gguf",
    )
    parser.add_argument("--configs", nargs="+", default=["K0", "FULL"])
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--context-tokens", type=int, default=512)
    parser.add_argument("--generation-tokens", type=int, default=128)
    parser.add_argument("--llm-cores", type=int, default=14)
    parser.add_argument("--system-cores", type=int, default=2)
    parser.add_argument("--sample-interval", type=float, default=0.05)
    parser.add_argument("--gpu-sample-interval", type=float, default=0.25)
    parser.add_argument("--gpu-warmup-timeout", type=float, default=5.0)
    parser.add_argument("--max-gpu-sample-age-ms", type=float, default=750.0)
    parser.add_argument(
        "--max-resource-jitter-p95-ms", type=float, default=5.0
    )
    parser.add_argument(
        "--max-resource-jitter-max-ms", type=float, default=30.0
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--child-timeout", type=float, default=1800.0)
    parser.add_argument(
        "--output-root",
        default=str(
            PROJECT_ROOT / "experiment_results" / "elastic_residency" / "c1"
        ),
    )
    parser.add_argument(
        "--run-id",
        default=f"c1_llm_residency_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument("--source-git-commit", default="")
    parser.add_argument("--source-git-dirty", choices=("true", "false"))
    parser.add_argument("--source-git-status-entry-count", type=int)
    parser.add_argument("--source-git-snapshot")
    parser.add_argument("--source-git-snapshot-sha256")
    parser.add_argument("--child-config")
    parser.add_argument("--run-index", type=int, default=1)
    parser.add_argument("--model-layers", type=int, default=32)
    parser.add_argument("--child-output-json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.runs < 1:
        raise ValueError("--runs 必须大于 0")
    if args.child_config:
        try:
            result = run_child(args)
        except Exception as exc:
            if args.child_output_json:
                Path(args.child_output_json).write_text(
                    json.dumps(
                        {
                            "config": args.child_config,
                            "run_index": args.run_index,
                            "passed": False,
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            raise
        Path(args.child_output_json).write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return
    print(run_parent(args))


if __name__ == "__main__":
    main()
