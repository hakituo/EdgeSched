#!/usr/bin/env python3
"""LLM+TTS 真实共置显存测量（独立入口，先 smoke 后矩阵）。

测量口径（与 docs/ELASTIC_RESIDENCY_EXPERIMENT_EXECUTION_PLAN_20260828.md 一致）：

1. direct NVML 记录整卡时间线与高水位，不使用 nvidia-smi；
2. LLM 加载并填充 KV/context 后记录共置基线 ``B_LLM``（整卡绝对值）；
3. TTS 每阶段 ``torch.cuda.synchronize()`` 后重置并记录 allocated/reserved
   与 peak allocated/peak reserved（进程内 torch 统计）；
4. ``P_colocated`` = TTS 执行期间整卡实测峰值（时间线样本 + 阶段快照）；
5. ``tts_colocated_increment_mb = P_colocated - B_LLM``；
6. ``colocated_positive_residual_mb = max(0, colocated_increment - isolated_increment)``
   （isolated 增量来自同配置 B1/B2 离线结果，运行时注入，缺失时保持 None）；
7. 容量裁决使用实测整卡增量：
   ``(P_colocated - B_idle) + G_safe <= gpu_total``；另保留物理峰值越界检查。
   M_L(k, context) 已包含该 LLM 配置实际的权重、KV buffer、compute buffer
   与 workspace，本入口不得再额外加一次 KV，避免重复计数。

本入口不做性能计时结论，只做显存容量与残差测量。
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import json
import os
import platform
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
    SCHEMA_VERSION,
    BackendEvidence,
    ExperimentArtifactWriter,
    ExperimentEvent,
    ExperimentRunConfig,
    FailureRecord,
    TimelineSample,
)
from experiments.elastic_residency.colocated_runtime_evidence import (  # noqa: E402
    assess_colocated_runtime,
)
from experiments.elastic_residency.colocated_capacity import (  # noqa: E402
    colocated_metrics,
)
from experiments.elastic_residency.colocated_audio_artifact import (  # noqa: E402
    output_wav_artifact as _output_wav_artifact,
)
from experiments.elastic_residency.colocated_plan import (  # noqa: E402
    ColocatedCandidate,
    build_execution_plan,
    parse_candidate_specs,
)
from experiments.elastic_residency.cpu_topology import (  # noqa: E402
    CPUTopology,
    PhysicalCoreAllocator,
    process_affinity_scope,
)
from experiments.elastic_residency.gpu_telemetry import (  # noqa: E402
    DirectNvmlProbe,
)
from experiments.elastic_residency.kv_measurement import (  # noqa: E402
    KV_MEASUREMENT_SEMANTICS,
    LlamaInitLogCapture,
    buffer_report_acceptance,
    capture_llama_kv_state,
)
from experiments.elastic_residency.llm_residency_curve import (  # noqa: E402
    _build_exact_prompt_tokens,
    _discover_model_metadata,
    _token_ids_sha256,
    resolve_config,
)
from experiments.elastic_residency.resource_sampler import ResourceSampler  # noqa: E402
from experiments.elastic_residency.tts_component_offload import (  # noqa: E402
    _actual_devices,
    _apply_placement,
    _build_xvector_prompt,
    _component_inventory,
    _git_source,
    _memory_snapshot,
    _nvidia_memory_snapshot,
    _placement_matches,
    _synchronize,
    placement_for,
)
from experiments.tts.profiling_utils import (  # noqa: E402
    model_tree_hash,
    sha256_file,
)


def parse_isolated_increments(items: list[str] | None) -> dict[str, float]:
    """解析 ``CONFIG=value`` 形式的 TTS 隔离增量参数。"""

    result: dict[str, float] = {}
    for item in items or []:
        name, separator, value = item.partition("=")
        if not separator or not name.strip():
            raise ValueError(f"非法的隔离增量条目: {item!r}，应为 CONFIG=value")
        result[name.strip().upper()] = float(value)
    return result


def _stage_snapshot(
    process: psutil.Process,
    *,
    baseline_card_used_mb: float,
) -> dict[str, Any]:
    """阶段末快照：torch 统计 + 整卡绝对值（调用前必须已 synchronize）。"""

    return _memory_snapshot(process, baseline_card_used_mb=baseline_card_used_mb)


def _tts_window_peak_mb(
    samples: list[Any],
    start_ns: int,
    end_ns: int,
    stage_snapshots: list[dict[str, Any]],
) -> float | None:
    """TTS 执行窗口内的整卡实测峰值（时间线样本与阶段快照取最大）。"""

    values = [
        sample.gpu_card_used_mb
        for sample in samples
        if sample.gpu_available
        and start_ns <= sample.monotonic_ns <= end_ns
    ]
    values.extend(
        snapshot["gpu_card_used_mb"]
        for snapshot in stage_snapshots
        if snapshot.get("gpu_card_used_mb") is not None
    )
    return max(values) if values else None


def run_child(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import soundfile as sf
    import torch
    from llama_cpp import Llama
    from qwen_tts import Qwen3TTSModel

    spec = placement_for(args.tts_config)
    llm_model_path = Path(args.llm_model).expanduser().resolve()
    tts_model_path = Path(args.tts_model).expanduser().resolve()
    ref_audio = Path(args.ref_audio).expanduser().resolve()
    output_wav = Path(args.child_output_wav).expanduser().resolve()
    if not llm_model_path.is_file():
        raise FileNotFoundError(f"找不到 GGUF: {llm_model_path}")
    if not tts_model_path.is_dir() or not ref_audio.is_file():
        raise FileNotFoundError("共置 child 找不到 TTS 模型或参考音频")
    if not torch.cuda.is_available():
        raise RuntimeError("共置测量要求 CUDA")

    topology = CPUTopology.discover()
    required_cores = args.system_cores + args.llm_cores
    if len(topology.physical_cores) < required_cores:
        raise RuntimeError(
            f"共置测量需要 {required_cores} 个物理核，实际只有 "
            f"{len(topology.physical_cores)}"
        )
    n_gpu_layers = resolve_config(args.llm_config, args.model_layers)
    expected_state_tokens = args.context_tokens + args.generation_tokens
    offload_kqv = n_gpu_layers != 0
    allocator = PhysicalCoreAllocator(topology)
    system_lease = allocator.reserve_system(args.system_cores)
    llm_lease = allocator.acquire("llm_tts_colocated", args.llm_cores, use_smt=True)
    process = psutil.Process()
    process.cpu_percent(interval=None)

    memory_baseline = _nvidia_memory_snapshot(os.getpid())
    baseline_card_used_mb = memory_baseline["gpu_card_used_mb"]
    gpu_total_mb = memory_baseline["gpu_card_total_mb"]
    if baseline_card_used_mb is None or gpu_total_mb is None:
        raise RuntimeError("共置测量无法读取整卡显存基线/总量")

    sampler: ResourceSampler | None = None
    llm: Any = None
    child_started_ns = time.monotonic_ns()
    child_started_wall_iso = dt.datetime.now(dt.timezone.utc).isoformat()
    try:
        with process_affinity_scope(llm_lease.affinity) as actual_affinity:
            sampler = ResourceSampler(
                interval_s=args.sample_interval,
                gpu_interval_s=args.gpu_sample_interval,
                gpu_probe=DirectNvmlProbe(include_process_memory=False),
                thread_affinity=system_lease.affinity,
            )
            gpu_ready = sampler.start(wait_for_gpu_timeout_s=args.gpu_warmup_timeout)
            # ---- LLM 加载与 KV/context 填充（同 KV 校准口径）----
            llm_load_started = time.perf_counter()
            with LlamaInitLogCapture() as log_capture:
                llm = Llama(
                    model_path=str(llm_model_path),
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
            llm_load_s = time.perf_counter() - llm_load_started
            buffer_report = log_capture.buffer_report()
            metadata = dict(llm.metadata)
            architecture = str(metadata.get("general.architecture", ""))
            actual_layers = int(metadata[f"{architecture}.block_count"])
            prompt_tokens = _build_exact_prompt_tokens(llm, expected_state_tokens)
            llm.eval(prompt_tokens)
            kv_native_state = capture_llama_kv_state(llm)
            _synchronize()
            b_llm_snapshot = _nvidia_memory_snapshot(os.getpid())
            b_llm_mb = b_llm_snapshot["gpu_card_used_mb"]
            if b_llm_mb is None:
                raise RuntimeError("共置基线 B_LLM 不可用，禁止继续测量")

            # ---- TTS 加载与放置 ----
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            torch.cuda.reset_peak_memory_stats()
            tts_load_started = time.perf_counter()
            wrapper = Qwen3TTSModel.from_pretrained(
                str(tts_model_path),
                device_map="cuda:0",
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
                local_files_only=True,
            )
            _synchronize()
            tts_load_s = time.perf_counter() - tts_load_started
            placement_started = time.perf_counter()
            _apply_placement(wrapper, spec)
            tts_placement_s = time.perf_counter() - placement_started
            inventory_placed = _component_inventory(wrapper)

            tts_prompt, tts_prompt_build_s = _build_xvector_prompt(wrapper, ref_audio)
            if spec.unload_speaker_after_prompt:
                wrapper.model.speaker_encoder = None
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            inventory_ready = _component_inventory(wrapper)
            actual_devices = _actual_devices(inventory_ready)

            input_text = wrapper._build_assistant_text(args.text)
            input_ids = wrapper._tokenize_texts([input_text])
            tts_prompt_dict = wrapper._prompt_items_to_voice_clone_prompt([tts_prompt])
            generation_kwargs = wrapper._merge_generate_kwargs(
                do_sample=False,
                subtalker_dosample=False,
                max_new_tokens=args.max_new_tokens,
                repetition_penalty=args.repetition_penalty,
            )
            memory_tts_ready = _stage_snapshot(
                process, baseline_card_used_mb=baseline_card_used_mb
            )

            # ---- TTS 执行窗口：每阶段 synchronize + reset peak ----
            tts_window_started_ns = time.monotonic_ns()
            stage_snapshots: list[dict[str, Any]] = []

            _synchronize()
            torch.cuda.reset_peak_memory_stats()
            talker_started = time.perf_counter()
            codes, _ = wrapper.model.generate(
                input_ids=input_ids,
                ref_ids=None,
                voice_clone_prompt=tts_prompt_dict,
                languages=[args.language],
                non_streaming_mode=True,
                **generation_kwargs,
            )
            _synchronize()
            talker_generate_s = time.perf_counter() - talker_started
            memory_talker_peak = _stage_snapshot(
                process, baseline_card_used_mb=baseline_card_used_mb
            )
            stage_snapshots.append(memory_talker_peak)

            codec_device = torch.device(spec.speech_codec_device)
            _synchronize()
            torch.cuda.reset_peak_memory_stats()
            transfer_started = time.perf_counter()
            decode_codes = [code.to(codec_device) for code in codes]
            _synchronize()
            code_transfer_s = time.perf_counter() - transfer_started
            memory_transfer_peak = _stage_snapshot(
                process, baseline_card_used_mb=baseline_card_used_mb
            )
            stage_snapshots.append(memory_transfer_peak)

            torch.cuda.reset_peak_memory_stats()
            codec_started = time.perf_counter()
            wavs, sample_rate = wrapper.model.speech_tokenizer.decode(
                [{"audio_codes": code} for code in decode_codes]
            )
            _synchronize()
            codec_decode_s = time.perf_counter() - codec_started
            memory_codec_peak = _stage_snapshot(
                process, baseline_card_used_mb=baseline_card_used_mb
            )
            stage_snapshots.append(memory_codec_peak)
            tts_window_finished_ns = time.monotonic_ns()

            output_wav.parent.mkdir(parents=True, exist_ok=True)
            sf.write(output_wav, wavs[0], sample_rate, subtype="PCM_16")
            audio_duration_s = len(wavs[0]) / sample_rate
            waveform = (
                wavs[0].detach().float().cpu().numpy()
                if torch.is_tensor(wavs[0])
                else wavs[0]
            )
            waveform_finite = bool(np.isfinite(waveform).all())
            waveform_rms = float(np.sqrt(np.mean(np.square(waveform))))
            waveform_peak = float(np.max(np.abs(waveform)))

            # ---- 卸载与资源归还 ----
            del wrapper
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
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
        if (
            sampler is not None
            and sampler._thread is not None
            and sampler._thread.is_alive()
        ):
            sampler.stop()
        llm_lease.release()
        system_lease.release()

    p_colocated_mb = _tts_window_peak_mb(
        # sampler.samples 仍在（stop() 只停线程不清数据）。
        sampler.samples,
        tts_window_started_ns,
        tts_window_finished_ns,
        stage_snapshots,
    )
    if p_colocated_mb is None:
        raise RuntimeError("TTS 执行窗口内没有任何可用整卡样本，禁止填 0 冒充")
    runtime_evidence = assess_colocated_runtime(
        sampler.samples,
        start_ns=tts_window_started_ns,
        end_ns=tts_window_finished_ns,
        capacity_snapshots=[b_llm_snapshot, *stage_snapshots],
        max_gpu_sample_age_ms=args.max_gpu_sample_age_ms,
        max_swap_growth_bytes=int(args.max_swap_growth_mb * 1024 * 1024),
    )
    isolated_map = parse_isolated_increments(args.tts_isolated_increment_mb)
    metrics = colocated_metrics(
        global_idle_baseline_mb=baseline_card_used_mb,
        b_llm_mb=b_llm_mb,
        p_colocated_mb=p_colocated_mb,
        gpu_total_mb=gpu_total_mb,
        safety_margin_mb=args.safety_margin_mb,
        tts_isolated_increment_mb=isolated_map.get(spec.name),
    )
    output_wav_artifact = _output_wav_artifact(output_wav, audio_duration_s)

    buffer_acceptance = buffer_report_acceptance(buffer_report)
    acceptance_checks = {
        "model_layer_count_matches": actual_layers == args.model_layers,
        "state_token_count_exact": len(prompt_tokens) == expected_state_tokens,
        "kv_position_bounds_valid": (
            0
            <= kv_native_state["kv_position_min"]
            <= kv_native_state["kv_position_max"]
            and kv_native_state["kv_position_max"] == expected_state_tokens - 1
            and kv_native_state["kv_tokens_used"] == expected_state_tokens
        ),
        "kv_sequence_state_bytes_positive": (
            kv_native_state["kv_sequence_state_bytes"] > 0
        ),
        "full_state_serialized_bytes_positive": (
            kv_native_state["full_state_serialized_bytes"] > 0
        ),
        **buffer_acceptance,
        "b_llm_recorded_before_tts_load": b_llm_mb is not None,
        "p_colocated_not_below_b_llm": p_colocated_mb >= b_llm_mb,
        "component_placement_matches": _placement_matches(spec, actual_devices),
        "torch_phase_peaks_available": all(
            snapshot["torch_peak_allocated_bytes"] > 0
            for snapshot in stage_snapshots
        ),
        **output_wav_artifact["checks"],
        "audio_waveform_finite": waveform_finite,
        "audio_waveform_non_silent": waveform_rms > 1e-5,
        "audio_waveform_peak_valid": 0 < waveform_peak <= 1.05,
        "gpu_ready_before_timeline": gpu_ready,
        "gpu_present_for_every_timeline_sample": (
            resource_summary["gpu_unavailable_timeline_sample_count"] == 0
        ),
        **runtime_evidence["checks"],
        "worker_affinity_matches": set(actual_affinity) == set(llm_lease.affinity),
        "sampler_affinity_matches": (
            set(resource_summary["sampler_thread_affinity"])
            == set(system_lease.affinity)
        ),
        "gpu_sampler_affinity_matches": (
            set(resource_summary["gpu_thread_affinity"]) == set(system_lease.affinity)
        ),
        "all_leases_released": allocator_after_release["active_lease_count"] == 0,
        "all_physical_cores_returned": (
            allocator_after_release["free_physical_core_count"]
            == len(topology.physical_cores)
        ),
        "gpu_telemetry_is_direct_nvml": (
            resource_summary["gpu_telemetry_backend"] == "direct_nvml"
            and not resource_summary["gpu_cli_subprocess_used"]
            and b_llm_snapshot["gpu_telemetry_backend"] == "direct_nvml"
            and not b_llm_snapshot["gpu_cli_subprocess_used"]
            and all(
                snapshot["gpu_telemetry_backend"] == "direct_nvml"
                and not snapshot["gpu_cli_subprocess_used"]
                for snapshot in stage_snapshots
            )
        ),
    }
    return {
        "config": f"{args.llm_config}@C{args.context_tokens}+{spec.name}",
        "llm_config": args.llm_config,
        "tts_config": spec.name,
        "run_index": args.run_index,
        "passed": all(acceptance_checks.values()),
        "acceptance_checks": acceptance_checks,
        "backend_kind": "real_llama_cpp_gguf_plus_qwen3_tts",
        "fallback_occurred": False,
        "n_gpu_layers": n_gpu_layers,
        "offload_kqv": offload_kqv,
        "op_offload": offload_kqv,
        "flash_attn": True,
        "model_layers": actual_layers,
        "context_tokens": args.context_tokens,
        "generation_tokens": args.generation_tokens,
        "expected_state_tokens": expected_state_tokens,
        "n_ctx": expected_state_tokens,
        "llm_logical_affinity": list(actual_affinity),
        "system_logical_affinity": list(system_lease.affinity),
        "prompt_tokens_sha256": _token_ids_sha256(prompt_tokens),
        "kv_state": kv_native_state,
        "full_state_serialized_bytes": kv_native_state[
            "full_state_serialized_bytes"
        ],
        "kv_sequence_state_bytes": kv_native_state["kv_sequence_state_bytes"],
        "kv_gpu_buffer_mb": buffer_report.get("kv_gpu_buffer_mb"),
        "kv_cpu_buffer_mb": buffer_report.get("kv_cpu_buffer_mb"),
        "compute_gpu_buffer_mb": buffer_report.get("compute_gpu_buffer_mb"),
        "compute_cpu_buffer_mb": buffer_report.get("compute_cpu_buffer_mb"),
        "kv_measurement_semantics": KV_MEASUREMENT_SEMANTICS,
        "colocated_metrics": metrics,
        "placement": asdict(spec),
        "actual_devices": actual_devices,
        "inventory_placed": inventory_placed,
        "inventory_ready": inventory_ready,
        "llm_load_s": llm_load_s,
        "tts_load_s": tts_load_s,
        "tts_placement_s": tts_placement_s,
        "tts_prompt_build_s": tts_prompt_build_s,
        "talker_generate_s": talker_generate_s,
        "code_transfer_s": code_transfer_s,
        "codec_decode_s": codec_decode_s,
        "audio_duration_s": audio_duration_s,
        "waveform_rms": waveform_rms,
        "waveform_peak": waveform_peak,
        "sample_rate": sample_rate,
        "output_wav": output_wav_artifact["path"],
        "output_wav_size_bytes": output_wav_artifact["size_bytes"],
        "output_wav_sha256": output_wav_artifact["sha256"],
        "memory_baseline": memory_baseline,
        "b_llm_snapshot": b_llm_snapshot,
        "memory_tts_ready": memory_tts_ready,
        "memory_talker_peak": memory_talker_peak,
        "memory_transfer_peak": memory_transfer_peak,
        "memory_codec_peak": memory_codec_peak,
        "resource_summary": resource_summary,
        "runtime_evidence": runtime_evidence,
        "phase_intervals_ns": {
            "tts_execution": [tts_window_started_ns, tts_window_finished_ns],
        },
        "quality_warnings": {
            "tts_isolated_increment_missing": (
                isolated_map.get(spec.name) is None
            ),
        },
        "allocator_after_release": allocator_after_release,
        "child_started_monotonic_ns": child_started_ns,
        "child_started_wall_iso": child_started_wall_iso,
        "child_finished_monotonic_ns": time.monotonic_ns(),
        "child_finished_wall_iso": dt.datetime.now(dt.timezone.utc).isoformat(),
        "timeline": timeline,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{path.name}:{line_number} 不是 JSON object")
        records.append(payload)
    return records


def run_parent(args: argparse.Namespace) -> Path:
    llm_model_path = Path(args.llm_model).expanduser().resolve()
    tts_model_path = Path(args.tts_model).expanduser().resolve()
    ref_audio = Path(args.ref_audio).expanduser().resolve()
    if not llm_model_path.is_file():
        raise FileNotFoundError(f"找不到 GGUF: {llm_model_path}")
    if not tts_model_path.is_dir() or not ref_audio.is_file():
        raise FileNotFoundError("共置 parent 找不到 TTS 模型或参考音频")
    if args.runs < 1:
        raise ValueError("--runs 必须至少为 1")
    if args.max_gpu_sample_age_ms <= 0:
        raise ValueError("--max-gpu-sample-age-ms 必须为正")
    if args.max_swap_growth_mb < 0:
        raise ValueError("--max-swap-growth-mb 不能为负数")
    metadata = _discover_model_metadata(llm_model_path)
    model_layers = metadata["block_count"]
    explicit_candidates = parse_candidate_specs(args.candidates)
    if explicit_candidates:
        llm_config_names = tuple(
            dict.fromkeys(candidate.llm_config for candidate in explicit_candidates)
        )
        tts_config_names = tuple(
            dict.fromkeys(candidate.tts_config for candidate in explicit_candidates)
        )
        contexts = tuple(
            dict.fromkeys(candidate.context_tokens for candidate in explicit_candidates)
        )
    else:
        llm_config_names = tuple(name.upper() for name in args.llm_configs)
        tts_config_names = tuple(name.upper() for name in args.tts_configs)
        contexts = tuple(args.contexts)
    for name in llm_config_names:
        resolve_config(name, model_layers)
    for name in tts_config_names:
        placement_for(name)
    if not contexts or len(set(contexts)) != len(contexts):
        raise ValueError("--contexts 必须非空且不能重复")
    if args.generation_tokens < 0:
        raise ValueError("--generation-tokens 不能为负数")
    if any(
        context + args.generation_tokens > metadata["context_length"]
        for context in contexts
    ):
        raise ValueError("context + generation 超过 GGUF 声明的上下文长度")

    candidates = explicit_candidates or tuple(
        ColocatedCandidate(context, llm_config, tts_config)
        for context in contexts
        for llm_config in llm_config_names
        for tts_config in tts_config_names
    )
    plan = build_execution_plan(candidates, runs=args.runs)
    expected_records = len(plan)

    output_root = Path(args.output_root).expanduser().resolve()
    run_root_existed = (output_root / args.run_id).exists()
    writer = ExperimentArtifactWriter(
        output_root,
        args.run_id,
        allow_existing=args.resume,
    )
    source_git, git_complete = _git_source(args, writer.layout.root)
    llm_sha256 = sha256_file(llm_model_path)
    tts_sha256, tts_files = model_tree_hash(tts_model_path)
    ref_audio_sha256 = sha256_file(ref_audio)
    config = ExperimentRunConfig(
        experiment_name="llm_tts_colocated",
        run_id=args.run_id,
        seed=args.seed,
        sample_interval_s=args.sample_interval,
        gpu_sample_interval_s=args.gpu_sample_interval,
        backends=(
            BackendEvidence(
                component="llm",
                backend_kind="real_llama_cpp_gguf",
                is_real=True,
                model_path=str(llm_model_path),
                model_sha256=llm_sha256,
            ),
            BackendEvidence(
                component="tts",
                backend_kind="real_qwen3_tts_component_split",
                is_real=True,
                model_path=str(tts_model_path),
                model_sha256=tts_sha256,
            ),
        ),
        parameters={
            "llm_configs": llm_config_names,
            "tts_configs": tts_config_names,
            "contexts": contexts,
            "runs": args.runs,
            "model_layers": model_layers,
            "llm_physical_cores": args.llm_cores,
            "system_physical_cores": args.system_cores,
            "expected_records": expected_records,
            "explicit_candidates": [
                candidate.label for candidate in explicit_candidates
            ],
            "matrix_order": (
                "explicit_candidate_then_run"
                if explicit_candidates
                else "context_then_llm_then_tts_then_run"
            ),
            "inter_run_cooldown_s": args.inter_run_cooldown,
            "gpu_telemetry_backend": "direct_nvml",
            "gpu_cli_subprocess_allowed": False,
            "periodic_gpu_process_memory_query": False,
            "max_gpu_sample_age_ms": args.max_gpu_sample_age_ms,
            "max_swap_growth_mb": args.max_swap_growth_mb,
            "safety_margin_mb": args.safety_margin_mb,
            "tts_isolated_increment_mb": parse_isolated_increments(
                args.tts_isolated_increment_mb
            ),
            "n_ctx_formula": "context_tokens + generation_tokens",
            "generation_tokens": args.generation_tokens,
            "performance_timing": "diagnostic_stage_time_only",
            "capacity_formula": (
                "(P_colocated - B_idle) + G_safe <= gpu_total and "
                "P_colocated <= gpu_total；M_L 已含 KV buffer"
            ),
            "kv_measurement_semantics": KV_MEASUREMENT_SEMANTICS,
        },
    )
    manifest_payload = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git": {"source_copy": source_git},
        "git_provenance_complete": git_complete,
        "llm_model": {
            "path": str(llm_model_path),
            "size_bytes": llm_model_path.stat().st_size,
            "sha256": llm_sha256,
            "metadata": metadata,
        },
        "tts_model": {
            "path": str(tts_model_path),
            "sha256": tts_sha256,
            "files": tts_files,
        },
        "ref_audio": {
            "path": str(ref_audio),
            "sha256": ref_audio_sha256,
        },
        "kv_measurement_semantics": KV_MEASUREMENT_SEMANTICS,
    }
    if run_root_existed:
        if (
            not writer.layout.run_config.is_file()
            or not writer.layout.manifest.is_file()
        ):
            raise ValueError("续跑目录缺少 run_config.json 或 manifest.json")
        existing_config = json.loads(
            writer.layout.run_config.read_text(encoding="utf-8")
        )
        if existing_config != config.to_dict():
            raise ValueError("--resume 的冻结运行配置与已有产物不一致")
        existing_manifest = json.loads(
            writer.layout.manifest.read_text(encoding="utf-8")
        )
        expected_manifest = {
            "schema_version": SCHEMA_VERSION,
            "run_id": args.run_id,
            **manifest_payload,
        }
        if existing_manifest != expected_manifest:
            raise ValueError("--resume 的环境、源码或模型 manifest 不一致")
    else:
        writer.initialize(config, manifest_payload)

    records = _read_jsonl(writer.layout.iterations)
    expected_keys = {
        (context, llm_config, tts_config, run_index)
        for context, llm_config, tts_config, run_index, _ in plan
    }
    completed_keys: set[tuple[int, str, str, int]] = set()
    for record in records:
        key = (
            int(record["context_tokens"]),
            str(record["llm_config"]),
            str(record["tts_config"]),
            int(record["run_index"]),
        )
        if key not in expected_keys:
            raise ValueError(f"续跑产物包含矩阵外记录: {key}")
        if key in completed_keys:
            raise ValueError(f"续跑产物包含重复记录: {key}")
        completed_keys.add(key)

    for context_tokens, llm_config, tts_config, run_index, ordinal in plan:
        key = (context_tokens, llm_config, tts_config, run_index)
        if key in completed_keys:
            print(
                f"[{ordinal}/{expected_records}] resume skip "
                f"C{context_tokens} {llm_config}+{tts_config} r{run_index}",
                flush=True,
            )
            continue
        print(
            f"[{ordinal}/{expected_records}] run "
            f"C{context_tokens} {llm_config}+{tts_config} r{run_index}",
            flush=True,
        )
        child_json = (
            writer.layout.root
            / f"C{context_tokens}_{llm_config}_{tts_config}_r{run_index}_child.json"
        )
        child_log = (
            writer.layout.root
            / f"C{context_tokens}_{llm_config}_{tts_config}_r{run_index}.log"
        )
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--llm-config",
            llm_config,
            "--tts-config",
            tts_config,
            "--run-index",
            str(run_index),
            "--model-layers",
            str(model_layers),
            "--child-output-json",
            str(child_json),
            "--child-output-wav",
            str(
                writer.layout.root
                / f"C{context_tokens}_{llm_config}_{tts_config}_r{run_index}.wav"
            ),
            "--llm-model",
            str(llm_model_path),
            "--tts-model",
            str(tts_model_path),
            "--ref-audio",
            str(ref_audio),
            "--text",
            args.text,
            "--language",
            args.language,
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--repetition-penalty",
            str(args.repetition_penalty),
            "--context-tokens",
            str(context_tokens),
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
            "--max-swap-growth-mb",
            str(args.max_swap_growth_mb),
            "--safety-margin-mb",
            str(args.safety_margin_mb),
            "--seed",
            str(args.seed + run_index - 1),
            *(
                [
                    "--tts-isolated-increment-mb",
                    *args.tts_isolated_increment_mb,
                ]
                if args.tts_isolated_increment_mb
                else []
            ),
        ]
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=args.child_timeout,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            child_log.write_text(
                stdout + "\n[stderr]\n" + stderr,
                encoding="utf-8",
            )
            writer.append_failure(
                FailureRecord(
                    stage=(
                        f"C{context_tokens}_{llm_config}_{tts_config}_r{run_index}"
                    ),
                    error_type="ChildProcessTimeout",
                    message=f"timeout={args.child_timeout}",
                    monotonic_ns=time.monotonic_ns(),
                    attributes={"log": str(child_log)},
                )
            )
            if not args.continue_on_failure:
                break
            continue
        child_log.write_text(
            completed.stdout + "\n[stderr]\n" + completed.stderr,
            encoding="utf-8",
        )
        if completed.returncode != 0 or not child_json.is_file():
            writer.append_failure(
                FailureRecord(
                    stage=(
                        f"C{context_tokens}_{llm_config}_{tts_config}_r{run_index}"
                    ),
                    error_type="ChildProcessFailure",
                    message=f"returncode={completed.returncode}",
                    monotonic_ns=time.monotonic_ns(),
                    attributes={"log": str(child_log)},
                )
            )
            if not args.continue_on_failure:
                break
            continue
        record = json.loads(child_json.read_text(encoding="utf-8"))
        record["matrix_execution_ordinal"] = ordinal
        timeline = record.pop("timeline", [])
        for sample in timeline:
            writer.append_timeline(TimelineSample(**sample))
        writer.append_event(
            ExperimentEvent(
                name="llm_tts_colocated_child",
                monotonic_ns=record["child_started_monotonic_ns"],
                wall_time_iso=record["child_started_wall_iso"],
                phase=f"C{context_tokens}/{llm_config}+{tts_config}",
                attributes={
                    "context_tokens": context_tokens,
                    "run_index": run_index,
                    "matrix_execution_ordinal": ordinal,
                    "finished_monotonic_ns": record["child_finished_monotonic_ns"],
                },
            )
        )
        writer.append_iteration(record)
        records.append(record)
        completed_keys.add(key)
        failed_checks = tuple(
            name for name, passed in record.get("acceptance_checks", {}).items()
            if not passed
        )
        if failed_checks:
            writer.append_failure(
                FailureRecord(
                    stage=f"C{context_tokens}_{llm_config}_{tts_config}_r{run_index}",
                    error_type="MeasurementAcceptanceFailure",
                    message=",".join(failed_checks),
                    monotonic_ns=time.monotonic_ns(),
                    attributes={"failed_checks": list(failed_checks)},
                )
            )
        failure_suffix = (
            f" failed_checks={','.join(failed_checks)}" if failed_checks else ""
        )
        print(
            f"[{ordinal}/{expected_records}] complete "
            f"passed={record['passed']}{failure_suffix}",
            flush=True,
        )
        if not record["passed"] and not args.continue_on_failure:
            break
        if args.inter_run_cooldown > 0:
            time.sleep(args.inter_run_cooldown)

    failure_count = len(_read_jsonl(writer.layout.failures))
    passed = (
        git_complete
        and len(records) == expected_records
        and all(record["passed"] for record in records)
        and failure_count == 0
    )
    summary = {
        "passed": passed,
        "expected_records": expected_records,
        "actual_records": len(records),
        "failure_count": failure_count,
        "git_provenance_complete": git_complete,
        "capacity_formula": (
            "(P_colocated - B_idle) + G_safe <= gpu_total and "
            "P_colocated <= gpu_total；M_L 已含 KV buffer"
        ),
        "kv_measurement_semantics": KV_MEASUREMENT_SEMANTICS,
        "records": records,
        "evidence_scope": (
            "llm_tts_colocated_formal_repeated_matrix"
            if args.runs >= 5
            else "llm_tts_colocated_smoke_or_pilot"
        ),
    }
    writer.write_summary(summary)
    fields = [
        "context_tokens",
        "generation_tokens",
        "expected_state_tokens",
        "llm_config",
        "tts_config",
        "run_index",
        "matrix_execution_ordinal",
        "passed",
        "n_gpu_layers",
        "global_idle_baseline_mb",
        "b_llm_mb",
        "p_colocated_mb",
        "gpu_total_mb",
        "tts_colocated_increment_mb",
        "tts_isolated_increment_mb",
        "colocated_positive_residual_mb",
        "colocated_card_delta_mb",
        "physical_headroom_mb",
        "safe_capacity_used_mb",
        "capacity_headroom_mb",
        "capacity_admissible",
        "output_wav_size_bytes",
        "output_wav_sha256",
        "kv_gpu_buffer_mb",
        "kv_tokens_used",
        "kv_sequence_state_bytes",
        "full_state_serialized_bytes",
    ]
    with writer.layout.summary_csv.open("w", encoding="utf-8", newline="") as file_obj:
        csv_writer = csv.DictWriter(file_obj, fieldnames=fields)
        csv_writer.writeheader()
        for record in records:
            metrics = record["colocated_metrics"]
            csv_writer.writerow(
                {
                    "context_tokens": record["context_tokens"],
                    "generation_tokens": record["generation_tokens"],
                    "expected_state_tokens": record["expected_state_tokens"],
                    "llm_config": record["llm_config"],
                    "tts_config": record["tts_config"],
                    "run_index": record["run_index"],
                    "matrix_execution_ordinal": record["matrix_execution_ordinal"],
                    "passed": record["passed"],
                    "n_gpu_layers": record["n_gpu_layers"],
                    "global_idle_baseline_mb": metrics[
                        "global_idle_baseline_mb"
                    ],
                    "b_llm_mb": metrics["b_llm_mb"],
                    "p_colocated_mb": metrics["p_colocated_mb"],
                    "gpu_total_mb": metrics["gpu_total_mb"],
                    "tts_colocated_increment_mb": metrics[
                        "tts_colocated_increment_mb"
                    ],
                    "tts_isolated_increment_mb": metrics[
                        "tts_isolated_increment_mb"
                    ],
                    "colocated_positive_residual_mb": metrics[
                        "colocated_positive_residual_mb"
                    ],
                    "colocated_card_delta_mb": metrics[
                        "colocated_card_delta_mb"
                    ],
                    "physical_headroom_mb": metrics["physical_headroom_mb"],
                    "safe_capacity_used_mb": metrics["safe_capacity_used_mb"],
                    "capacity_headroom_mb": metrics["capacity_headroom_mb"],
                    "capacity_admissible": metrics["capacity_admissible"],
                    "output_wav_size_bytes": record["output_wav_size_bytes"],
                    "output_wav_sha256": record["output_wav_sha256"],
                    "kv_gpu_buffer_mb": record.get("kv_gpu_buffer_mb"),
                    "kv_tokens_used": record["kv_state"]["kv_tokens_used"],
                    "kv_sequence_state_bytes": record["kv_state"][
                        "kv_sequence_state_bytes"
                    ],
                    "full_state_serialized_bytes": record["kv_state"][
                        "full_state_serialized_bytes"
                    ],
                }
            )
    return writer.layout.root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--llm-model",
        default="/home/leslie/mvp_core/models/llm/Qwen3.5-4B-Q4_K_M.gguf",
    )
    parser.add_argument(
        "--tts-model",
        default="/home/leslie/mvp_core/models/Qwen3-TTS-12Hz-0.6B-Base",
    )
    parser.add_argument(
        "--ref-audio",
        default="/home/leslie/mvp_core/models/tts_refs/ref_calm.wav",
    )
    parser.add_argument("--text", default="你好，这是共置显存测量。")
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--llm-configs", nargs="+", default=["K16"])
    parser.add_argument("--tts-configs", nargs="+", default=["T0", "T3"])
    parser.add_argument("--contexts", nargs="+", type=int, default=[2048])
    parser.add_argument(
        "--candidates",
        nargs="+",
        help=(
            "显式候选，格式 CONTEXT:LLM:TTS；提供后覆盖 llm-configs/"
            "tts-configs/contexts，并按给定顺序 fail-fast"
        ),
    )
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--generation-tokens", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--safety-margin-mb", type=float, default=1024.0)
    parser.add_argument(
        "--tts-isolated-increment-mb",
        nargs="+",
        default=None,
        help="同配置 TTS 隔离增量，格式 CONFIG=value，例如 T0=1234.5 T3=890.0；"
        "缺省时残差保持 None，禁止估计",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--child-timeout", type=float, default=1800.0)
    parser.add_argument("--inter-run-cooldown", type=float, default=2.0)
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="子进程失败后继续矩阵；正式运行默认 fail-fast",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="只在冻结配置、源码、环境和模型完全一致时续跑缺失单元",
    )
    parser.add_argument(
        "--output-root",
        default=str(
            PROJECT_ROOT
            / "experiment_results"
            / "elastic_residency"
            / "llm_tts_colocated"
        ),
    )
    parser.add_argument(
        "--run-id",
        default=f"llm_tts_colocated_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument("--source-git-commit", default="")
    parser.add_argument("--source-git-dirty", choices=("true", "false"))
    parser.add_argument("--source-git-status-entry-count", type=int)
    parser.add_argument("--source-git-snapshot")
    parser.add_argument("--source-git-snapshot-sha256")
    parser.add_argument("--llm-config")
    parser.add_argument("--tts-config")
    parser.add_argument("--run-index", type=int, default=1)
    parser.add_argument("--model-layers", type=int, default=32)
    parser.add_argument("--child-output-json")
    parser.add_argument("--child-output-wav")
    parser.add_argument("--context-tokens", type=int, default=2048)
    parser.add_argument("--llm-cores", type=int, default=14)
    parser.add_argument("--system-cores", type=int, default=2)
    parser.add_argument("--sample-interval", type=float, default=0.05)
    parser.add_argument("--gpu-sample-interval", type=float, default=0.25)
    parser.add_argument("--gpu-warmup-timeout", type=float, default=5.0)
    parser.add_argument("--max-gpu-sample-age-ms", type=float, default=750.0)
    parser.add_argument("--max-swap-growth-mb", type=float, default=4.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.llm_config or args.tts_config:
        if not (args.llm_config and args.tts_config):
            raise SystemExit("--llm-config 与 --tts-config 必须成对用于 child 模式")
        try:
            result = run_child(args)
        except Exception as exc:
            if args.child_output_json:
                Path(args.child_output_json).write_text(
                    json.dumps(
                        {
                            "config": f"{args.llm_config}+{args.tts_config}",
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
    result_root = run_parent(args)
    print(result_root)
    summary = json.loads((result_root / "summary.json").read_text(encoding="utf-8"))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
