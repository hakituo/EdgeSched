#!/usr/bin/env python3
"""B2：Qwen3-TTS T0–T3 的 20/50/100 字、每档五轮真实矩阵。"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import psutil


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.elastic_residency.contracts import (  # noqa: E402
    BackendEvidence,
    ExperimentArtifactWriter,
    ExperimentRunConfig,
    FailureRecord,
)
from experiments.elastic_residency.tts_component_offload import (  # noqa: E402
    CONFIG_NAMES,
    _actual_devices,
    _apply_placement,
    _build_xvector_prompt,
    _component_inventory,
    _git_source,
    _memory_snapshot,
    _nvidia_memory_snapshot,
    _placement_matches,
    _synchronize,
    _wait_for_gpu_probe,
    placement_for,
)
from experiments.elastic_residency.tts_runtime_evidence import (  # noqa: E402
    SwapUsageSampler,
    compact_runtime_evidence,
    finalize_child_runtime_evidence,
)
from experiments.tts.profiling_utils import (  # noqa: E402
    model_tree_hash,
    percentile,
    sha256_file,
)


TEXTS = {
    "chars_20": "别着急，先喝口水，再继续处理后面的任务。",
    "chars_50": (
        "每个测试句子都会保存独立音频和结构化指标，后续可以从原始结果"
        "重新计算平均值、尾延迟和峰值显存。复核。"
    ),
    "chars_100": (
        "如果长句的实时率明显高于短句，说明生成成本会随文本长度快速增加，"
        "调度器就需要限制单个语音块的最大长度。反过来，如果不同句长的实"
        "时率比较稳定，我们可以把句子切分策略主要用于控制首块等待和播放缓"
        "冲。再测。"
    ),
}
EXPECTED_LENGTHS = {"chars_20": 20, "chars_50": 50, "chars_100": 100}
WARMUP_TEXT = "这是一次不计入统计的预热，用来稳定执行路径。"


def _validate_texts() -> None:
    actual = {name: len(text) for name, text in TEXTS.items()}
    if actual != EXPECTED_LENGTHS:
        raise ValueError(f"B2 冻结文本长度错误: {actual}")


def _cpu_time(process: psutil.Process) -> float:
    times = process.cpu_times()
    return float(times.user + times.system)


def _synthesize_once(
    *,
    wrapper: Any,
    prompt_dict: Any,
    text: str,
    language: str,
    generation_kwargs: dict[str, Any],
    codec_device_name: str,
    process: psutil.Process,
    baseline_card_used_mb: float,
) -> tuple[dict[str, Any], Any, int]:
    import numpy as np
    import torch

    cpu_started = _cpu_time(process)
    tokenize_started = time.perf_counter()
    input_text = wrapper._build_assistant_text(text)
    input_ids = wrapper._tokenize_texts([input_text])
    text_tokenize_s = time.perf_counter() - tokenize_started
    tokenize_cpu_s = _cpu_time(process) - cpu_started

    _synchronize()
    torch.cuda.reset_peak_memory_stats()
    cpu_started = _cpu_time(process)
    talker_started = time.perf_counter()
    codes, _ = wrapper.model.generate(
        input_ids=input_ids,
        ref_ids=None,
        voice_clone_prompt=prompt_dict,
        languages=[language],
        non_streaming_mode=True,
        **generation_kwargs,
    )
    _synchronize()
    talker_generate_s = time.perf_counter() - talker_started
    talker_cpu_s = _cpu_time(process) - cpu_started
    memory_talker_peak = _memory_snapshot(
        process, baseline_card_used_mb=baseline_card_used_mb
    )

    codec_device = torch.device(codec_device_name)
    code_devices_before = [str(code.device) for code in codes]
    _synchronize()
    torch.cuda.reset_peak_memory_stats()
    cpu_started = _cpu_time(process)
    transfer_started = time.perf_counter()
    decode_codes = [code.to(codec_device) for code in codes]
    _synchronize()
    code_transfer_s = time.perf_counter() - transfer_started
    transfer_cpu_s = _cpu_time(process) - cpu_started
    code_devices_after = [str(code.device) for code in decode_codes]
    memory_transfer_peak = _memory_snapshot(
        process, baseline_card_used_mb=baseline_card_used_mb
    )

    torch.cuda.reset_peak_memory_stats()
    cpu_started = _cpu_time(process)
    codec_started = time.perf_counter()
    wavs, sample_rate = wrapper.model.speech_tokenizer.decode(
        [{"audio_codes": code} for code in decode_codes]
    )
    _synchronize()
    codec_decode_s = time.perf_counter() - codec_started
    codec_cpu_s = _cpu_time(process) - cpu_started
    memory_codec_peak = _memory_snapshot(
        process, baseline_card_used_mb=baseline_card_used_mb
    )

    waveform = (
        wavs[0].detach().float().cpu().numpy()
        if torch.is_tensor(wavs[0])
        else np.asarray(wavs[0])
    )
    waveform_finite = bool(np.isfinite(waveform).all())
    waveform_rms = float(np.sqrt(np.mean(np.square(waveform))))
    waveform_peak = float(np.max(np.abs(waveform)))
    synthesis_s = (
        text_tokenize_s + talker_generate_s + code_transfer_s + codec_decode_s
    )
    cpu_time_s = tokenize_cpu_s + talker_cpu_s + transfer_cpu_s + codec_cpu_s
    audio_duration_s = len(waveform) / sample_rate
    metrics = {
        "text_tokenize_s": text_tokenize_s,
        "talker_generate_s": talker_generate_s,
        "code_transfer_s": code_transfer_s,
        "codec_decode_s": codec_decode_s,
        "synthesis_s": synthesis_s,
        "ttfa_s": synthesis_s,
        "cpu_time_s": cpu_time_s,
        "process_cpu_percent": cpu_time_s / synthesis_s * 100.0,
        "audio_duration_s": audio_duration_s,
        "rtf": synthesis_s / audio_duration_s,
        "waveform_finite": waveform_finite,
        "waveform_rms": waveform_rms,
        "waveform_peak": waveform_peak,
        "code_devices_before_transfer": code_devices_before,
        "code_devices_after_transfer": code_devices_after,
        "codec_code_shapes": [list(code.shape) for code in codes],
        "memory_talker_peak": memory_talker_peak,
        "memory_transfer_peak": memory_transfer_peak,
        "memory_codec_peak": memory_codec_peak,
    }
    return metrics, waveform, int(sample_rate)


def _run_child_workload(args: argparse.Namespace) -> dict[str, Any]:
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

    _validate_texts()
    spec = placement_for(args.child_config)
    model_path = Path(args.model).expanduser().resolve()
    ref_audio = Path(args.ref_audio).expanduser().resolve()
    output_dir = Path(args.child_output_dir).expanduser().resolve()
    if not model_path.is_dir() or not ref_audio.is_file():
        raise FileNotFoundError("B2 child 找不到模型或参考音频")
    if not torch.cuda.is_available():
        raise RuntimeError("B2 真实矩阵要求 CUDA")
    if not _wait_for_gpu_probe(args.gpu_probe_timeout):
        raise RuntimeError("B2 计时前未获得可用 NVIDIA GPU 样本")
    output_dir.mkdir(parents=True, exist_ok=True)
    process = psutil.Process()
    process.cpu_percent(interval=None)
    memory_baseline = _nvidia_memory_snapshot(os.getpid())
    baseline_card_used_mb = memory_baseline["gpu_card_used_mb"]
    if baseline_card_used_mb is None:
        raise RuntimeError("B2 无法获得空闲整卡显存基线")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    wrapper = Qwen3TTSModel.from_pretrained(
        str(model_path),
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    )
    _synchronize()
    load_s = time.perf_counter() - load_started
    placement_started = time.perf_counter()
    _apply_placement(wrapper, spec)
    placement_s = time.perf_counter() - placement_started
    prompt, prompt_build_s = _build_xvector_prompt(wrapper, ref_audio)
    prompt_device = str(prompt.ref_spk_embedding.device)
    if spec.unload_speaker_after_prompt:
        wrapper.model.speaker_encoder = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    inventory_ready = _component_inventory(wrapper)
    actual_devices = _actual_devices(inventory_ready)
    memory_ready = _memory_snapshot(
        process, baseline_card_used_mb=baseline_card_used_mb
    )
    prompt_dict = wrapper._prompt_items_to_voice_clone_prompt([prompt])
    generation_kwargs = wrapper._merge_generate_kwargs(
        do_sample=False,
        subtalker_dosample=False,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
    )

    warmup_metrics, warmup_wav, warmup_rate = _synthesize_once(
        wrapper=wrapper,
        prompt_dict=prompt_dict,
        text=WARMUP_TEXT,
        language=args.language,
        generation_kwargs=generation_kwargs,
        codec_device_name=spec.speech_codec_device,
        process=process,
        baseline_card_used_mb=baseline_card_used_mb,
    )
    sf.write(output_dir / "warmup.wav", warmup_wav, warmup_rate, subtype="PCM_16")

    records: list[dict[str, Any]] = []
    tiers = list(TEXTS)
    iteration = 0
    for round_index in range(args.rounds):
        tier_order = tiers[round_index % len(tiers) :] + tiers[: round_index % len(tiers)]
        for tier in tier_order:
            iteration += 1
            text = TEXTS[tier]
            metrics, waveform, sample_rate = _synthesize_once(
                wrapper=wrapper,
                prompt_dict=prompt_dict,
                text=text,
                language=args.language,
                generation_kwargs=generation_kwargs,
                codec_device_name=spec.speech_codec_device,
                process=process,
                baseline_card_used_mb=baseline_card_used_mb,
            )
            wav_path = output_dir / f"{iteration:02d}_{tier}_r{round_index + 1}.wav"
            write_started = time.perf_counter()
            sf.write(wav_path, waveform, sample_rate, subtype="PCM_16")
            wav_write_s = time.perf_counter() - write_started
            memory_phases = (
                metrics["memory_talker_peak"],
                metrics["memory_transfer_peak"],
                metrics["memory_codec_peak"],
            )
            checks = {
                "gpu_telemetry_is_direct_nvml": all(
                    snapshot["gpu_telemetry_backend"] == "direct_nvml"
                    and not snapshot["gpu_cli_subprocess_used"]
                    for snapshot in memory_phases
                ),
                "component_placement_matches": _placement_matches(
                    spec, actual_devices
                ),
                "prompt_device_matches_speaker_stage": (
                    prompt_device == spec.speaker_device
                ),
                "codes_start_on_talker_device": all(
                    device == spec.talker_device
                    for device in metrics["code_devices_before_transfer"]
                ),
                "codes_arrive_on_codec_device": all(
                    device == spec.speech_codec_device
                    for device in metrics["code_devices_after_transfer"]
                ),
                "timing_segments_positive": all(
                    metrics[name] > 0
                    for name in (
                        "text_tokenize_s",
                        "talker_generate_s",
                        "code_transfer_s",
                        "codec_decode_s",
                    )
                ),
                "gpu_memory_evidence_available": all(
                    snapshot["gpu_card_available"]
                    and snapshot["gpu_card_delta_from_baseline_mb"] is not None
                    for snapshot in memory_phases
                ),
                "torch_phase_peaks_available": all(
                    snapshot["torch_peak_allocated_bytes"] > 0
                    and snapshot["torch_peak_reserved_bytes"] > 0
                    for snapshot in memory_phases
                ),
                "waveform_finite": metrics["waveform_finite"],
                "waveform_non_silent": metrics["waveform_rms"] > 1e-5,
                "waveform_peak_valid": 0 < metrics["waveform_peak"] <= 1.05,
                "wav_written": wav_path.is_file() and wav_path.stat().st_size > 44,
            }
            records.append(
                {
                    "config": spec.name,
                    "iteration": iteration,
                    "round": round_index + 1,
                    "tier": tier,
                    "text": text,
                    "text_chars": len(text),
                    "passed": all(checks.values()),
                    "fallback_occurred": False,
                    "acceptance_checks": checks,
                    **metrics,
                    "sample_rate": sample_rate,
                    "wav_write_s": wav_write_s,
                    "output_wav": str(wav_path),
                    "output_wav_sha256": sha256_file(wav_path),
                }
            )

    return {
        "config": spec.name,
        "passed": len(records) == args.rounds * 3
        and all(record["passed"] for record in records),
        "placement": vars(spec),
        "actual_devices": actual_devices,
        "prompt_device": prompt_device,
        "inventory_ready": inventory_ready,
        "memory_baseline": memory_baseline,
        "memory_ready": memory_ready,
        "load_s": load_s,
        "placement_s": placement_s,
        "prompt_build_s": prompt_build_s,
        "warmup": warmup_metrics,
        "records": records,
    }


def run_child(args: argparse.Namespace) -> dict[str, Any]:
    """执行真实 TTS 工作负载，并补齐全子进程 swap/GPU 新鲜度门禁。"""

    swap_sampler = SwapUsageSampler(
        interval_s=args.swap_sample_interval,
        tolerance_bytes=int(args.max_swap_growth_mb * 1024 * 1024),
    )
    swap_sampler.start()
    try:
        child = _run_child_workload(args)
    finally:
        swap_report = swap_sampler.stop()
    return finalize_child_runtime_evidence(
        child,
        swap_report=swap_report,
        max_gpu_freshness_ms=args.max_gpu_sample_freshness_ms,
    )


def _metric_summary(records: list[dict[str, Any]], name: str) -> dict[str, float]:
    values = [float(record[name]) for record in records]
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def _summarize_group(records: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = (
        "ttfa_s",
        "talker_generate_s",
        "code_transfer_s",
        "codec_decode_s",
        "audio_duration_s",
        "rtf",
        "process_cpu_percent",
    )
    return {
        "samples": len(records),
        "passed_count": sum(bool(record["passed"]) for record in records),
        "all_passed": all(record["passed"] for record in records),
        "rtf_below_one_count": sum(record["rtf"] < 1.0 for record in records),
        **{name: _metric_summary(records, name) for name in metrics},
        "torch_peak_allocated_mb": max(
            max(
                record[phase]["torch_peak_allocated_bytes"]
                for phase in (
                    "memory_talker_peak",
                    "memory_transfer_peak",
                    "memory_codec_peak",
                )
            )
            / 1048576.0
            for record in records
        ),
        "torch_peak_reserved_mb": max(
            max(
                record[phase]["torch_peak_reserved_bytes"]
                for phase in (
                    "memory_talker_peak",
                    "memory_transfer_peak",
                    "memory_codec_peak",
                )
            )
            / 1048576.0
            for record in records
        ),
        "peak_rss_mb": max(
            max(
                record[phase]["rss_bytes"]
                for phase in (
                    "memory_talker_peak",
                    "memory_transfer_peak",
                    "memory_codec_peak",
                )
            )
            / 1048576.0
            for record in records
        ),
        "max_card_delta_mb": max(
            max(
                record[phase]["gpu_card_delta_from_baseline_mb"] or 0.0
                for phase in (
                    "memory_talker_peak",
                    "memory_transfer_peak",
                    "memory_codec_peak",
                )
            )
            for record in records
        ),
    }


def _pareto_configs(config_summaries: dict[str, dict[str, Any]]) -> list[str]:
    names = list(config_summaries)
    pareto: list[str] = []
    for name in names:
        candidate = config_summaries[name]
        objectives = (
            candidate["max_card_delta_mb"],
            candidate["ttfa_s"]["p95"],
            candidate["rtf"]["p95"],
        )
        dominated = False
        for other_name in names:
            if other_name == name:
                continue
            other = config_summaries[other_name]
            other_objectives = (
                other["max_card_delta_mb"],
                other["ttfa_s"]["p95"],
                other["rtf"]["p95"],
            )
            if all(left <= right for left, right in zip(other_objectives, objectives)) and any(
                left < right for left, right in zip(other_objectives, objectives)
            ):
                dominated = True
                break
        if not dominated:
            pareto.append(name)
    return pareto


def run_parent(args: argparse.Namespace) -> Path:
    _validate_texts()
    model_path = Path(args.model).expanduser().resolve()
    ref_audio = Path(args.ref_audio).expanduser().resolve()
    model_sha256, model_files = model_tree_hash(model_path)
    writer = ExperimentArtifactWriter(args.output_root, args.run_id)
    source_git, git_complete = _git_source(args, writer.layout.root)
    config_names = tuple(name.upper() for name in args.configs)
    expected_records = len(config_names) * len(TEXTS) * args.rounds
    config = ExperimentRunConfig(
        experiment_name="tts_component_offload_b2_matrix",
        run_id=args.run_id,
        seed=args.seed,
        backends=(
            BackendEvidence(
                component="tts",
                backend_kind="real_qwen3_tts_component_split",
                is_real=True,
                model_path=str(model_path),
                model_sha256=model_sha256,
            ),
        ),
        parameters={
            "configs": config_names,
            "texts": TEXTS,
            "rounds": args.rounds,
            "warmups_per_config": 1,
            "expected_records": expected_records,
            "language": args.language,
            "max_new_tokens": args.max_new_tokens,
            "x_vector_only_mode": True,
            "true_audio_streaming": False,
            "ttfa_semantics": "complete_sentence_audio_ready_in_memory",
            "gpu_telemetry_backend": "direct_nvml",
            "gpu_cli_subprocess_allowed": False,
            "gpu_sample_freshness_semantics": (
                "synchronous_direct_nvml_query_duration_upper_bound"
            ),
            "max_gpu_sample_freshness_ms": args.max_gpu_sample_freshness_ms,
            "swap_sample_interval_s": args.swap_sample_interval,
            "max_swap_growth_mb": args.max_swap_growth_mb,
        },
    )
    writer.initialize(
        config,
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "git": {"source_copy": source_git},
            "git_provenance_complete": git_complete,
            "model_file_count": model_files,
            "reference_audio": str(ref_audio),
            "reference_audio_sha256": sha256_file(ref_audio),
        },
    )

    child_results: list[dict[str, Any]] = []
    all_records: list[dict[str, Any]] = []
    for config_name in config_names:
        placement_for(config_name)
        child_json = writer.layout.root / f"{config_name}_child.json"
        child_dir = writer.layout.root / "audio" / config_name
        child_log = writer.layout.root / f"{config_name}.log"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child-config",
            config_name,
            "--child-output-json",
            str(child_json),
            "--child-output-dir",
            str(child_dir),
            "--model",
            str(model_path),
            "--ref-audio",
            str(ref_audio),
            "--language",
            args.language,
            "--rounds",
            str(args.rounds),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--repetition-penalty",
            str(args.repetition_penalty),
            "--seed",
            str(args.seed),
            "--gpu-probe-timeout",
            str(args.gpu_probe_timeout),
            "--max-gpu-sample-freshness-ms",
            str(args.max_gpu_sample_freshness_ms),
            "--swap-sample-interval",
            str(args.swap_sample_interval),
            "--max-swap-growth-mb",
            str(args.max_swap_growth_mb),
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
                    stage=config_name,
                    error_type="ChildProcessFailure",
                    message=f"returncode={completed.returncode}",
                    monotonic_ns=time.monotonic_ns(),
                    attributes={"log": str(child_log)},
                )
            )
            child_results.append({"config": config_name, "passed": False})
            continue
        child = json.loads(child_json.read_text(encoding="utf-8"))
        child_results.append(child)
        if not child.get("passed", False):
            writer.append_failure(
                FailureRecord(
                    stage=config_name,
                    error_type="ChildAcceptanceFailure",
                    message="TTS child 运行完成但证据门禁未全部通过",
                    monotonic_ns=time.monotonic_ns(),
                    attributes={
                        "runtime_evidence": compact_runtime_evidence(child),
                        "failed_record_count": sum(
                            record.get("passed") is not True
                            for record in child.get("records", [])
                        ),
                    },
                )
            )
        for record in child["records"]:
            writer.append_iteration(record)
            all_records.append(record)

    by_config = {
        name: _summarize_group(
            [record for record in all_records if record["config"] == name]
        )
        for name in config_names
        if any(record["config"] == name for record in all_records)
    }
    by_config_tier = {
        name: {
            tier: _summarize_group(
                [
                    record
                    for record in all_records
                    if record["config"] == name and record["tier"] == tier
                ]
            )
            for tier in TEXTS
        }
        for name in by_config
    }
    passed = (
        git_complete
        and len(all_records) == expected_records
        and len(child_results) == len(config_names)
        and all(child.get("passed", False) for child in child_results)
        and all(record["passed"] for record in all_records)
    )
    summary = {
        "passed": passed,
        "expected_records": expected_records,
        "actual_records": len(all_records),
        "git_provenance_complete": git_complete,
        "by_config": by_config,
        "by_config_tier": by_config_tier,
        "pareto_objectives": ["max_card_delta_mb", "ttfa_p95", "rtf_p95"],
        "pareto_configs": _pareto_configs(by_config) if by_config else [],
        "capacity_decision": "pending_llm_layer_increment_from_stage_c",
        "performance_evidence_scope": "b2_tts_only_matrix_not_paper_main",
        "children": [
            {
                "config": child.get("config"),
                "passed": child.get("passed", False),
                "actual_devices": child.get("actual_devices"),
                "memory_ready": child.get("memory_ready"),
                "load_s": child.get("load_s"),
                "placement_s": child.get("placement_s"),
                "prompt_build_s": child.get("prompt_build_s"),
                "runtime_evidence": compact_runtime_evidence(child),
            }
            for child in child_results
        ],
    }
    writer.write_summary(summary)
    fields = [
        "config",
        "tier",
        "samples",
        "ttfa_mean_s",
        "ttfa_p95_s",
        "rtf_mean",
        "rtf_p95",
        "rtf_below_one_count",
        "talker_mean_s",
        "codec_mean_s",
        "transfer_mean_ms",
        "cpu_mean_percent",
        "max_card_delta_mb",
        "torch_peak_allocated_mb",
        "torch_peak_reserved_mb",
        "peak_rss_mb",
    ]
    with writer.layout.summary_csv.open(
        "w", encoding="utf-8", newline=""
    ) as file_obj:
        csv_writer = csv.DictWriter(file_obj, fieldnames=fields)
        csv_writer.writeheader()
        for name, tiers in by_config_tier.items():
            for tier, item in tiers.items():
                csv_writer.writerow(
                    {
                        "config": name,
                        "tier": tier,
                        "samples": item["samples"],
                        "ttfa_mean_s": item["ttfa_s"]["mean"],
                        "ttfa_p95_s": item["ttfa_s"]["p95"],
                        "rtf_mean": item["rtf"]["mean"],
                        "rtf_p95": item["rtf"]["p95"],
                        "rtf_below_one_count": item["rtf_below_one_count"],
                        "talker_mean_s": item["talker_generate_s"]["mean"],
                        "codec_mean_s": item["codec_decode_s"]["mean"],
                        "transfer_mean_ms": item["code_transfer_s"]["mean"]
                        * 1000.0,
                        "cpu_mean_percent": item["process_cpu_percent"]["mean"],
                        "max_card_delta_mb": item["max_card_delta_mb"],
                        "torch_peak_allocated_mb": item[
                            "torch_peak_allocated_mb"
                        ],
                        "torch_peak_reserved_mb": item[
                            "torch_peak_reserved_mb"
                        ],
                        "peak_rss_mb": item["peak_rss_mb"],
                    }
                )
    return writer.layout.root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="/home/leslie/mvp_core/models/Qwen3-TTS-12Hz-0.6B-Base",
    )
    parser.add_argument(
        "--ref-audio",
        default="/home/leslie/mvp_core/models/tts_refs/ref_calm.wav",
    )
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--configs", nargs="+", default=list(CONFIG_NAMES))
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-probe-timeout", type=float, default=5.0)
    parser.add_argument("--max-gpu-sample-freshness-ms", type=float, default=750.0)
    parser.add_argument("--swap-sample-interval", type=float, default=0.05)
    parser.add_argument("--max-swap-growth-mb", type=float, default=0.0)
    parser.add_argument("--child-timeout", type=float, default=3600.0)
    parser.add_argument(
        "--output-root",
        default=str(
            PROJECT_ROOT / "experiment_results" / "elastic_residency" / "b2"
        ),
    )
    parser.add_argument(
        "--run-id",
        default=f"b2_tts_matrix_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument("--source-git-commit", default="")
    parser.add_argument("--source-git-dirty", choices=("true", "false"))
    parser.add_argument("--source-git-status-entry-count", type=int)
    parser.add_argument("--source-git-snapshot")
    parser.add_argument("--source-git-snapshot-sha256")
    parser.add_argument("--child-config", choices=CONFIG_NAMES)
    parser.add_argument("--child-output-json")
    parser.add_argument("--child-output-dir")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.rounds < 1:
        raise ValueError("--rounds 必须大于 0")
    if args.max_gpu_sample_freshness_ms <= 0:
        raise ValueError("--max-gpu-sample-freshness-ms 必须大于 0")
    if args.swap_sample_interval <= 0:
        raise ValueError("--swap-sample-interval 必须大于 0")
    if args.max_swap_growth_mb < 0:
        raise ValueError("--max-swap-growth-mb 不能为负数")
    if args.child_config:
        try:
            result = run_child(args)
        except Exception as exc:
            if args.child_output_json:
                Path(args.child_output_json).write_text(
                    json.dumps(
                        {
                            "config": args.child_config,
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
