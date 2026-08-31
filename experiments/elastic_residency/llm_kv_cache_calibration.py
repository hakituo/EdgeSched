#!/usr/bin/env python3
"""KV Cache 容量与状态语义校准（独立入口，不影响 C2/C3 性能曲线）。

与 ``llm_residency_curve.py`` 的关系：

- 复用同一模型路径、CPU affinity、direct NVML 与 artifact contract；
- 模型初始化参数与 C2/C3 一致（n_gpu_layers/offload_kqv/op_offload/
  flash_attn/use_mmap/seed/n_threads），``n_ctx`` 与填充状态均精确等于
  ``context + generation``；默认 generation=128，与 C2/C3 推理后状态一致；
- KV 状态查询全部位于推理窗口之外（本入口根本不做性能计时）；
- KV buffer 分配由 n_ctx 确定性决定，因此每个 (config, context)
  只需 1 轮，不需要 C2/C3 的五轮重复。

正式矩阵：16 configs × 3 contexts × 1 run = 48 条。
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
    BackendEvidence,
    ExperimentArtifactWriter,
    ExperimentEvent,
    ExperimentRunConfig,
    FailureRecord,
    SCHEMA_VERSION,
    TimelineSample,
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
    _native_memory_snapshot,
    _token_ids_sha256,
    resolve_config,
)
from experiments.elastic_residency.resource_sampler import ResourceSampler  # noqa: E402
from experiments.elastic_residency.tts_component_offload import (  # noqa: E402
    _git_source,
    _nvidia_memory_snapshot,
)
from experiments.tts.profiling_utils import sha256_file  # noqa: E402


FORMAL_CONFIGS = (
    "K0",
    "K2",
    "K4",
    "K6",
    "K8",
    "K10",
    "K12",
    "K14",
    "K16",
    "K18",
    "K20",
    "K24",
    "K28",
    "K30",
    "K32",
    "FULL",
)
FORMAL_CONTEXTS = (512, 2048, 4096)


def _kv_prefill(llm: Any, prompt_tokens: list[int]) -> float:
    """填充 context（唯一一次 eval），返回耗时仅作诊断，不作性能证据。"""

    started = time.perf_counter()
    llm.eval(prompt_tokens)
    return time.perf_counter() - started


def run_child(args: argparse.Namespace) -> dict[str, Any]:
    from llama_cpp import Llama

    model_path = Path(args.model).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"找不到 GGUF: {model_path}")
    topology = CPUTopology.discover()
    required_cores = args.system_cores + args.llm_cores
    if len(topology.physical_cores) < required_cores:
        raise RuntimeError(
            f"KV 校准需要 {required_cores} 个物理核，实际只有 "
            f"{len(topology.physical_cores)}"
        )
    n_gpu_layers = resolve_config(args.child_config, args.model_layers)
    expected_state_tokens = args.context_tokens + args.generation_tokens
    offload_kqv = n_gpu_layers != 0
    allocator = PhysicalCoreAllocator(topology)
    system_lease = allocator.reserve_system(args.system_cores)
    llm_lease = allocator.acquire(
        "llm_kv_calibration",
        args.llm_cores,
        use_smt=True,
    )
    process = psutil.Process()
    baseline = _nvidia_memory_snapshot(os.getpid())
    baseline_card_used_mb = baseline["gpu_card_used_mb"]
    if baseline_card_used_mb is None:
        raise RuntimeError("KV 校准无法读取整卡显存基线")

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
            load_started_ns = time.monotonic_ns()
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
            load_s = time.perf_counter() - load_started
            load_finished_ns = time.monotonic_ns()
            buffer_report = log_capture.buffer_report()
            metadata = dict(llm.metadata)
            architecture = str(metadata.get("general.architecture", ""))
            actual_layers = int(metadata[f"{architecture}.block_count"])
            memory_loaded = _native_memory_snapshot(
                process=process,
                baseline_card_used_mb=baseline["gpu_card_used_mb"],
            )
            # 加载后、prefill 前：KV 缓存应为空（position 为 -1）。
            kv_state_before_prefill = capture_llama_kv_state(llm)
            prompt_tokens = _build_exact_prompt_tokens(llm, expected_state_tokens)
            prefill_s = _kv_prefill(llm, prompt_tokens)
            prefill_finished_ns = time.monotonic_ns()
            # KV 查询在 eval 之后、计时窗口之外（prefill_s 已在此前固定）。
            kv_state = capture_llama_kv_state(llm)
            memory_after_prefill = _native_memory_snapshot(
                process=process,
                baseline_card_used_mb=baseline["gpu_card_used_mb"],
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
                baseline_card_used_mb=baseline["gpu_card_used_mb"],
            )
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

    buffer_acceptance = buffer_report_acceptance(buffer_report)
    acceptance_checks = {
        "model_layer_count_matches": actual_layers == args.model_layers,
        "state_token_count_exact": len(prompt_tokens) == expected_state_tokens,
        "kv_positions_empty_before_prefill": (
            kv_state_before_prefill["kv_position_min"] == -1
            and kv_state_before_prefill["kv_position_max"] == -1
            and kv_state_before_prefill["kv_tokens_used"] == 0
        ),
        "kv_position_bounds_valid": (
            0 <= kv_state["kv_position_min"] <= kv_state["kv_position_max"]
            and kv_state["kv_position_max"] == expected_state_tokens - 1
        ),
        "kv_tokens_used_exact": (
            kv_state["kv_tokens_used"] == expected_state_tokens
        ),
        "kv_sequence_state_bytes_positive": (
            kv_state["kv_sequence_state_bytes"] > 0
        ),
        "full_state_serialized_bytes_positive": (
            kv_state["full_state_serialized_bytes"] > 0
        ),
        "kv_state_bytes_consistent_with_full_state": (
            kv_state["full_state_serialized_bytes"] >= kv_state["kv_sequence_state_bytes"]
        ),
        **buffer_acceptance,
        "gpu_ready_before_timeline": gpu_ready,
        "gpu_present_for_every_timeline_sample": (
            resource_summary["gpu_unavailable_timeline_sample_count"] == 0
        ),
        "worker_affinity_matches": set(actual_affinity) == set(llm_lease.affinity),
        "sampler_affinity_matches": (
            set(resource_summary["sampler_thread_affinity"])
            == set(system_lease.affinity)
        ),
        "gpu_sampler_affinity_matches": (
            set(resource_summary["gpu_thread_affinity"]) == set(system_lease.affinity)
        ),
        "swap_unused": all(
            sample.get("system_swap_used_bytes") == 0 for sample in timeline
        ),
        "all_leases_released": allocator_after_release["active_lease_count"] == 0,
        "all_physical_cores_returned": (
            allocator_after_release["free_physical_core_count"]
            == len(topology.physical_cores)
        ),
        "gpu_memory_evidence_available": (
            memory_after_prefill["gpu_card_delta_from_baseline_mb"] is not None
        ),
        "gpu_telemetry_is_direct_nvml": (
            resource_summary["gpu_telemetry_backend"] == "direct_nvml"
            and not resource_summary["gpu_cli_subprocess_used"]
            and memory_after_prefill["gpu_telemetry_backend"] == "direct_nvml"
            and not memory_after_prefill["gpu_cli_subprocess_used"]
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
        "expected_state_tokens": expected_state_tokens,
        "n_ctx": expected_state_tokens,
        "llm_physical_cores": args.llm_cores,
        "llm_logical_affinity": list(actual_affinity),
        "llm_threads": len(llm_lease.affinity),
        "system_physical_cores": args.system_cores,
        "system_logical_affinity": list(system_lease.affinity),
        "prompt_tokens_sha256": _token_ids_sha256(prompt_tokens),
        "load_s": load_s,
        "prefill_s_diagnostic_only": prefill_s,
        "unload_s": unload_s,
        "kv_state_before_prefill": kv_state_before_prefill,
        "kv_state": kv_state,
        # 兼容字段：与 full_state_serialized_bytes 同义。
        "kv_state_bytes": kv_state["full_state_serialized_bytes"],
        "buffer_report": buffer_report,
        "kv_gpu_buffer_mb": buffer_report.get("kv_gpu_buffer_mb"),
        "kv_cpu_buffer_mb": buffer_report.get("kv_cpu_buffer_mb"),
        "compute_gpu_buffer_mb": buffer_report.get("compute_gpu_buffer_mb"),
        "compute_cpu_buffer_mb": buffer_report.get("compute_cpu_buffer_mb"),
        "memory_baseline": baseline,
        "memory_loaded": memory_loaded,
        "memory_after_prefill": memory_after_prefill,
        "memory_after_unload": memory_after_unload,
        "resource_summary": resource_summary,
        "phase_intervals_ns": {
            "load": [load_started_ns, load_finished_ns],
            "prefill": [load_finished_ns, prefill_finished_ns],
            "unload": [unload_started_ns, unload_finished_ns],
        },
        "quality_warnings": {
            "gpu_process_memory_is_estimated": (
                memory_after_prefill["gpu_process_total_mb"] is None
            ),
        },
        "allocator_after_release": allocator_after_release,
        "child_started_monotonic_ns": child_started_ns,
        "child_started_wall_iso": child_started_wall_iso,
        "child_finished_monotonic_ns": time.monotonic_ns(),
        "child_finished_wall_iso": dt.datetime.now(dt.timezone.utc).isoformat(),
        "timeline": timeline,
    }


def _evidence_scope(
    contexts: tuple[int, ...],
    configs: tuple[str, ...],
    runs: int,
) -> str:
    if (
        contexts == FORMAL_CONTEXTS
        and runs == 1
        and len(configs) == len(FORMAL_CONFIGS)
        and set(configs) == set(FORMAL_CONFIGS)
    ):
        return "kv_calibration_formal_48"
    return "kv_calibration_smoke"


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
    model_path = Path(args.model).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"找不到 GGUF: {model_path}")
    if args.runs != 1:
        raise ValueError("KV 校准是确定性容量数据，--runs 固定为 1")
    metadata = _discover_model_metadata(model_path)
    model_layers = metadata["block_count"]
    config_names = tuple(name.upper() for name in args.configs)
    for name in config_names:
        resolve_config(name, model_layers)
    contexts = tuple(args.contexts)
    if not contexts or len(set(contexts)) != len(contexts):
        raise ValueError("--contexts 必须非空且不能重复")
    if not config_names or len(set(config_names)) != len(config_names):
        raise ValueError("--configs 必须非空且不能重复")
    if any(context < 2 for context in contexts):
        raise ValueError("所有 context token 数必须至少为 2")
    if args.generation_tokens < 0:
        raise ValueError("--generation-tokens 不能为负数")
    if any(
        context + args.generation_tokens > metadata["context_length"]
        for context in contexts
    ):
        raise ValueError("context + generation 超过 GGUF 声明的上下文长度")

    plan: list[tuple[int, str, int]] = []
    ordinal = 0
    for context_tokens in contexts:
        for config_name in config_names:
            ordinal += 1
            plan.append((context_tokens, config_name, ordinal))
    expected_records = len(plan)

    output_root = Path(args.output_root).expanduser().resolve()
    run_root_existed = (output_root / args.run_id).exists()
    writer = ExperimentArtifactWriter(
        output_root,
        args.run_id,
        allow_existing=args.resume,
    )
    source_git, git_complete = _git_source(args, writer.layout.root)
    model_sha256 = sha256_file(model_path)
    config = ExperimentRunConfig(
        experiment_name="llm_kv_cache_calibration",
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
            "runs": 1,
            "model_layers": model_layers,
            "contexts": contexts,
            "llm_physical_cores": args.llm_cores,
            "system_physical_cores": args.system_cores,
            "expected_records": expected_records,
            "matrix_order": "context_then_config",
            "inter_run_cooldown_s": args.inter_run_cooldown,
            "gpu_telemetry_backend": "direct_nvml",
            "gpu_cli_subprocess_allowed": False,
            "periodic_gpu_process_memory_query": False,
            "n_ctx_formula": "context_tokens + generation_tokens",
            "generation_tokens": args.generation_tokens,
            "performance_timing": "none_diagnostic_prefill_only",
            "kv_query_outside_inference_window": True,
            "kv_measurement_semantics": KV_MEASUREMENT_SEMANTICS,
        },
    )
    manifest_payload = {
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
        expected_manifest = {
            "schema_version": SCHEMA_VERSION,
            "run_id": args.run_id,
            **manifest_payload,
        }
        existing_manifest = json.loads(
            writer.layout.manifest.read_text(encoding="utf-8")
        )
        if existing_config != config.to_dict():
            raise ValueError("--resume 的冻结运行配置与已有产物不一致")
        if existing_manifest != expected_manifest:
            raise ValueError("--resume 的环境、源码或模型 manifest 不一致")
    else:
        writer.initialize(config, manifest_payload)

    records = _read_jsonl(writer.layout.iterations)
    expected_keys = {(context, name) for context, name, _ in plan}
    completed_keys: set[tuple[int, str]] = set()
    for record in records:
        key = (int(record["context_tokens"]), str(record["config"]))
        if key not in expected_keys:
            raise ValueError(f"续跑产物包含矩阵外记录: {key}")
        if key in completed_keys:
            raise ValueError(f"续跑产物包含重复记录: {key}")
        completed_keys.add(key)

    for context_tokens, config_name, ordinal in plan:
        key = (context_tokens, config_name)
        if key in completed_keys:
            print(
                f"[{ordinal}/{expected_records}] resume skip "
                f"C{context_tokens} {config_name}",
                flush=True,
            )
            continue
        print(
            f"[{ordinal}/{expected_records}] run "
            f"C{context_tokens} {config_name}",
            flush=True,
        )
        child_json = (
            writer.layout.root / f"C{context_tokens}_{config_name}_child.json"
        )
        child_log = writer.layout.root / f"C{context_tokens}_{config_name}.log"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child-config",
            config_name,
            "--run-index",
            "1",
            "--model-layers",
            str(model_layers),
            "--child-output-json",
            str(child_json),
            "--model",
            str(model_path),
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
            "--seed",
            str(args.seed),
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
                    stage=f"C{context_tokens}_{config_name}",
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
                    stage=f"C{context_tokens}_{config_name}",
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
                name="llm_kv_calibration_child",
                monotonic_ns=record["child_started_monotonic_ns"],
                wall_time_iso=record["child_started_wall_iso"],
                phase=f"C{context_tokens}/{config_name}",
                attributes={
                    "context_tokens": context_tokens,
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
        "model_metadata": metadata,
        "kv_measurement_semantics": KV_MEASUREMENT_SEMANTICS,
        "by_context_config": {
            str(context_tokens): {
                name: record
                for name in config_names
                for record in records
                if record["context_tokens"] == context_tokens
                and record["config"] == name
            }
            for context_tokens in contexts
        },
        "records": records,
        "evidence_scope": _evidence_scope(contexts, config_names, args.runs),
    }
    writer.write_summary(summary)
    fields = [
        "context_tokens",
        "generation_tokens",
        "expected_state_tokens",
        "config",
        "matrix_execution_ordinal",
        "passed",
        "n_gpu_layers",
        "n_ctx",
        "kv_gpu_buffer_mb",
        "kv_cpu_buffer_mb",
        "compute_gpu_buffer_mb",
        "compute_cpu_buffer_mb",
        "kv_tokens_used",
        "kv_sequence_state_bytes",
        "full_state_serialized_bytes",
        "kv_state_bytes",
        "gpu_delta_mb",
    ]
    with writer.layout.summary_csv.open("w", encoding="utf-8", newline="") as file_obj:
        csv_writer = csv.DictWriter(file_obj, fieldnames=fields)
        csv_writer.writeheader()
        for record in records:
            csv_writer.writerow(
                {
                    "context_tokens": record["context_tokens"],
                    "generation_tokens": record["generation_tokens"],
                    "expected_state_tokens": record["expected_state_tokens"],
                    "config": record["config"],
                    "matrix_execution_ordinal": record["matrix_execution_ordinal"],
                    "passed": record["passed"],
                    "n_gpu_layers": record["n_gpu_layers"],
                    "n_ctx": record["n_ctx"],
                    "kv_gpu_buffer_mb": record.get("kv_gpu_buffer_mb"),
                    "kv_cpu_buffer_mb": record.get("kv_cpu_buffer_mb"),
                    "compute_gpu_buffer_mb": record.get("compute_gpu_buffer_mb"),
                    "compute_cpu_buffer_mb": record.get("compute_cpu_buffer_mb"),
                    "kv_tokens_used": record["kv_state"]["kv_tokens_used"],
                    "kv_sequence_state_bytes": record["kv_state"][
                        "kv_sequence_state_bytes"
                    ],
                    "full_state_serialized_bytes": record["kv_state"][
                        "full_state_serialized_bytes"
                    ],
                    "kv_state_bytes": record["kv_state_bytes"],
                    "gpu_delta_mb": record["memory_after_prefill"][
                        "gpu_card_delta_from_baseline_mb"
                    ],
                }
            )
    return writer.layout.root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="/home/leslie/mvp_core/models/llm/Qwen3.5-4B-Q4_K_M.gguf",
    )
    parser.add_argument("--configs", nargs="+", default=list(FORMAL_CONFIGS))
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--contexts", nargs="+", type=int, default=list(FORMAL_CONTEXTS))
    parser.add_argument("--context-tokens", type=int, default=512)
    parser.add_argument("--generation-tokens", type=int, default=128)
    parser.add_argument("--llm-cores", type=int, default=14)
    parser.add_argument("--system-cores", type=int, default=2)
    parser.add_argument("--sample-interval", type=float, default=0.05)
    parser.add_argument("--gpu-sample-interval", type=float, default=0.25)
    parser.add_argument("--gpu-warmup-timeout", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--child-timeout", type=float, default=1800.0)
    parser.add_argument("--inter-run-cooldown", type=float, default=1.0)
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
            PROJECT_ROOT / "experiment_results" / "elastic_residency" / "kv_calibration"
        ),
    )
    parser.add_argument(
        "--run-id",
        default=f"kv_calibration_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}",
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
    result_root = run_parent(args)
    print(result_root)
    summary = json.loads((result_root / "summary.json").read_text(encoding="utf-8"))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
