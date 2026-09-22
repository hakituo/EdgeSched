"""LLM+Forge 共置矩阵父进程编排。"""
from __future__ import annotations

import datetime as dt
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from experiments.elastic_residency.contracts import (
    SCHEMA_VERSION,
    BackendEvidence,
    ExperimentArtifactWriter,
    ExperimentEvent,
    ExperimentRunConfig,
    FailureRecord,
    TimelineSample,
)
from experiments.elastic_residency.forge_b1_reprofile import (
    build_txt2img_payload,
    forge_get,
    forge_post,
)
from experiments.elastic_residency.host_memory import collect_host_memory_state
from experiments.elastic_residency.llm_forge_plan import (
    LlmForgeCandidate,
    build_execution_plan,
    candidate_profile,
    select_candidates,
)
from experiments.elastic_residency.llm_residency_curve import (
    _discover_model_metadata,
    resolve_config,
)
from experiments.elastic_residency.tts_component_offload import _git_source
from experiments.tts.profiling_utils import sha256_file

def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

def _validate_forge_profile(
    screen: dict[str, Any], forge: dict[str, Any]
) -> dict[str, float | str]:
    checks = forge.get("acceptance_checks", {})
    if not forge.get("passed") or not checks or not all(checks.values()):
        raise ValueError("Forge 隔离校准未通过全部门禁")
    sampler = forge.get("resource_sampler", {})
    if sampler.get("gpu_telemetry_backend") != "direct_nvml" or sampler.get(
        "gpu_cli_subprocess_used"
    ):
        raise ValueError("Forge 隔离校准不是纯 direct NVML")
    expected = screen["forge"]
    if expected.get("run_id") != forge.get("run_id"):
        raise ValueError("screen 与 Forge summary 的 run id 不一致")
    for screen_value, forge_value, label in (
        (expected["m_d_incremental_estimate_mb"], forge["m_d_incremental_estimate_mb"], "M_D"),
        (expected["global_idle_high_water_mb"], forge["global_idle_high_water_mb"], "idle"),
    ):
        if abs(float(screen_value) - float(forge_value)) > 1e-6:
            raise ValueError(f"screen 与 Forge summary 的 {label} 不一致")
    return {
        "run_id": str(forge["run_id"]),
        "global_idle_high_water_mb": float(forge["global_idle_high_water_mb"]),
        "forge_resident_baseline_mb": float(forge["forge_resident_baseline_mb"]),
        "m_d_incremental_estimate_mb": float(forge["m_d_incremental_estimate_mb"]),
    }


def _failure(writer: ExperimentArtifactWriter, stage: str, message: str) -> None:
    writer.append_failure(
        FailureRecord(
            stage=stage,
            error_type="MeasurementFailure",
            message=message,
            monotonic_ns=time.monotonic_ns(),
        )
    )


def _child_command(
    args: Any,
    candidate: LlmForgeCandidate,
    profile: dict[str, float],
    run_index: int,
    ordinal: int,
    model_layers: int,
    child_json: Path,
    child_image: Path,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).with_name("llm_forge_colocated.py")),
        "--child",
        "--llm-config", candidate.llm_config,
        "--context-tokens", str(candidate.context_tokens),
        "--run-index", str(run_index),
        "--model-layers", str(model_layers),
        "--llm-model", str(Path(args.llm_model).expanduser().resolve()),
        "--child-output-json", str(child_json),
        "--child-output-image", str(child_image),
        "--forge-url", args.forge_url,
        "--forge-seed", str(args.seed_base + args.warmup + ordinal),
        "--width", str(args.width), "--height", str(args.height),
        "--steps", str(args.steps), "--cfg-scale", str(args.cfg_scale),
        "--forge-sampler", args.forge_sampler,
        "--request-timeout", str(args.request_timeout),
        "--generation-tokens", str(args.generation_tokens),
        "--llm-cores", str(args.llm_cores),
        "--system-cores", str(args.system_cores),
        "--sample-interval", str(args.sample_interval),
        "--gpu-sample-interval", str(args.gpu_sample_interval),
        "--gpu-warmup-timeout", str(args.gpu_warmup_timeout),
        "--max-gpu-sample-age-ms", str(args.max_gpu_sample_age_ms),
        "--max-swap-growth-mb", str(args.max_swap_growth_mb),
        "--global-idle-reference-mb", str(profile["global_idle_reference_mb"]),
        "--safety-margin-mb", str(profile["safety_margin_mb"]),
        "--llm-isolated-increment-mb", str(profile["llm_isolated_increment_mb"]),
        "--forge-isolated-increment-mb", str(profile["forge_isolated_increment_mb"]),
        "--expected-gpu-total-mb", str(profile["gpu_total_mb"]),
        "--expected-forge-resident-baseline-mb", str(args.forge_resident_baseline_mb),
        "--max-forge-resident-drift-mb", str(args.max_forge_resident_drift_mb),
        "--seed", str(args.seed + ordinal - 1),
    ]


def run_parent(args: Any) -> Path:
    model_path = Path(args.llm_model).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"找不到 GGUF: {model_path}")
    screen_path = Path(args.screen_analysis).expanduser().resolve()
    forge_path = Path(args.forge_summary).expanduser().resolve()
    screen, forge = _load_json(screen_path), _load_json(forge_path)
    forge_profile = _validate_forge_profile(screen, forge)
    args.forge_resident_baseline_mb = forge_profile["forge_resident_baseline_mb"]
    metadata = _discover_model_metadata(model_path)
    model_layers = int(metadata["block_count"])
    candidates, boundaries, exploratory = select_candidates(
        screen, args.candidates,
        allow_exploratory_fallback=args.allow_exploratory_fallback,
    )
    expected_candidates = set(boundaries)
    for candidate in candidates:
        resolve_config(candidate.llm_config, model_layers)
        candidate_profile(
            screen, candidate,
            allow_lower_fallback=args.allow_exploratory_fallback,
        )
    plan = build_execution_plan(candidates, runs=args.runs)
    expected_records = len(plan)

    models = forge_get(args.forge_url.rstrip("/") + "/sdapi/v1/sd-models", 10.0)
    options = forge_get(args.forge_url.rstrip("/") + "/sdapi/v1/options", 10.0)
    if not isinstance(models, list) or not models or not options.get("sd_model_checkpoint"):
        raise ValueError("Forge preflight 未返回有效模型列表或当前 checkpoint")
    # 论文环境约束是 WSL2 的有效内存上限而非宿主机容量；这两项必须随运行落盘，
    # 否则复现差异分不清来自代码还是来自内存上限与 swap 状态。
    host_memory_state = collect_host_memory_state()
    output_root = Path(args.output_root).expanduser().resolve()
    existed = (output_root / args.run_id).exists()
    writer = ExperimentArtifactWriter(output_root, args.run_id, allow_existing=args.resume)
    source_git, git_complete = _git_source(args, writer.layout.root)
    llm_sha = sha256_file(model_path)
    config = ExperimentRunConfig(
        experiment_name="llm_forge_colocated",
        run_id=args.run_id,
        seed=args.seed,
        sample_interval_s=args.sample_interval,
        gpu_sample_interval_s=args.gpu_sample_interval,
        backends=(
            BackendEvidence("llm", "real_llama_cpp_gguf", True, str(model_path), llm_sha),
            BackendEvidence("diffusion", "real_forge_http_api", True),
        ),
        parameters={
            "candidates": [item.label for item in candidates],
            "candidate_selection": {
                "screened_h_d_boundaries": [item.label for item in boundaries],
                "exploratory_lower_residency_allowed": args.allow_exploratory_fallback,
                "exploratory_candidates": [item.label for item in exploratory],
            },
            "runs": args.runs,
            "warmup": args.warmup,
            "generation_tokens": args.generation_tokens,
            "expected_records": expected_records,
            "screen_analysis": str(screen_path),
            "forge_summary": str(forge_path),
            "forge_profile": forge_profile,
            "forge_request": {
                "width": args.width, "height": args.height, "steps": args.steps,
                "cfg_scale": args.cfg_scale, "sampler": args.forge_sampler,
                "seed_base": args.seed_base, "request_timeout_s": args.request_timeout,
            },
            "llm_physical_cores": args.llm_cores,
            "system_physical_cores": args.system_cores,
            "inter_run_cooldown_s": args.inter_run_cooldown,
            "child_timeout_s": args.child_timeout,
            "continue_on_failure": args.continue_on_failure,
            "capacity_formula": "(P_colocated-B_idle_reference)+G_safe<=gpu_total",
            "gpu_telemetry_backend": "direct_nvml",
            "gpu_cli_subprocess_allowed": False,
            "periodic_gpu_process_memory_query": False,
            "max_gpu_sample_age_ms": args.max_gpu_sample_age_ms,
            "max_swap_growth_mb": args.max_swap_growth_mb,
            "max_forge_resident_drift_mb": args.max_forge_resident_drift_mb,
            # swappiness 是主机内存回收策略的对照变量，必须进配置而不是藏在环境里。
            "host_memory_condition": host_memory_state,
        },
    )
    manifest = {
        "python": platform.python_version(), "platform": platform.platform(),
        "git": {"source_copy": source_git}, "git_provenance_complete": git_complete,
        "llm_model": {"path": str(model_path), "sha256": llm_sha,
                      "size_bytes": model_path.stat().st_size, "metadata": metadata},
        "forge": {"base_url": args.forge_url, "active_checkpoint": options.get("sd_model_checkpoint"),
                  "available_models": [item.get("title") for item in models]},
        "source_artifacts": {"screen_sha256": sha256_file(screen_path),
                             "forge_summary_sha256": sha256_file(forge_path)},
        "host_memory": host_memory_state,
    }
    expected_manifest = {"schema_version": SCHEMA_VERSION, "run_id": args.run_id, **manifest}
    if existed:
        if _load_json(writer.layout.run_config) != config.to_dict() or _load_json(writer.layout.manifest) != expected_manifest:
            raise ValueError("--resume 的配置、源码、模型或输入 artifact 不一致")
    else:
        writer.initialize(config, manifest)

    try:
        for index in range(args.warmup):
            payload = build_txt2img_payload(
                seed=args.seed_base + index, width=args.width, height=args.height,
                steps=args.steps, cfg_scale=args.cfg_scale, sampler=args.forge_sampler,
            )
            forge_post(args.forge_url.rstrip("/") + "/sdapi/v1/txt2img", payload, args.request_timeout)
            writer.append_event(ExperimentEvent("forge_warmup", time.monotonic_ns(),
                dt.datetime.now(dt.timezone.utc).isoformat(), "preflight", {"index": index + 1}))
        if args.warmup and args.inter_run_cooldown > 0:
            time.sleep(args.inter_run_cooldown)
    except Exception as exc:
        _failure(writer, "forge_warmup", f"{type(exc).__name__}: {exc}")
        writer.write_summary({"passed": False, "failure_count": 1, "records": []})
        raise

    records = _read_jsonl(writer.layout.iterations)
    completed = {(int(r["context_tokens"]), str(r["llm_config"]), int(r["run_index"])) for r in records}
    expected_keys = {(c, llm, run) for c, llm, run, _ in plan}
    if not completed.issubset(expected_keys) or len(completed) != len(records):
        raise ValueError("续跑记录包含矩阵外或重复 key")
    for context, llm_config, run_index, ordinal in plan:
        key = (context, llm_config, run_index)
        if key in completed:
            continue
        candidate = LlmForgeCandidate(context, llm_config)
        label = f"C{context}_{llm_config}_Forge_r{run_index}"
        child_json = writer.layout.root / f"{label}_child.json"
        child_image = writer.layout.root / f"{label}.png"
        print(f"[{ordinal}/{expected_records}] run {label}", flush=True)
        command = _child_command(args, candidate, candidate_profile(
            screen, candidate,
            allow_lower_fallback=args.allow_exploratory_fallback),
                                 run_index, ordinal, model_layers, child_json, child_image)
        try:
            result = subprocess.run(command, capture_output=True, text=True,
                                    timeout=args.child_timeout, check=False)
            (writer.layout.root / f"{label}.log").write_text(
                result.stdout + "\n[stderr]\n" + result.stderr, encoding="utf-8")
        except subprocess.TimeoutExpired as exc:
            (writer.layout.root / f"{label}.log").write_text(
                f"{exc.stdout or ''}\n[stderr]\n{exc.stderr or ''}", encoding="utf-8")
            _failure(writer, label, f"ChildProcessTimeout: {exc}")
            if not args.continue_on_failure:
                break
            continue
        if result.returncode != 0 or not child_json.is_file():
            _failure(writer, label, f"ChildProcessFailure: returncode={result.returncode}")
            if not args.continue_on_failure:
                break
            continue
        record = _load_json(child_json)
        record["matrix_execution_ordinal"] = ordinal
        for sample in record.pop("timeline", []):
            writer.append_timeline(TimelineSample(**sample))
        writer.append_iteration(record)
        writer.append_event(ExperimentEvent("llm_forge_colocated_child",
            record["child_started_monotonic_ns"], record["child_started_wall_iso"],
            f"C{context}/{llm_config}+Forge", {"run_index": run_index, "ordinal": ordinal}))
        records.append(record)
        completed.add(key)
        print(f"[{ordinal}/{expected_records}] complete passed={record['passed']}", flush=True)
        failed = [name for name, passed in record["acceptance_checks"].items() if not passed]
        if failed:
            _failure(writer, label, "acceptance: " + ",".join(failed))
        if (failed or not record["passed"]) and not args.continue_on_failure:
            break
        if args.inter_run_cooldown > 0:
            time.sleep(args.inter_run_cooldown)

    failure_count = len(_read_jsonl(writer.layout.failures))
    passed = git_complete and len(records) == expected_records and failure_count == 0 and all(r["passed"] for r in records)
    formal = not exploratory and set(candidates) == expected_candidates and args.runs >= 5
    scope = ("llm_forge_colocated_exploratory_fallback"
             if exploratory else "llm_forge_colocated_formal_repeated_matrix"
             if formal else "llm_forge_colocated_smoke_or_pilot")
    summary = {"passed": passed, "expected_records": expected_records,
               "actual_records": len(records), "failure_count": failure_count,
               "git_provenance_complete": git_complete, "records": records,
               "candidate_selection": config.parameters["candidate_selection"],
               "evidence_scope": scope}
    writer.write_summary(summary)
    writer.write_summary_csv({"run_id": args.run_id, "passed": passed,
        "expected_records": expected_records, "actual_records": len(records),
        "failure_count": failure_count, "evidence_scope": summary["evidence_scope"]})
    return writer.layout.root
