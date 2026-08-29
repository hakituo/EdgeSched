#!/usr/bin/env python3
"""Forge b=1 同日复测（C 阶段收尾）：为 H_D 冻结提供当前驱动/服务状态下的真实显存证据。

背景：C2 驻留曲线完成后，H_TTS/H_D 冻结前必须用同日 Forge b=1 复测替换
2026-08-28 batch_scaling 的约 5904 MiB 旧整卡峰值预估（见
docs/ELASTIC_RESIDENCY_EXPERIMENT_EXECUTION_PLAN_20260828.md 阶段 C）。

口径：
- ``global_idle_high_water_mb``：脚本启动、Forge 空闲时的整卡高水位，
  对齐 C2/B2 的空闲基线口径；
- ``forge_resident_baseline_mb``：warmup 触发模型加载并冷却后的常驻整卡占用；
- ``peak_card_used_mb``：各测量轮窗口内的整卡峰值；
- ``m_d_incremental_estimate_mb = peak_max - global_idle_high_water_mb``，
  即 M_D(b=1) 的增量估计，供 analyze_results.py 计算 H_D。

用法（WSL2，Forge 服务需先在 forge-cu128 环境启动）：
  /home/leslie/.venvs/mvp_core-cu128/bin/python \\
      experiments/elastic_residency/forge_b1_reprofile.py \\
      --run-id forge_b1_reprofile_20260829_verified_v1 \\
      --source-git-commit <当前 commit> --source-git-dirty false \\
      --source-git-status-entry-count 0

产物：experiment_results/elastic_residency/forge/<run_id>/。
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.elastic_residency.contracts import (  # noqa: E402
    BackendEvidence,
    ExperimentRunConfig,
)
from experiments.tts.profiling_utils import percentile  # noqa: E402


FORGE_PROMPT = "a serene mountain lake at sunrise, photorealistic, fine details"


def build_txt2img_payload(
    *,
    seed: int,
    width: int,
    height: int,
    steps: int,
    cfg_scale: float,
    sampler: str,
) -> dict[str, Any]:
    """构造与 batch_scaling_experiment 相同口径的 b=1 txt2img payload。"""

    return {
        "prompt": FORGE_PROMPT,
        "negative_prompt": "",
        "width": width,
        "height": height,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "sampler_name": sampler,
        "seed": seed,
        "batch_size": 1,
        "n_iter": 1,
    }


def forge_get(url: str, timeout: float) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def forge_post(url: str, payload: Mapping[str, Any], timeout: float) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def extract_gpu_used_series(
    samples: Sequence[Any],
    start_ns: int | None = None,
    end_ns: int | None = None,
) -> list[float]:
    """从时间线样本提取 [start_ns, end_ns] 窗口内可用的整卡显存序列。"""

    series: list[float] = []
    for sample in samples:
        if not getattr(sample, "gpu_available", False):
            continue
        timestamp = sample.monotonic_ns
        if start_ns is not None and timestamp < start_ns:
            continue
        if end_ns is not None and timestamp > end_ns:
            continue
        series.append(float(sample.gpu_card_used_mb))
    return series


def summarize_gpu_series(series: Sequence[float]) -> Mapping[str, float | int]:
    """对单个窗口的整卡显存序列给出高水位与样本数。"""

    if not series:
        return {"high_water_mb": None, "sample_count": 0}
    return {
        "high_water_mb": max(series),
        "low_water_mb": min(series),
        "sample_count": len(series),
    }


def evaluate_swap_growth(
    swap_values: Sequence[int],
    tolerance_bytes: int,
) -> Mapping[str, Any]:
    """评估时间线 swap 增长是否在容忍范围内。

    Forge 复测必须与 Forge 服务同机运行，WSL2/Forge 带来数百 KB 级的
    系统性 swap 噪声属于正常现象；这里只拒绝实验期间的真实换页增长
    （超过 tolerance_bytes），口径与 A1/C2 的 ``swap==0`` 不同，需在
    manifest 与 summary 中保留完整数值供人工复核。
    """

    if not swap_values:
        return {
            "start_bytes": None,
            "max_bytes": None,
            "growth_bytes": None,
            "within_tolerance": True,
        }
    start = int(swap_values[0])
    maximum = int(max(swap_values))
    growth = max(0, maximum - start)
    return {
        "start_bytes": start,
        "max_bytes": maximum,
        "growth_bytes": growth,
        "within_tolerance": growth <= int(tolerance_bytes),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--forge-url", default="http://127.0.0.1:7860")
    parser.add_argument(
        "--output-root",
        default=str(
            PROJECT_ROOT / "experiment_results" / "elastic_residency" / "forge"
        ),
    )
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--idle-observe-s", type=float, default=3.0)
    parser.add_argument("--cooldown-s", type=float, default=2.0)
    parser.add_argument("--min-baseline-gpu-samples", type=int, default=8)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--sampler", default="Euler")
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--request-timeout", type=float, default=1800.0)
    parser.add_argument("--sample-interval", type=float, default=0.05)
    parser.add_argument("--gpu-sample-interval", type=float, default=0.25)
    parser.add_argument("--gpu-warmup-timeout", type=float, default=5.0)
    parser.add_argument("--system-cores", type=int, default=2)
    parser.add_argument(
        "--max-swap-growth-mb",
        type=float,
        default=4.0,
        help="swap 增长容忍上限（MiB）；Forge/WSL2 系统噪声通常低于 1 MiB",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source-git-commit", default="")
    parser.add_argument("--source-git-dirty", choices=("true", "false"))
    parser.add_argument("--source-git-status-entry-count", type=int)
    parser.add_argument("--source-git-snapshot")
    parser.add_argument("--source-git-snapshot-sha256")
    return parser


def main() -> None:
    # psutil 相关的重依赖延迟到运行时导入，保证 Windows 侧可静态导入本模块做单元测试。
    from experiments.elastic_residency.contracts import (
        ExperimentArtifactWriter,
        ExperimentEvent,
        FailureRecord,
    )
    from experiments.elastic_residency.cpu_topology import (
        CPUTopology,
        PhysicalCoreAllocator,
    )
    from experiments.elastic_residency.resource_sampler import ResourceSampler
    from experiments.elastic_residency.tts_component_offload import _git_source

    args = build_parser().parse_args()
    if args.iterations < 1:
        raise SystemExit("--iterations 必须为正")
    if args.warmup < 0:
        raise SystemExit("--warmup 不能为负")

    writer = ExperimentArtifactWriter(args.output_root, args.run_id)
    source_git, git_complete = _git_source(args, writer.layout.root)

    run_config = ExperimentRunConfig(
        experiment_name="forge_b1_reprofile",
        run_id=args.run_id,
        seed=args.seed,
        sample_interval_s=args.sample_interval,
        gpu_sample_interval_s=args.gpu_sample_interval,
        backends=(
            BackendEvidence(
                component="diffusion",
                backend_kind="real_forge_http_api",
                is_real=True,
            ),
        ),
        parameters={
            "forge_base_url": args.forge_url,
            "iterations": args.iterations,
            "warmup": args.warmup,
            "width": args.width,
            "height": args.height,
            "steps": args.steps,
            "cfg_scale": args.cfg_scale,
            "sampler": args.sampler,
            "seed_base": args.seed_base,
            "idle_observe_s": args.idle_observe_s,
            "cooldown_s": args.cooldown_s,
            "system_physical_cores": args.system_cores,
            "gpu_telemetry_backend": "direct_nvml",
            "gpu_cli_subprocess_allowed": False,
        },
    )

    def _event(name: str, phase: str, attributes: Mapping[str, Any] | None = None) -> None:
        writer.append_event(
            ExperimentEvent(
                name=name,
                monotonic_ns=time.monotonic_ns(),
                wall_time_iso=dt.datetime.now(dt.timezone.utc).isoformat(),
                phase=phase,
                attributes=dict(attributes or {}),
            )
        )

    def _failure(stage: str, error: BaseException | str, *, fatal: bool = True) -> None:
        writer.append_failure(
            FailureRecord(
                stage=stage,
                error_type=type(error).__name__ if isinstance(error, BaseException) else "Error",
                message=str(error),
                monotonic_ns=time.monotonic_ns(),
                fatal=fatal,
            )
        )

    # 1. 检查 Forge 服务与模型信息。
    forge_models_url = args.forge_url.rstrip("/") + "/sdapi/v1/sd-models"
    try:
        models = forge_get(forge_models_url, timeout=10.0)
    except (OSError, urllib.error.URLError) as exc:
        _failure("forge_availability_check", exc)
        writer.write_summary(
            {
                "run_id": args.run_id,
                "passed": False,
                "failure_count": 1,
                "forge_api_available": False,
            }
        )
        print("Forge 服务不可用，请先在 forge-cu128 环境启动 Forge。", file=sys.stderr)
        raise SystemExit(1)
    model_titles = [m.get("title") for m in models if m.get("title")]
    _event("forge_available", "preflight", {"model_count": len(model_titles)})

    manifest = {
        "git": {"source_copy": source_git},
        "git_provenance_complete": git_complete,
        "forge": {
            "base_url": args.forge_url,
            "available_models": model_titles,
        },
        "python": sys.version.split()[0],
        "platform": sys.platform,
    }
    writer.initialize(run_config, manifest)

    # 2. 分配系统核给采样线程，与 A1/C2 的资源采样口径一致。
    topology = CPUTopology.discover()
    if len(topology.physical_cores) < args.system_cores:
        _failure("cpu_topology", RuntimeError("物理核不足以保留系统核心"))
        raise SystemExit(1)
    allocator = PhysicalCoreAllocator(topology)
    system_lease = allocator.reserve_system(args.system_cores)

    sampler = ResourceSampler(
        interval_s=args.sample_interval,
        gpu_interval_s=args.gpu_sample_interval,
        writer=writer,
        thread_affinity=system_lease.affinity,
    )
    try:
        gpu_ready = sampler.start(wait_for_gpu_timeout_s=args.gpu_warmup_timeout)
        if not gpu_ready:
            raise RuntimeError("GPU 探针在超时内没有成功样本")
        _event("sampler_started", "baseline")

        # 3. 全局空闲高水位（Forge 尚未触发本轮生成）。
        idle_window_start_ns = time.monotonic_ns()
        time.sleep(args.idle_observe_s)
        idle_window_end_ns = time.monotonic_ns()
        idle_series = extract_gpu_used_series(
            sampler.samples, idle_window_start_ns, idle_window_end_ns
        )
        idle_summary = summarize_gpu_series(idle_series)
        _event("idle_window_measured", "baseline", dict(idle_summary))

        txt2img_url = args.forge_url.rstrip("/") + "/sdapi/v1/txt2img"

        # 4. warmup：触发 Forge 模型加载，不计入统计。
        for warmup_index in range(args.warmup):
            payload = build_txt2img_payload(
                seed=args.seed_base + warmup_index,
                width=args.width,
                height=args.height,
                steps=args.steps,
                cfg_scale=args.cfg_scale,
                sampler=args.sampler,
            )
            _event("warmup_started", "warmup", {"index": warmup_index + 1})
            forge_post(txt2img_url, payload, timeout=args.request_timeout)
            _event("warmup_finished", "warmup", {"index": warmup_index + 1})

        time.sleep(args.cooldown_s)
        # 常驻基线只取冷却窗口：warmup 期间的显存冲高不应计入 Forge 常驻值。
        cooldown_window_end_ns = time.monotonic_ns()
        cooldown_window_start_ns = cooldown_window_end_ns - int(args.cooldown_s * 1e9)
        cooldown_series = extract_gpu_used_series(
            sampler.samples, cooldown_window_start_ns, cooldown_window_end_ns
        )
        resident_summary = summarize_gpu_series(cooldown_series)
        _event("resident_baseline_measured", "baseline", dict(resident_summary))

        # 5. 正式测量轮。
        iteration_peaks: list[float | None] = []
        iteration_records: list[dict[str, Any]] = []
        for run_index in range(1, args.iterations + 1):
            payload = build_txt2img_payload(
                seed=args.seed_base + args.warmup + run_index,
                width=args.width,
                height=args.height,
                steps=args.steps,
                cfg_scale=args.cfg_scale,
                sampler=args.sampler,
            )
            _event("iteration_started", "measurement", {"run_index": run_index})
            started_ns = time.monotonic_ns()
            started_perf = time.perf_counter()
            response = forge_post(txt2img_url, payload, timeout=args.request_timeout)
            latency_s = time.perf_counter() - started_perf
            finished_ns = time.monotonic_ns()
            images = response.get("images", [])
            peak_series = extract_gpu_used_series(sampler.samples, started_ns, finished_ns)
            peak = summarize_gpu_series(peak_series)
            iteration_peaks.append(peak["high_water_mb"])  # type: ignore[arg-type]
            record = {
                "run_index": run_index,
                "latency_s": latency_s,
                "image_count": len(images),
                "window_start_monotonic_ns": started_ns,
                "window_end_monotonic_ns": finished_ns,
                "gpu_peak_high_water_mb": peak["high_water_mb"],
                "gpu_peak_sample_count": peak["sample_count"],
            }
            iteration_records.append(record)
            writer.append_iteration(record)
            _event("iteration_finished", "measurement", dict(record))

        time.sleep(args.cooldown_s)
    finally:
        sampler_stats = sampler.stop()
        system_lease.release()

    allocator_snapshot = allocator.snapshot()

    # 6. 汇总与门禁。
    valid_peaks = [peak for peak in iteration_peaks if peak is not None]
    idle_high = idle_summary["high_water_mb"]
    resident_high = resident_summary["high_water_mb"]
    peak_max = max(valid_peaks) if valid_peaks else None
    peak_mean = statistics.mean(valid_peaks) if valid_peaks else None
    latencies = [record["latency_s"] for record in iteration_records]

    swap_evaluation = evaluate_swap_growth(
        [sample.system_swap_used_bytes for sample in sampler.samples],
        int(args.max_swap_growth_mb * 1024 * 1024),
    )

    checks = {
        "forge_api_available": True,
        "git_provenance_complete": git_complete,
        "gpu_telemetry_is_direct_nvml": (
            sampler_stats["gpu_telemetry_backend"] == "direct_nvml"
            and not sampler_stats["gpu_cli_subprocess_used"]
        ),
        "idle_baseline_samples_sufficient": (
            idle_summary["sample_count"] >= args.min_baseline_gpu_samples
        ),
        "resident_baseline_samples_sufficient": (
            resident_summary["sample_count"] >= args.min_baseline_gpu_samples
        ),
        "iterations_complete": len(iteration_records) == args.iterations,
        "every_iteration_has_gpu_peak": all(
            record["gpu_peak_high_water_mb"] is not None for record in iteration_records
        ),
        "single_image_per_iteration": all(
            record["image_count"] == 1 for record in iteration_records
        ),
        "swap_growth_within_tolerance": bool(
            swap_evaluation["within_tolerance"]
        ),
        "all_leases_released": allocator_snapshot["active_lease_count"] == 0,
        "all_physical_cores_returned": (
            allocator_snapshot["free_physical_core_count"]
            == allocator_snapshot["physical_core_count"]
        ),
    }
    m_d_estimate = (
        peak_max - idle_high if peak_max is not None and idle_high is not None else None
    )

    summary = {
        "run_id": args.run_id,
        "passed": all(checks.values()),
        "failure_count": 0,
        "acceptance_checks": checks,
        "global_idle_high_water_mb": idle_high,
        "forge_resident_baseline_mb": resident_high,
        "peak_card_used_mb": {
            "max": peak_max,
            "mean": peak_mean,
        },
        "m_d_incremental_estimate_mb": m_d_estimate,
        "m_d_estimate_method": "peak_max_minus_global_idle_high_water",
        "iterations": {
            "expected": args.iterations,
            "actual": len(iteration_records),
            "latency_s": {
                "mean": statistics.mean(latencies) if latencies else None,
                "min": min(latencies) if latencies else None,
                "max": max(latencies) if latencies else None,
                "p95": percentile(latencies, 0.95) if latencies else None,
            },
        },
        "swap": {
            **dict(swap_evaluation),
            "tolerance_bytes": int(args.max_swap_growth_mb * 1024 * 1024),
            "policy": "growth_within_tolerance",
        },
        "records": iteration_records,
        "resource_sampler": sampler_stats,
        "allocator_after_release": allocator_snapshot,
        "evidence_scope": "forge_b1_reprofile",
    }
    writer.write_summary(summary)
    writer.write_summary_csv(
        {
            "run_id": args.run_id,
            "passed": summary["passed"],
            "global_idle_high_water_mb": idle_high,
            "forge_resident_baseline_mb": resident_high,
            "peak_card_used_mb_max": peak_max,
            "m_d_incremental_estimate_mb": m_d_estimate,
            "latency_mean_s": summary["iterations"]["latency_s"]["mean"],
        }
    )

    print(json.dumps({k: summary[k] for k in (
        "run_id", "passed", "global_idle_high_water_mb",
        "forge_resident_baseline_mb", "peak_card_used_mb",
        "m_d_incremental_estimate_mb",
    )}, ensure_ascii=False, indent=2))
    if not summary["passed"]:
        failed = [name for name, value in checks.items() if not value]
        print(f"门禁未通过: {failed}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
