"""Qwen3-TTS Base 的 GPU 中文句长 profiling。"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from domain.models.tts import (  # noqa: E402
    TTSChunkRequest,
    TTSChunkStatus,
    TTSOutputMode,
    VoiceCloneProfile,
)
from experiments.framework.reproducibility import (  # noqa: E402
    ReproducibilityManager,
)
from experiments.tts.profiling_utils import (  # noqa: E402
    collect_runtime_platform,
    linear_fit,
    model_tree_hash,
    percentile,
    sha256_file,
)
from services.tts.qwen3_tts_worker import Qwen3TTSWorker  # noqa: E402
from services.tts.worker_protocol import TTSWorkerConfig  # noqa: E402


TEXTS: dict[str, tuple[str, ...]] = {
    "short": (
        "今天的阳光很好，我们慢慢把事情做好。",
        "我已经准备好了，现在可以开始测试语音。",
        "别着急，先喝口水，再继续处理后面的任务。",
        "窗外有一点风，房间里听起来很安静。",
        "这是一段短句，用来观察语音生成速度。",
    ),
    "medium": (
        "今天下午我们会依次检查模型加载、语音合成和文件写入，确保每个环节都有准确而完整的记录。",
        "如果设备资源突然变得紧张，调度器应该根据实时负载选择合适路径，而不是永远固定使用同一种配置。",
        "为了判断语音会不会断流，我们既要记录合成耗时，也要对照生成音频的实际播放时长和运行抖动。",
        "这轮实验保持模型、音色和采样参数不变，只改变中文句子的长度，从而观察计算成本怎样随输入增长。",
        "每个测试句子都会保存独立音频和结构化指标，后续可以从原始结果重新计算平均值、尾延迟和峰值显存。",
    ),
    "long": (
        "当系统同时处理本地语言模型、语音合成和图像生成任务时，单独模型的平均速度并不能直接代表用户体验。我们还需要关注第一段完整语音何时准备好、后续句子能否持续供应，以及设备切换是否带来额外停顿。",
        "本次句长实验会让模型在显卡上保持常驻，先完成一次不计入统计的预热，再依次运行短句、中句和长句。所有文本使用同一音色画像和固定随机种子，并保存每次合成耗时、音频时长、实时率与峰值显存。",
        "如果长句的实时率明显高于短句，说明生成成本会随文本长度快速增加，调度器就需要限制单个语音块的最大长度。反过来，如果不同句长的实时率比较稳定，我们可以把句子切分策略主要用于控制首块等待和播放缓冲。",
        "评估结果必须能够从汇总数据追溯到每一次真实运行，因此除了平均值以外，还会保留轮次、文本编号、字符数量、输出文件哈希、工作进程编号和设备信息。任何失败、乱序或者后端回退都必须显式记录，不能用模拟结果补齐。",
        "完成这一步以后，我们才能决定是否值得继续研究语言模型与语音模型的设备放置。如果显卡上的实时率仍然始终大于一，那么仅仅调整 CPU 核心数量无法满足连续播放约束，应优先研究真正的音频内部流式或更快的推理后端。",
    ),
}

EXPECTED_LENGTHS = {"short": (15, 25), "medium": (40, 60), "long": (80, 120)}
WARMUP_TEXT = "这是一次不计入统计的预热，用来稳定模型执行路径和显存分配。"


def _validate_text_lengths() -> None:
    for tier, texts in TEXTS.items():
        lower, upper = EXPECTED_LENGTHS[tier]
        if len(texts) != 5:
            raise ValueError(f"{tier} 必须恰好包含 5 条文本")
        for index, text in enumerate(texts):
            if not lower <= len(text) <= upper:
                raise ValueError(
                    f"{tier}[{index}] 长度 {len(text)} 不在 [{lower}, {upper}]"
                )


def _tier_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [record for record in records if record["success"]]
    metrics = (
        "synthesis_s",
        "audio_duration_s",
        "rtf",
        "first_chunk_ready_s",
        "peak_vram_allocated_mb",
        "peak_rss_mb",
    )
    summary: dict[str, Any] = {
        "samples": len(records),
        "successes": len(successful),
        "failures": len(records) - len(successful),
        "order_ok": all(record["order_ok"] for record in records),
        "fallback_occurred": any(record["fallback_occurred"] for record in records),
        "mean_text_chars": (
            statistics.mean(record["text_chars"] for record in records)
            if records
            else 0.0
        ),
        "rtf_below_one_count": sum(record["rtf"] < 1.0 for record in successful),
    }
    for metric in metrics:
        values = [record[metric] for record in successful]
        summary[metric] = {
            "mean": statistics.mean(values) if values else 0.0,
            "median": statistics.median(values) if values else 0.0,
            "p95": percentile(values, 0.95),
            "min": min(values) if values else 0.0,
            "max": max(values) if values else 0.0,
        }
    return summary


def _write_summary_csv(path: Path, summary: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "length_tier",
                "samples",
                "mean_text_chars",
                "successes",
                "order_ok",
                "mean_synthesis_s",
                "p95_synthesis_s",
                "mean_audio_duration_s",
                "mean_rtf",
                "p95_rtf",
                "rtf_below_one_count",
                "mean_first_chunk_ready_s",
                "p95_first_chunk_ready_s",
                "peak_vram_mb",
            ]
        )
        for tier in ("short", "medium", "long"):
            item = summary[tier]
            writer.writerow(
                [
                    tier,
                    item["samples"],
                    item["mean_text_chars"],
                    item["successes"],
                    item["order_ok"],
                    item["synthesis_s"]["mean"],
                    item["synthesis_s"]["p95"],
                    item["audio_duration_s"]["mean"],
                    item["rtf"]["mean"],
                    item["rtf"]["p95"],
                    item["rtf_below_one_count"],
                    item["first_chunk_ready_s"]["mean"],
                    item["first_chunk_ready_s"]["p95"],
                    item["peak_vram_allocated_mb"]["max"],
                ]
            )


def _write_chart(path: Path, summary: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    tiers = ("short", "medium", "long")
    xs = [summary[tier]["mean_text_chars"] for tier in tiers]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    chart_specs = (
        (axes[0, 0], "rtf", "RTF", True),
        (axes[0, 1], "first_chunk_ready_s", "First chunk ready (s)", False),
        (axes[1, 0], "synthesis_s", "Synthesis time (s)", False),
        (axes[1, 1], "peak_vram_allocated_mb", "Peak VRAM allocated (MiB)", False),
    )
    for axis, metric, ylabel, draw_constraint in chart_specs:
        means = [summary[tier][metric]["mean"] for tier in tiers]
        p95s = [summary[tier][metric]["p95"] for tier in tiers]
        axis.plot(xs, means, marker="o", label="mean")
        axis.plot(xs, p95s, marker="s", linestyle="--", label="p95")
        if draw_constraint:
            axis.axhline(1.0, color="red", linestyle=":", label="RTF=1")
        axis.set_xlabel("Chinese text length (characters)")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle("Qwen3-TTS Base GPU sentence-length profile")
    fig.savefig(path, dpi=180)
    plt.close(fig)


async def run_profile(args: argparse.Namespace) -> Path:
    _validate_text_lengths()
    if args.device != "cuda:0":
        raise ValueError("GPU 句长 profiling 固定使用 --device cuda:0")
    if args.rounds < 1:
        raise ValueError("--rounds 必须大于 0")

    model_path = Path(args.model).expanduser().resolve()
    ref_audio_path = Path(args.ref_audio).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"找不到模型: {model_path}")
    if not ref_audio_path.is_file():
        raise FileNotFoundError(f"找不到参考音频: {ref_audio_path}")
    if not args.x_vector_only and not args.ref_text.strip():
        raise ValueError("默认 ICL 模式必须提供参考音频准确转写 --ref-text")

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (Path(args.output_root) / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    git_root = Path(args.git_root).expanduser().resolve() if args.git_root else None
    runtime_platform = collect_runtime_platform(git_root)
    model_hash_started = time.perf_counter()
    model_sha256, model_files = model_tree_hash(model_path)
    model_hash_s = time.perf_counter() - model_hash_started
    ref_audio_sha256 = sha256_file(ref_audio_path)

    profile = VoiceCloneProfile(
        profile_id=args.profile_id,
        ref_audio_path=str(ref_audio_path),
        ref_text=args.ref_text,
        language=args.language,
        x_vector_only_mode=args.x_vector_only,
        audio_sha256=ref_audio_sha256,
    )
    config = TTSWorkerConfig(
        model_path=str(model_path),
        device=args.device,
        voice_profiles=(profile,),
        output_dir=str(run_dir),
        dtype=args.dtype,
        attention_backend=args.attention_backend,
        backend_kind="real_qwen3_tts",
        generation_kwargs={
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
            "repetition_penalty": args.repetition_penalty,
        },
    )
    frozen_config = {
        "run_id": run_id,
        "device": args.device,
        "dtype": args.dtype,
        "attention_backend": args.attention_backend,
        "backend_kind": "real_qwen3_tts",
        "fallback_allowed": False,
        "voice_clone_mode": "x_vector_only_ablation" if args.x_vector_only else "icl",
        "language": args.language,
        "rounds": args.rounds,
        "texts_per_tier": 5,
        "measured_chunks": args.rounds * 15,
        "warmup_chunks": 1,
        "true_audio_streaming": False,
        "metric_name": "first_chunk_ready_s",
        "generation_kwargs": config.generation_kwargs,
        "runtime_platform": runtime_platform,
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(frozen_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    worker = Qwen3TTSWorker(config)
    records: list[dict[str, Any]] = []
    warmup_record: dict[str, Any] = {}
    health = None
    fatal_error = ""
    experiment_started = time.perf_counter()
    try:
        health = await worker.start()
        warmup_request = TTSChunkRequest(
            request_id=f"{run_id}_warmup",
            request_epoch=0,
            chunk_id=0,
            raw_text=WARMUP_TEXT,
            synthesis_text=WARMUP_TEXT,
            language=args.language,
            voice_profile_id=args.profile_id,
            output_mode=TTSOutputMode.BYTES,
            seed=args.seed,
        )
        warmup_result = await worker.synthesize(warmup_request)
        warmup_record = warmup_result.to_record()

        chunk_id = 0
        tier_names = ["short", "medium", "long"]
        for round_index in range(args.rounds):
            tier_order = tier_names[round_index % 3 :] + tier_names[: round_index % 3]
            for tier in tier_order:
                texts = list(TEXTS[tier])
                text_order = texts[round_index % 5 :] + texts[: round_index % 5]
                for text in text_order:
                    text_index = TEXTS[tier].index(text)
                    request = TTSChunkRequest(
                        request_id=run_id,
                        request_epoch=0,
                        chunk_id=chunk_id,
                        raw_text=text,
                        synthesis_text=text,
                        language=args.language,
                        voice_profile_id=args.profile_id,
                        output_mode=TTSOutputMode.FILE,
                        seed=args.seed + 1 + chunk_id,
                    )
                    try:
                        result = await worker.synthesize(request)
                        record = result.to_record()
                        record.update(
                            {
                                "round": round_index,
                                "length_tier": tier,
                                "text_id": f"{tier}_{text_index}",
                                "text_chars": len(text),
                                "success": result.status == TTSChunkStatus.COMPLETED
                                and bool(result.wav_path)
                                and Path(result.wav_path).is_file(),
                                "order_ok": result.chunk_id == chunk_id,
                                "fallback_occurred": result.backend_kind
                                != "real_qwen3_tts",
                            }
                        )
                    except Exception as exc:
                        record = {
                            "request_id": run_id,
                            "request_epoch": 0,
                            "chunk_id": chunk_id,
                            "round": round_index,
                            "length_tier": tier,
                            "text_id": f"{tier}_{text_index}",
                            "text_chars": len(text),
                            "raw_text": text,
                            "synthesis_text": text,
                            "success": False,
                            "order_ok": False,
                            "fallback_occurred": False,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                        fatal_error = record["error"]
                    records.append(record)
                    chunk_id += 1
                    if fatal_error:
                        break
                if fatal_error:
                    break
            if fatal_error:
                break
    except Exception as exc:
        fatal_error = f"{type(exc).__name__}: {exc}"
    finally:
        await worker.stop()
    total_s = time.perf_counter() - experiment_started

    with (run_dir / "iterations.jsonl").open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")

    successful_records = [record for record in records if record.get("success")]
    summary = {
        tier: _tier_summary(
            [record for record in records if record.get("length_tier") == tier]
        )
        for tier in ("short", "medium", "long")
    }
    profile_models = {
        "scope": "gpu_sentence_length_only",
        "synthesis_time_model": linear_fit(
            (record["text_chars"] for record in successful_records),
            (record["synthesis_s"] for record in successful_records),
        )
        if successful_records
        else {},
        "rtf_model": linear_fit(
            (record["text_chars"] for record in successful_records),
            (record["rtf"] for record in successful_records),
        )
        if successful_records
        else {},
        "vram_model": linear_fit(
            (record["text_chars"] for record in successful_records),
            (record["peak_vram_allocated_mb"] for record in successful_records),
        )
        if successful_records
        else {},
    }
    (run_dir / "profile_models.json").write_text(
        json.dumps(profile_models, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_summary_csv(run_dir / "summary.csv", summary)
    if successful_records:
        _write_chart(run_dir / "gpu_sentence_length_profile.png", summary)

    fallback_occurred = any(record.get("fallback_occurred") for record in records)
    order_ok = (
        [record["chunk_id"] for record in records]
        == list(range(len(records)))
        and all(record.get("order_ok") for record in records)
    )
    manifest_config = {
        **frozen_config,
        "tts_model_path": str(model_path),
        "tts_model_tree_sha256": model_sha256,
        "tts_model_files": model_files,
        "model_hash_s": model_hash_s,
        "reference_audio_path": str(ref_audio_path),
        "reference_audio_sha256": ref_audio_sha256,
        "reference_text_supplied": bool(args.ref_text.strip()),
        "text_sets": TEXTS,
        "expected_length_ranges": EXPECTED_LENGTHS,
    }
    reproducibility = ReproducibilityManager(seed=args.seed)
    manifest = reproducibility.create_manifest(
        manifest_config, "qwen3_tts_gpu_sentence_length_profile"
    )
    manifest.update(
        {
            "run_id": run_id,
            "host": platform.node(),
            "total_s": total_s,
            "success": not fatal_error
            and len(successful_records) == args.rounds * 15
            and not fallback_occurred
            and order_ok,
            "fatal_error": fatal_error,
            "fallback_occurred": fallback_occurred,
            "output_order_ok": order_ok,
            "worker_start_health": health.to_dict() if health else {},
            "warmup": warmup_record,
            "summary": summary,
            "profile_models": profile_models,
            "all_measured_rtf_below_one": bool(successful_records)
            and all(record["rtf"] < 1.0 for record in successful_records),
        }
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "success": manifest["success"],
                "fallback_occurred": fallback_occurred,
                "output_order_ok": order_ok,
                "summary": summary,
            },
            ensure_ascii=False,
        )
    )
    if not manifest["success"]:
        raise RuntimeError(f"GPU 句长 profiling 未通过: {fatal_error or '验收失败'}")
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=os.environ.get(
            "XIAOYOU_BENCH_TTS_PATH",
            "/home/leslie/mvp_core/models/Qwen3-TTS-12Hz-0.6B-Base",
        ),
    )
    parser.add_argument(
        "--ref-audio",
        default=os.environ.get(
            "XIAOYOU_TTS_REF_AUDIO_PATH",
            "/home/leslie/mvp_core/models/tts_refs/ref_calm.wav",
        ),
    )
    parser.add_argument("--ref-text", default="")
    parser.add_argument("--x-vector-only", action="store_true")
    parser.add_argument("--profile-id", default="calm")
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--device", choices=("cuda:0",), default="cuda:0")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--run-id")
    parser.add_argument(
        "--git-root",
        default=os.environ.get("XIAOYOU_GIT_ROOT", ""),
    )
    parser.add_argument(
        "--output-root",
        default=str(
            PROJECT_ROOT
            / "experiment_results"
            / "tts"
            / "profiling"
            / "gpu_sentence_length"
        ),
    )
    return parser


def main() -> None:
    asyncio.run(run_profile(build_parser().parse_args()))


if __name__ == "__main__":
    main()
