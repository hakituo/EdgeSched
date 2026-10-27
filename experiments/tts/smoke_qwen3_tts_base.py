"""Qwen3-TTS Base 独立 CPU/GPU 冒烟与重复块画像实验。"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from domain.models.tts import (  # noqa: E402
    TTSChunkRequest,
    TTSOutputMode,
    VoiceCloneProfile,
)
from experiments.framework.reproducibility import (  # noqa: E402
    ReproducibilityManager,
)
from experiments.tts.profiling_utils import (  # noqa: E402
    collect_runtime_platform,
    model_tree_hash,
    select_physical_cpu_affinity,
    sha256_file,
)
from services.tts.qwen3_tts_worker import Qwen3TTSWorker  # noqa: E402
from services.tts.worker_protocol import TTSWorkerConfig  # noqa: E402


DEFAULT_TEXTS = (
    "你好，这是第一段独立语音测试。",
    "第二段用于确认模型和音色画像不会重复加载。",
    "第三段用于检查输出顺序和逐块性能记录。",
)


async def run_smoke(args: argparse.Namespace) -> Path:
    model_path = Path(args.model).expanduser().resolve()
    ref_audio_path = Path(args.ref_audio).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"找不到模型: {model_path}")
    if not ref_audio_path.is_file():
        raise FileNotFoundError(f"找不到参考音频: {ref_audio_path}")
    if not args.x_vector_only and not args.ref_text.strip():
        raise ValueError("默认 ICL 模式必须通过 --ref-text 提供参考音频的准确转写")

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (Path(args.output_root) / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
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
    affinity = tuple(args.affinity or ())
    if args.device == "cpu" and args.cpu_threads > 0 and not affinity:
        affinity = select_physical_cpu_affinity(args.cpu_threads)
    if args.device == "cpu" and args.cpu_threads > 0 and len(affinity) != args.cpu_threads:
        raise ValueError("CPU smoke 的 --affinity 数量必须等于 --cpu-threads")
    config = TTSWorkerConfig(
        model_path=str(model_path),
        device=args.device,
        voice_profiles=(profile,),
        output_dir=str(run_dir),
        cpu_threads=args.cpu_threads,
        cpu_affinity=affinity,
        dtype=args.dtype,
        attention_backend=args.attention_backend,
        backend_kind="real_qwen3_tts",
        generation_kwargs={
            "do_sample": args.do_sample,
            "max_new_tokens": args.max_new_tokens,
            "repetition_penalty": args.repetition_penalty,
        },
    )
    worker = Qwen3TTSWorker(config)
    results = []
    experiment_started = time.perf_counter()
    try:
        health = await worker.start()
        for chunk_id in range(args.chunks):
            text = DEFAULT_TEXTS[chunk_id % len(DEFAULT_TEXTS)]
            request = TTSChunkRequest(
                request_id=run_id,
                request_epoch=0,
                chunk_id=chunk_id,
                raw_text=text,
                synthesis_text=text,
                language=args.language,
                voice_profile_id=args.profile_id,
                output_mode=TTSOutputMode.FILE,
                seed=args.seed + chunk_id,
            )
            results.append(await worker.synthesize(request))
    finally:
        await worker.stop()
    total_s = time.perf_counter() - experiment_started

    if [result.chunk_id for result in results] != list(range(args.chunks)):
        raise RuntimeError("chunk_id 顺序验收失败")
    if not all(result.backend_kind == "real_qwen3_tts" for result in results):
        raise RuntimeError("真实冒烟出现非真实后端结果")
    if not all(result.model_load_count == 1 for result in results):
        raise RuntimeError("模型被重复加载")
    if not all(result.voice_prompt_build_count == 1 for result in results):
        raise RuntimeError("音色 prompt 构建次数不是 1")

    chunks_path = run_dir / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as output:
        for result in results:
            output.write(json.dumps(result.to_record(), ensure_ascii=False) + "\n")

    reproducibility = ReproducibilityManager(seed=args.seed)
    manifest_config = {
        "tts_model_path": str(model_path),
        "tts_model_tree_sha256": model_sha256,
        "tts_model_files": model_files,
        "model_hash_s": model_hash_s,
        "reference_audio_path": str(ref_audio_path),
        "reference_audio_sha256": ref_audio_sha256,
        "voice_profile_id": args.profile_id,
        "voice_clone_mode": "x_vector_only_ablation" if args.x_vector_only else "icl",
        "reference_text_supplied": bool(args.ref_text.strip()),
        "language": args.language,
        "cpu_threads": args.cpu_threads,
        "requested_cpu_affinity": list(affinity),
        "actual_cpu_affinity": list(health.cpu_affinity),
        "dtype": args.dtype,
        "attention_backend": args.attention_backend,
        "device": args.device,
        "backend_kind": "real_qwen3_tts",
        "fallback_allowed": False,
        "fallback_occurred": any(
            result.backend_kind != "real_qwen3_tts" for result in results
        ),
        "sentence_level_streaming": True,
        "true_audio_streaming": False,
        "non_streaming_mode": True,
        "chunks": args.chunks,
        "generation_kwargs": config.generation_kwargs,
    }
    manifest = reproducibility.create_manifest(
        manifest_config, "qwen3_tts_base_smoke"
    )
    manifest["run_id"] = run_id
    manifest["host"] = platform.node()
    git_root = Path(args.git_root).expanduser().resolve() if args.git_root else None
    manifest["runtime_platform"] = collect_runtime_platform(git_root)
    manifest["total_s"] = total_s
    manifest["worker_start_health"] = health.to_dict()
    manifest["result_summary"] = {
        "chunk_ids": [result.chunk_id for result in results],
        "worker_epochs": sorted({result.worker_epoch for result in results}),
        "model_load_counts": sorted({result.model_load_count for result in results}),
        "voice_prompt_build_counts": sorted(
            {result.voice_prompt_build_count for result in results}
        ),
        "mean_rtf": sum(result.rtf for result in results) / len(results),
        "first_chunk_ready_s": results[0].first_chunk_ready_s,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"run_dir": str(run_dir), **manifest["result_summary"]}, ensure_ascii=False))
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=os.environ.get(
            "XIAOYOU_BENCH_TTS_PATH",
            "/home/leslie/Autarkia/models/Qwen3-TTS-12Hz-0.6B-Base",
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
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=0)
    parser.add_argument("--affinity", type=int, nargs="*")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--chunks", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument(
        "--git-root",
        default=os.environ.get("XIAOYOU_GIT_ROOT", ""),
        help="用于记录 commit/dirty 状态的源码 Git 根目录",
    )
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "experiment_results" / "tts" / "smoke"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.chunks < 1:
        raise ValueError("--chunks 必须大于 0")
    asyncio.run(run_smoke(args))


if __name__ == "__main__":
    main()
