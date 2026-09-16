"""Qwen3-TTS 推理后端对照：上游动态路径 vs CUDA Graph 静态路径。

**目的**：判定「TTS 后端能否达到实时（RTF < 1）」，不是换模型。
模型、参考音频、文本、克隆模式与采样参数全部沿用 B2 冻结口径
（`tts_component_offload_matrix.TEXTS`，x-vector 克隆，非流式完整波形，
`do_sample=False`）。两条路径在**同一进程内先后测量**：跑完并释放前者再加载
后者，避免 8GB 显存同时驻留两份模型。后端加载见 `qwen3_tts_backends`。

**边界（必须随结果一起报告）**：

- CUDA Graph 路径来自外部 MIT 项目 `faster-qwen3-tts`，通过
  ``--faster-qwen3-tts-path`` 挂到 ``sys.path``，**不安装进本仓库环境**；
- 它用静态 KV cache + attention mask，BF16 下与上游走不同 SDPA kernel，
  **不是位精确**，两路径 WAV/PCM SHA-256 不同属预期，需逐条记录；
- 首次调用的图捕获成本单独记为 ``graph_capture_s``，不计入 RTF；
- 只测「单句完整波形」，不涉及流式与播放，也不改任何冻结放置矩阵。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.elastic_residency.tts_component_offload_matrix import (  # noqa: E402
    EXPECTED_LENGTHS,
    TEXTS,
)
from experiments.tts.profiling_utils import (  # noqa: E402
    collect_runtime_platform,
    model_tree_hash,
    sha256_file,
)
from experiments.tts.qwen3_tts_backends import (  # noqa: E402,F401
    DEFAULT_FASTER_PATH,
    PATH_CUDA_GRAPH,
    PATH_UPSTREAM,
    build_synth,
    package_version,
    release_gpu,
)


def to_mono(audio: Any) -> np.ndarray:
    """统一成 (samples,) float32；上游与加速路径都可能返回 list。"""

    array = audio[0] if isinstance(audio, (list, tuple)) else audio
    return np.asarray(array, dtype=np.float32).reshape(-1)


def pcm_sha256(samples: np.ndarray) -> str:
    """对 PCM16 负载取哈希；与 WAV 头无关，便于跨后端比较。"""

    clipped = np.clip(samples, -1.0, 1.0)
    return hashlib.sha256((clipped * 32767.0).astype(np.int16).tobytes()).hexdigest()


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """按 (path, text) 汇总 RTF 与耗时；纯函数，便于静态验证。"""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(f"{row['path']}|{row['text']}", []).append(row)
    summary: dict[str, Any] = {}
    for key, items in grouped.items():
        rtf = [float(item["rtf"]) for item in items]
        elapsed = [float(item["elapsed_s"]) for item in items]
        audio = [float(item["audio_s"]) for item in items]
        summary[key] = {
            "rounds": len(items),
            "rtf_mean": round(sum(rtf) / len(rtf), 4),
            "rtf_min": round(min(rtf), 4),
            "rtf_max": round(max(rtf), 4),
            "elapsed_mean_s": round(sum(elapsed) / len(elapsed), 4),
            "audio_mean_s": round(sum(audio) / len(audio), 4),
            "distinct_pcm_sha256": sorted({item["pcm_sha256"] for item in items}),
        }
    return summary


def _measure(
    *,
    path: str,
    rounds: int,
    synth: Any,
    run_dir: Path,
    save_audio: bool,
) -> list[dict[str, Any]]:
    import torch

    rows: list[dict[str, Any]] = []
    for name, text in TEXTS.items():
        for index in range(1, rounds + 1):
            torch.cuda.synchronize()
            started = time.perf_counter()
            audio, sample_rate = synth(text)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            samples = to_mono(audio)
            duration = float(len(samples)) / float(sample_rate)
            record = {
                "path": path,
                "text": name,
                "text_chars": len(text),
                "round": index,
                "elapsed_s": round(elapsed, 4),
                "audio_s": round(duration, 4),
                "sample_rate": int(sample_rate),
                "samples": int(len(samples)),
                "rtf": round(elapsed / duration, 4),
                "realtime_x": round(duration / elapsed, 3),
                "pcm_sha256": pcm_sha256(samples),
            }
            if save_audio:
                import soundfile as sf

                audio_dir = run_dir / "audio"
                audio_dir.mkdir(parents=True, exist_ok=True)
                target = audio_dir / f"{path}_{name}_r{index}.wav"
                sf.write(str(target), samples, int(sample_rate))
                record["wav_path"] = str(target)
            rows.append(record)
            print(f"[{path}] {name} r{index} RTF={record['rtf']} audio={duration:.2f}s")
    return rows


def run_probe(args: argparse.Namespace) -> Path:
    lengths = {name: len(text) for name, text in TEXTS.items()}
    if lengths != EXPECTED_LENGTHS:
        raise ValueError(f"B2 冻结文本长度不一致: {lengths}")
    if not Path(args.model).is_dir():
        raise FileNotFoundError(f"找不到本地模型目录: {args.model}")
    if not Path(args.ref_audio).is_file():
        raise FileNotFoundError(f"找不到参考音频: {args.ref_audio}")

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (Path(args.output_root) / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    model_sha256, model_files = model_tree_hash(Path(args.model))
    rows: list[dict[str, Any]] = []
    path_meta: dict[str, Any] = {}
    for path in args.paths:
        synth, meta = build_synth(args, path)
        path_meta[path] = meta
        rows.extend(
            _measure(
                path=path,
                rounds=args.rounds,
                synth=synth,
                run_dir=run_dir,
                save_audio=not args.no_audio,
            )
        )
        release_gpu()

    with (run_dir / "chunks.jsonl").open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = aggregate(rows)
    manifest = {
        "run_id": run_id,
        "purpose": "tts_backend_compare_not_model_swap",
        "paths": list(args.paths),
        "path_meta": path_meta,
        "tts_model_path": str(Path(args.model).resolve()),
        "tts_model_tree_sha256": model_sha256,
        "tts_model_file_count": len(model_files),
        "reference_audio_path": str(Path(args.ref_audio).resolve()),
        "reference_audio_sha256": sha256_file(Path(args.ref_audio)),
        "language": args.language,
        "voice_clone_mode": "x_vector_only_ablation" if args.xvec_only else "icl",
        "texts": TEXTS,
        "rounds": args.rounds,
        "max_new_tokens": args.max_new_tokens,
        "repetition_penalty": args.repetition_penalty,
        "do_sample": False,
        "non_streaming_mode": True,
        "attn_implementation": args.attn_implementation,
        "max_seq_len": args.max_seq_len,
        "faster_qwen3_tts_path": str(
            Path(args.faster_qwen3_tts_path).expanduser().resolve()
        ),
        "bit_exact_across_paths": False,
        "bit_exact_note": (
            "CUDA Graph 静态 KV cache 与上游走不同 SDPA kernel，"
            "BF16 下不是位精确，WAV/PCM SHA 不同属预期"
        ),
        "fallback_allowed": False,
        "runtime_platform": collect_runtime_platform(PROJECT_ROOT),
        "summary": summary,
    }
    if PATH_CUDA_GRAPH in args.paths:
        manifest["faster_qwen3_tts_version"] = package_version("faster_qwen3_tts")
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"run_dir": str(run_dir), "summary": summary}, ensure_ascii=False))
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
    parser.add_argument(
        "--paths",
        nargs="+",
        choices=(PATH_UPSTREAM, PATH_CUDA_GRAPH),
        default=[PATH_UPSTREAM, PATH_CUDA_GRAPH],
    )
    parser.add_argument(
        "--faster-qwen3-tts-path",
        dest="faster_qwen3_tts_path",
        default=DEFAULT_FASTER_PATH,
    )
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument(
        "--attn-implementation", default="eager", choices=("eager", "sdpa")
    )
    parser.add_argument(
        "--xvec-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="CUDA Graph 路径的克隆模式；默认与 B2 的 x-vector 口径一致",
    )
    parser.add_argument(
        "--warmup-text", default="这是一次不计入统计的预热，用来稳定执行路径。"
    )
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "experiment_results" / "tts" / "cudagraph"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.rounds < 1:
        raise ValueError("--rounds 必须大于 0")
    run_probe(args)


if __name__ == "__main__":
    main()
