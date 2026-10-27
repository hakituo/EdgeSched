"""B2：当前接入后端（加速整模型 GPU）的 TTS 句长画像。

**B2 就是本文件产出的画像**。当前接入后端 = CUDA Graph 静态路径，整模型驻 GPU
（放置维度 T0），短/中/长 = 20/50/100 字，每档五轮。旧后端的组件放置矩阵
T0–T4 及其三份产物已删除，B2 不再引用任何旧后端数值；T1–T4 只有在确认加速
后端支持后才重测，否则由 ``b2_profile_contract.placement_status`` 显式排除，
不为了凑齐档位沿用旧值。

每轮记录：RTF、波形就绪时间、加载与图准备成本、准备与执行阶段内存峰值、
输出长度；波形正确性由主环境 ASR 回转核对补写同目录 ``correctness.json``。

两条推理路径在同一进程内先后测量（跑完释放前者再加载后者，避免 8GB 显存
同时驻留），因此 A/B 对照与 B2 画像共用同一入口：B2 取加速路径，A/B 取两条。

**边界（必须随结果一起报告）**：

- CUDA Graph 路径来自外部 MIT 项目 `faster-qwen3-tts`，通过
  ``--faster-qwen3-tts-path`` 挂到 ``sys.path``，**不安装进本仓库环境**；
- 它用静态 KV cache + attention mask，BF16 下与上游走不同 SDPA kernel，
  **不是位精确**，两路径 WAV/PCM SHA-256 不同属预期，需逐条记录；
- 首次调用的图捕获成本单独记为 ``graph_capture_s``，不计入 RTF；
- 只测「单句完整波形」，不涉及流式与播放。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.tts.b2_profile_contract import (  # noqa: E402
    EXPECTED_LENGTHS,
    PLACEMENT_CURRENT,
    TEXTS,
    WARMUP_TEXT,
    aggregate,
    build_b2_summary,
    correctness_sequence,
    length_sequence,
    pcm_sha256,
    placement_status,
    to_mono,
    validate_texts,
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
    accelerated_placement_capability,
    build_synth,
    package_version,
    release_gpu,
)
from experiments.tts.tts_phase_memory import PhaseMemory, gpu_snapshot  # noqa: E402


def measure(
    *,
    path: str,
    sequence: list[tuple[str, str]],
    synth: Any,
    run_dir: Path,
    save_audio: bool,
    execute: PhaseMemory,
) -> list[dict[str, Any]]:
    """按序列逐句测量；内存采样发生在计时区间之外，不污染耗时。"""

    import torch

    rows: list[dict[str, Any]] = []
    for call_index, (name, text) in enumerate(sequence, start=1):
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
            "call_index": call_index,
            "text_chars": len(text),
            "expected_text": text,
            "elapsed_s": round(elapsed, 4),
            "waveform_ready_s": round(elapsed, 4),
            "waveform_ready_semantics": (
                "同步后单句完整波形就绪（含解码），非流式，不含播放"
            ),
            "audio_s": round(duration, 4),
            "sample_rate": int(sample_rate),
            "samples": int(len(samples)),
            "rtf": round(elapsed / duration, 4),
            "realtime_x": round(duration / elapsed, 3),
            "pcm_sha256": pcm_sha256(samples),
            "memory": execute.sample(),
        }
        if save_audio:
            import soundfile as sf

            audio_dir = run_dir / "audio"
            audio_dir.mkdir(parents=True, exist_ok=True)
            target = audio_dir / f"{path}_{name}_c{call_index:02d}.wav"
            sf.write(str(target), samples, int(sample_rate))
            record["wav_path"] = str(target)
        rows.append(record)
        print(
            f"[{path}] {name} c{call_index} ready={record['waveform_ready_s']}s "
            f"audio={duration:.2f}s RTF={record['rtf']}"
        )
    return rows


def _sequence(args: argparse.Namespace) -> list[tuple[str, str]]:
    if args.sequence == "correctness":
        return correctness_sequence()
    return length_sequence(args.rounds)


def resolve_git_state(
    args: argparse.Namespace, runtime_platform: dict[str, Any]
) -> dict[str, Any]:
    """来源状态优先用启动器在主机侧读到的值。

    WSL2 同步过去的目录没有 ``.git``，进程内探测必然返回 ``available=false``，
    因此来源门必须由启动器传入（与旧矩阵 ``--source-git-*`` 约定一致）。
    """

    if args.source_git_commit:
        if args.source_git_dirty not in ("true", "false"):
            raise ValueError("--source-git-dirty 必须是 true 或 false")
        return {
            "available": True,
            "commit": args.source_git_commit,
            "dirty": args.source_git_dirty == "true",
            "dirty_file_count": args.source_git_status_entry_count,
            "source": "launcher_supplied",
        }
    probed = dict(runtime_platform.get("git") or {})
    probed["source"] = "in_process_probe"
    probed.setdefault("available", False)
    return probed


def run_probe(args: argparse.Namespace) -> Path:
    validate_texts()
    if not Path(args.model).is_dir():
        raise FileNotFoundError(f"找不到本地模型目录: {args.model}")
    if not Path(args.ref_audio).is_file():
        raise FileNotFoundError(f"找不到参考音频: {args.ref_audio}")

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (Path(args.output_root) / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    model_sha256, model_files = model_tree_hash(Path(args.model))
    idle_card_used_mb = gpu_snapshot(os.getpid()).get("gpu_card_used_mb")
    capability = accelerated_placement_capability(args.faster_qwen3_tts_path)
    placement = placement_status(capability)
    sequence = _sequence(args)

    rows: list[dict[str, Any]] = []
    path_meta: dict[str, Any] = {}
    phases: dict[str, Any] = {}
    for path in args.paths:
        prepare = PhaseMemory("prepare", os.getpid())
        prepare.sample()
        synth, meta = build_synth(args, path)
        prepare.sample()
        prepare_record = {**prepare.stop(), **meta}
        prepare_record["sample_semantics"] = (
            "加载前与加载(含图捕获)后各采样一次；卡级与 RSS 为这两点的较大值"
        )
        path_meta[path] = meta
        execute = PhaseMemory("execute", os.getpid())
        rows.extend(
            measure(
                path=path,
                sequence=sequence,
                synth=synth,
                run_dir=run_dir,
                save_audio=not args.no_audio,
                execute=execute,
            )
        )
        phases[path] = {
            "prepare": prepare_record,
            "execute": execute.stop(),
            "placement_status": placement,
        }
        release_gpu()

    with (run_dir / "chunks.jsonl").open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = aggregate(rows)
    runtime_platform = collect_runtime_platform(PROJECT_ROOT)
    git_state = resolve_git_state(args, runtime_platform)
    manifest = {
        "run_id": run_id,
        "purpose": "b2_tts_profile_on_current_backend",
        "paths": list(args.paths),
        "path_meta": path_meta,
        "phases": phases,
        "placement": PLACEMENT_CURRENT,
        "placement_status": placement,
        "sequence": args.sequence,
        "sequence_items": [name for name, _ in sequence],
        "tts_model_path": str(Path(args.model).resolve()),
        "tts_model_tree_sha256": model_sha256,
        "tts_model_file_count": len(model_files),
        "reference_audio_path": str(Path(args.ref_audio).resolve()),
        "reference_audio_sha256": sha256_file(Path(args.ref_audio)),
        "language": args.language,
        "voice_clone_mode": "x_vector_only_ablation" if args.xvec_only else "icl",
        "texts": TEXTS,
        "expected_lengths": EXPECTED_LENGTHS,
        "rounds": args.rounds,
        "idle_card_used_mb": idle_card_used_mb,
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
        "source_git_state": git_state,
        "runtime_platform": runtime_platform,
        "summary": summary,
    }
    if PATH_CUDA_GRAPH in args.paths:
        manifest["faster_qwen3_tts_version"] = package_version("faster_qwen3_tts")
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if PATH_CUDA_GRAPH in args.paths:
        b2_summary = build_b2_summary(
            rows,
            integration_path=PATH_CUDA_GRAPH,
            phases=phases[PATH_CUDA_GRAPH],
            idle_card_used_mb=idle_card_used_mb,
            git_state=git_state,
        )
        b2_summary["run_id"] = run_id
        (run_dir / "summary.json").write_text(
            json.dumps(b2_summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps({"run_dir": str(run_dir), "summary": summary}, ensure_ascii=False))
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
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument(
        "--sequence",
        choices=("lengths", "correctness"),
        default="lengths",
        help="lengths=20/50/100 字画像；correctness=重复与交替序列查状态残留",
    )
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
    parser.add_argument("--warmup-text", default=WARMUP_TEXT)
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument(
        "--source-git-commit",
        default="",
        help="主机侧读到的提交；WSL2 副本没有 .git，来源门必须由启动器传入",
    )
    parser.add_argument(
        "--source-git-dirty", choices=("true", "false", ""), default=""
    )
    parser.add_argument("--source-git-status-entry-count", type=int, default=None)
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
