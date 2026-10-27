"""V0 的 TTS 持续合成负载：线程模式（同进程）与进程模式（独立进程）。

负载与 S0 加速后端同口径（``cuda_graph_static``，整模型 GPU，x-vector 音色），
循环合成固定句子，直到收到停止信号；每次合成记录开始/结束时刻与音频时长。
时间戳统一用 ``time.monotonic_ns()``，同机跨进程可比。

进程模式协议：子进程加载完成后向 stdout 打印一行 ``READY``；父进程关闭其 stdin
即为停止信号；子进程把全部记录写到 ``--out`` 后退出。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

#: 固定合成文本（约 4 s 音频），与 S0 句长同量级
LOOP_TEXT = "边缘计算能够在本地处理数据，降低网络延迟并保护隐私。"
DEFAULT_TTS_MODEL = os.environ.get(
    "EDGESCHED_TTS_MODEL", "/home/leslie/Autarkia/models/Qwen3-TTS-12Hz-0.6B-Base"
)
DEFAULT_REF_AUDIO = str(PROJECT_ROOT / "models/tts_refs/ref_calm.wav")


def load_synth(model: str, ref_audio: str) -> tuple[Callable[[str], Any], dict[str, Any]]:
    """加载加速 TTS 后端（与 S0 ``_load_tts_cuda_graph`` 相同参数）。"""

    from experiments.tts.b2_profile_contract import WARMUP_TEXT
    from experiments.tts.qwen3_tts_backends import DEFAULT_FASTER_PATH, cuda_graph_synth

    args = SimpleNamespace(
        model=model,
        ref_audio=ref_audio,
        language="Chinese",
        max_new_tokens=512,
        repetition_penalty=1.05,
        xvec_only=True,
        attn_implementation="eager",
        max_seq_len=2048,
        faster_qwen3_tts_path=DEFAULT_FASTER_PATH,
        warmup_text=WARMUP_TEXT,
    )
    return cuda_graph_synth(args)


def audio_seconds(output: Any) -> float:
    """generate_voice_clone 返回 (wavs, sr)，wavs 为列表。"""

    import numpy as np

    wavs, sample_rate = output[0], output[1]
    wav = wavs[0] if isinstance(wavs, (list, tuple)) else wavs
    return float(np.asarray(wav).reshape(-1).shape[0]) / float(sample_rate)


class SynthLoop:
    """在后台线程循环合成，直到 stop()。"""

    def __init__(self, synth: Callable[[str], Any], text: str = LOOP_TEXT) -> None:
        self._synth = synth
        self._text = text
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.records: list[dict[str, Any]] = []
        self.error: str | None = None

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                start = time.monotonic_ns()
                output = self._synth(self._text)
                end = time.monotonic_ns()
                self.records.append(
                    {"start_ns": start, "end_ns": end, "audio_s": audio_seconds(output)}
                )
        except Exception as exc:  # pragma: no cover - 仅真实后端命中
            self.error = f"{type(exc).__name__}: {exc}"

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="v0-tts-loop", daemon=True)
        self._thread.start()

    def stop(self, timeout_s: float = 60.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)


def run_for(loop: SynthLoop, seconds: float) -> None:
    """只跑 TTS 的单元：持续合成固定时长。"""

    loop.start()
    time.sleep(seconds)
    loop.stop()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="V0 TTS 持续合成子进程")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--affinity", default="", help="逗号分隔的逻辑 CPU")
    parser.add_argument("--tts-model", default=DEFAULT_TTS_MODEL)
    parser.add_argument("--ref-audio", default=DEFAULT_REF_AUDIO)
    args = parser.parse_args(argv)
    if args.affinity:
        os.sched_setaffinity(0, [int(cpu) for cpu in args.affinity.split(",")])
    synth, meta = load_synth(args.tts_model, args.ref_audio)
    loop = SynthLoop(synth)
    loop.start()
    print("READY", flush=True)
    sys.stdin.read()  # 父进程关闭 stdin 即停止
    loop.stop()
    payload = {
        "load_meta": {k: meta.get(k) for k in ("load_s", "graph_capture_s")},
        "affinity": sorted(os.sched_getaffinity(0)),
        "records": loop.records,
        "error": loop.error,
    }
    args.out.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0 if loop.error is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
