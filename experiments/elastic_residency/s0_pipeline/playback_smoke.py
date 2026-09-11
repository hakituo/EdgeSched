"""S0 播放器 smoke：只验证「测量能力」，不是真实流水线实验。

与真实 S0 运行的区别（必须同时记录）：

- 输入是确定性合成音频或已保存 WAV，**不是** LLM→TTS 的真实产物；
- 验收目标是设备能否打开、时间戳能否采集、缺音/断流能否被记录并分阶段；
- **测量通过与性能 SLO 通过严格分开**：``measurement_ok`` 只判测量可信，
  零断流由 ``continuity_ok`` 单独报告，不参与通过判定，允许测出断流。

用法（真实设备，无注入间隙）：

    python -m experiments.elastic_residency.s0_pipeline.playback_smoke \
        --output-json <path>

验证断流可被记录（人为注入供给间隙）：

    ... --gap-seconds 1.0
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.elastic_residency.s0_pipeline.events import (
    EventRecorder,
    Stage,
)
from experiments.elastic_residency.s0_pipeline.playback import ContinuousPlayer

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_TONE_HZ = 220.0
_AMPLITUDE = 0.05


def synthetic_sentence(sample_rate: int, seconds: float, index: int) -> np.ndarray:
    """确定性合成一句音频；句间用幅度微调区分，不做随机。"""

    count = max(1, int(sample_rate * seconds))
    timeline = np.arange(count, dtype=np.float32) / float(sample_rate)
    envelope = np.clip(np.linspace(1.0, 0.2, count), 0.0, 1.0).astype(np.float32)
    gain = 1.0 - 0.05 * (index % 4)
    wave = np.sin(2 * np.pi * _TONE_HZ * timeline).astype(np.float32)
    return (_AMPLITUDE * gain * wave * envelope).astype(np.float32)


def evaluate_measurement(
    *,
    stats: dict[str, Any],
    stages: dict[str, int],
    sentence_count: int,
    gap_injected: bool,
    starvation: dict[str, Any],
) -> dict[str, Any]:
    """把 smoke 结果分成「测量可信」与「连续性」两类结论。"""

    opened = bool(stats.get("is_real"))
    callbacks = int(stats.get("callback_count", 0))
    consumed = int(stats.get("consumed_samples", 0))
    submits = int(stages.get(Stage.PLAYBACK_SUBMIT.value, 0))
    starts = int(stages.get(Stage.PLAYBACK_START.value, 0))
    episodes = int(starvation.get("starve_episode_count", 0))
    faults = int(starvation.get("supply_fault_episode_count", 0))
    in_window = int(starvation.get("starve_samples_in_window", 0))
    underflow = int(stats.get("device_underflow_count", 0))
    callback_starves = int(starvation.get("starve_callback_count", 0))

    checks = {
        "device_opened": opened,
        "callback_observed": callbacks > 0,
        "samples_consumed": consumed > 0,
        "submit_events_complete": submits == sentence_count,
        "start_events_complete": starts == sentence_count,
        "episode_units_separated": episodes <= callback_starves,
        "phases_classified": bool(starvation.get("by_phase") is not None),
    }
    if gap_injected:
        # 注入的间隙必须表现为「供给缺口」，启动等待不算
        checks["supply_fault_detected"] = faults > 0

    measurement_ok = all(checks.values())
    # 连续性只看应连续供给区间内的缺口，不含启动等待与收尾补零
    continuity_ok = faults == 0 and underflow == 0
    return {
        "measurement_ok": measurement_ok,
        "measurement_checks": checks,
        # 连续性只报告，不参与 measurement_ok；零断流需真实设备级证据
        "continuity_ok": continuity_ok,
        "continuity": {
            "starve_episode_count": episodes,
            "starve_callback_count": callback_starves,
            "starve_samples_total": int(starvation.get("starve_samples_total", 0)),
            "supply_fault_episode_count": int(
                starvation.get("supply_fault_episode_count", 0)
            ),
            "supply_fault_samples": int(starvation.get("supply_fault_samples", 0)),
            "starve_samples_in_window": in_window,
            "window_samples": int(starvation.get("window_samples", 0)),
            "starve_ratio_in_window": starvation.get("starve_ratio_in_window"),
            "by_phase": starvation.get("by_phase", {}),
            "device_underflow_count": underflow,
        },
        "note": (
            "measurement_ok 只表示测量可信；continuity_ok 不是本 smoke 的通过条件。"
            "断流次数为合并后的独立段落，不等于缺供给 callback 次数；"
            "启动等待与收尾补零不计入供给缺口"
        ),
    }


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    recorder = EventRecorder(request_id=args.request_id, config="playback_smoke")
    player = ContinuousPlayer(
        sample_rate=args.sample_rate,
        recorder=recorder,
        blocksize=args.blocksize,
        device=args.device,
        max_pending_sentences=args.max_pending_sentences,
    )
    started = time.perf_counter()
    try:
        player.open()
    except Exception as exc:  # noqa: BLE001 - 设备不可用必须落盘而不是静默
        return {
            "passed": False,
            "measurement_ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "stage": "device_open",
            "note": "设备不可用：本环境没有可写音频 sink，播放测量在此阻塞",
        }

    calibration = player.calibrate_clock()
    try:
        for index in range(args.sentences):
            if args.wav_dir:
                samples = _load_wav(Path(args.wav_dir), index)
            else:
                samples = synthetic_sentence(
                    args.sample_rate, args.sentence_seconds, index
                )
            if args.gap_seconds > 0 and index == max(1, args.sentences // 2):
                # 先排空待播队列再注入间隙，否则缓冲会把它吸收掉，
                # 测出来的就不是断流而是「预缓冲掩盖了断流」
                while player.pending_sentences > 0:
                    await asyncio.sleep(0.01)
                await asyncio.sleep(args.gap_seconds)
            await player.submit(index, samples)
        player.mark_producer_done()
        finished = player.wait_finished(timeout_s=args.finish_timeout_s)
    finally:
        player.close()

    events = [item.to_dict() for item in recorder.snapshot()]
    stages: dict[str, int] = {}
    for event in events:
        stages[event["stage"]] = stages.get(event["stage"], 0) + 1
    stats = player.stats.to_dict()
    starvation = player.starve_summary()
    verdict = evaluate_measurement(
        stats=stats,
        stages=stages,
        sentence_count=args.sentences,
        gap_injected=args.gap_seconds > 0,
        starvation=starvation,
    )
    return {
        "passed": bool(verdict["measurement_ok"] and finished),
        "request_id": args.request_id,
        "elapsed_s": round(time.perf_counter() - started, 3),
        "finished": bool(finished),
        "audio_source": (
            f"wav_dir:{args.wav_dir}" if args.wav_dir else "synthetic_tone"
        ),
        "sentences": args.sentences,
        "sentence_seconds": args.sentence_seconds,
        "gap_seconds": args.gap_seconds,
        "sample_rate": args.sample_rate,
        "blocksize": args.blocksize,
        "device": args.device,
        "playback_stats": stats,
        "clock_calibration": (
            calibration.to_dict() if calibration is not None else None
        ),
        "clock_mapping": [item.to_dict() for item in recorder.mappings()],
        "starvation": starvation,
        "stage_counts": stages,
        **verdict,
        "events": events,
    }


def _load_wav(wav_dir: Path, index: int) -> np.ndarray:
    from services.tts.wav_codec import decode_pcm16_wav  # noqa: PLC0415

    files = sorted(wav_dir.glob("*.wav"))
    if not files:
        raise FileNotFoundError(f"目录下没有 WAV: {wav_dir}")
    payload = files[index % len(files)].read_bytes()
    samples, _ = decode_pcm16_wav(payload)
    return np.asarray(samples, dtype=np.float32).reshape(-1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--blocksize", type=int, default=1024)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--sentences", type=int, default=4)
    parser.add_argument("--sentence-seconds", type=float, default=1.5)
    parser.add_argument("--gap-seconds", type=float, default=0.0)
    parser.add_argument("--max-pending-sentences", type=int, default=4)
    parser.add_argument("--finish-timeout-s", type=float, default=60.0)
    parser.add_argument("--wav-dir", default="")
    parser.add_argument("--request-id", default="playback-smoke")
    parser.add_argument("--output-json", default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = asyncio.run(_run(args))
    if args.output_json:
        out = Path(args.output_json).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(
        json.dumps(
            {
                "passed": result.get("passed"),
                "measurement_ok": result.get("measurement_ok"),
                "continuity_ok": result.get("continuity_ok"),
                "error": result.get("error", ""),
            },
            ensure_ascii=False,
        )
    )
    raise SystemExit(0 if result.get("passed") else 1)


if __name__ == "__main__":
    main()
