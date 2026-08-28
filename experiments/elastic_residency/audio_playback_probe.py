#!/usr/bin/env python3
"""跨 WSL2/Windows 的真实 PortAudio 播放与 underrun 探针。"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import threading
import time
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WINDOWS_PYTHON = Path(
    os.environ.get(
        "XIAOYOU_WINDOWS_PYTHON",
        "/mnt/d/AI/xiaoyou-core/venv_core/Scripts/python.exe",
    )
)


@dataclass(frozen=True)
class PlaybackProbeResult:
    passed: bool
    backend: str
    transport: str
    device_id: int | None
    device_name: str
    host_api: str
    source_sample_rate: int
    source_channels: int
    sample_rate: int
    channels: int
    frame_count: int
    audio_duration_s: float
    playback_wall_s: float
    stream_latency_s: float
    callback_count: int
    output_underflow_count: int
    callback_status_count: int
    callback_gap_max_ms: float
    error: str = ""


def _load_pcm16_wav(path: Path) -> tuple[np.ndarray, int, int]:
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        frame_count = wav_file.getnframes()
        raw = wav_file.readframes(frame_count)
    if sample_width != 2:
        raise ValueError(f"只支持 PCM16 WAV，当前 sample_width={sample_width}")
    if channels <= 0 or sample_rate <= 0 or frame_count <= 0:
        raise ValueError("WAV 的声道、采样率和帧数必须为正")
    audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return audio.reshape(-1, channels), sample_rate, channels


def _output_devices(sounddevice: Any) -> list[tuple[int, dict[str, Any]]]:
    return [
        (index, dict(device))
        for index, device in enumerate(sounddevice.query_devices())
        if int(device.get("max_output_channels", 0)) > 0
    ]


def _choose_output_device(
    sounddevice: Any,
    requested_device: int | None,
    *,
    prefer_wasapi: bool,
) -> tuple[int, dict[str, Any], str]:
    devices = _output_devices(sounddevice)
    if not devices:
        raise RuntimeError("PortAudio 未枚举到输出设备")
    by_id = dict(devices)
    if requested_device is not None:
        if requested_device not in by_id:
            raise ValueError(f"设备 {requested_device} 不是可用输出设备")
        chosen_id = requested_device
    else:
        chosen_id = -1
        if prefer_wasapi:
            for host_api in sounddevice.query_hostapis():
                if "WASAPI" not in str(host_api.get("name", "")).upper():
                    continue
                candidate = int(host_api.get("default_output_device", -1))
                if candidate in by_id:
                    chosen_id = candidate
                    break
        if chosen_id < 0:
            default_device = sounddevice.default.device
            candidate = int(default_device[1])
            chosen_id = candidate if candidate in by_id else devices[0][0]
    chosen = by_id[chosen_id]
    host_index = int(chosen.get("hostapi", -1))
    host_apis = sounddevice.query_hostapis()
    host_name = (
        str(host_apis[host_index].get("name", "unknown"))
        if 0 <= host_index < len(host_apis)
        else "unknown"
    )
    return chosen_id, chosen, host_name


def _resample_linear(
    audio: np.ndarray,
    source_rate: int,
    target_rate: int,
) -> np.ndarray:
    if source_rate == target_rate:
        return audio
    target_frames = max(1, round(len(audio) * target_rate / source_rate))
    source_positions = np.arange(len(audio), dtype=np.float64)
    target_positions = np.linspace(
        0,
        max(0, len(audio) - 1),
        target_frames,
        dtype=np.float64,
    )
    channels = [
        np.interp(target_positions, source_positions, audio[:, channel])
        for channel in range(audio.shape[1])
    ]
    return np.stack(channels, axis=1).astype(np.float32, copy=False)


def _negotiate_output_format(
    sounddevice: Any,
    audio: np.ndarray,
    source_rate: int,
    source_channels: int,
    device_id: int,
    device: dict[str, Any],
) -> tuple[np.ndarray, int, int]:
    default_rate = round(float(device.get("default_samplerate", source_rate)))
    rates = list(dict.fromkeys((source_rate, default_rate)))
    channel_options = [source_channels]
    if source_channels == 1 and int(device.get("max_output_channels", 0)) >= 2:
        channel_options.append(2)
    last_error: Exception | None = None
    for rate in rates:
        for channels in channel_options:
            try:
                sounddevice.check_output_settings(
                    device=device_id,
                    channels=channels,
                    dtype="float32",
                    samplerate=rate,
                )
            except Exception as exc:
                last_error = exc
                continue
            converted = _resample_linear(audio, source_rate, rate)
            if channels == 2 and source_channels == 1:
                converted = np.repeat(converted, 2, axis=1)
            return converted, rate, channels
    raise RuntimeError(f"输出设备不接受 WAV 格式或默认格式: {last_error}")


def _play_native(
    wav_path: Path,
    *,
    requested_device: int | None,
    blocksize: int,
    prefer_wasapi: bool,
) -> PlaybackProbeResult:
    import sounddevice as sd

    source_audio, source_rate, source_channels = _load_pcm16_wav(wav_path)
    device_id, device, host_name = _choose_output_device(
        sd,
        requested_device,
        prefer_wasapi=prefer_wasapi,
    )
    audio, sample_rate, channels = _negotiate_output_format(
        sd,
        source_audio,
        source_rate,
        source_channels,
        device_id,
        device,
    )
    position = 0
    callback_times: list[float] = []
    underflows = 0
    status_count = 0
    finished = threading.Event()

    def callback(
        outdata: np.ndarray,
        frames: int,
        _time_info: Any,
        status: Any,
    ) -> None:
        nonlocal position, underflows, status_count
        callback_times.append(time.perf_counter())
        if status:
            status_count += 1
            if bool(getattr(status, "output_underflow", False)):
                underflows += 1
        remaining = len(audio) - position
        copied = min(frames, remaining)
        outdata.fill(0)
        if copied > 0:
            outdata[:copied] = audio[position : position + copied]
            position += copied
        if position >= len(audio):
            raise sd.CallbackStop()

    started = time.perf_counter()
    with sd.OutputStream(
        samplerate=sample_rate,
        channels=channels,
        dtype="float32",
        blocksize=blocksize,
        latency="low",
        device=device_id,
        callback=callback,
        finished_callback=finished.set,
    ) as stream:
        timeout_s = (len(audio) / sample_rate) + 10.0
        if not finished.wait(timeout_s):
            raise TimeoutError("真实音频播放未在期限内结束")
        stream_latency_s = float(stream.latency)
    playback_wall_s = time.perf_counter() - started
    callback_gaps = [
        (right - left) * 1000.0
        for left, right in zip(callback_times, callback_times[1:])
    ]
    return PlaybackProbeResult(
        passed=bool(callback_times) and underflows == 0,
        backend=("windows_wasapi" if "WASAPI" in host_name.upper() else "native"),
        transport="direct_portaudio",
        device_id=device_id,
        device_name=str(device.get("name", "")),
        host_api=host_name,
        source_sample_rate=source_rate,
        source_channels=source_channels,
        sample_rate=sample_rate,
        channels=channels,
        frame_count=len(audio),
        audio_duration_s=len(source_audio) / source_rate,
        playback_wall_s=playback_wall_s,
        stream_latency_s=stream_latency_s,
        callback_count=len(callback_times),
        output_underflow_count=underflows,
        callback_status_count=status_count,
        callback_gap_max_ms=max(callback_gaps, default=0.0),
    )


def _wslpath_windows(path: Path) -> str:
    completed = subprocess.run(
        ["wslpath", "-w", str(path.resolve())],
        check=True,
        capture_output=True,
        text=True,
        timeout=5.0,
    )
    return completed.stdout.strip()


def _play_via_windows(
    wav_path: Path,
    *,
    windows_python: Path,
    requested_device: int | None,
    blocksize: int,
) -> PlaybackProbeResult:
    if not windows_python.is_file():
        raise FileNotFoundError(f"找不到 Windows Python: {windows_python}")
    command = [
        str(windows_python),
        _wslpath_windows(Path(__file__)),
        "--wav",
        _wslpath_windows(wav_path),
        "--backend",
        "native",
        "--prefer-wasapi",
        "--blocksize",
        str(blocksize),
        "--json",
    ]
    if requested_device is not None:
        command.extend(("--device", str(requested_device)))
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=120.0,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(
            "Windows 播放探针没有返回 JSON: " + completed.stderr.strip()
        )
    payload = json.loads(lines[-1])
    if completed.returncode not in (0, 2):
        raise RuntimeError(payload.get("error") or completed.stderr.strip())
    payload["transport"] = "wsl_to_windows_portaudio"
    return PlaybackProbeResult(**payload)


def probe_wav(
    wav_path: str | Path,
    *,
    backend: str = "auto",
    windows_python: str | Path = DEFAULT_WINDOWS_PYTHON,
    requested_device: int | None = None,
    blocksize: int = 1024,
    prefer_wasapi: bool = False,
) -> PlaybackProbeResult:
    path = Path(wav_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"找不到 WAV: {path}")
    if blocksize <= 0:
        raise ValueError("blocksize 必须大于 0")
    if backend not in {"auto", "native", "windows"}:
        raise ValueError(f"不支持的 backend: {backend}")
    is_wsl = "microsoft" in platform.release().lower()
    if backend == "windows" or (backend == "auto" and is_wsl):
        if backend == "auto":
            try:
                import sounddevice as sd

                if _output_devices(sd):
                    return _play_native(
                        path,
                        requested_device=requested_device,
                        blocksize=blocksize,
                        prefer_wasapi=prefer_wasapi,
                    )
            except (ImportError, OSError):
                pass
        return _play_via_windows(
            path,
            windows_python=Path(windows_python),
            requested_device=requested_device,
            blocksize=blocksize,
        )
    return _play_native(
        path,
        requested_device=requested_device,
        blocksize=blocksize,
        prefer_wasapi=prefer_wasapi,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav", required=True)
    parser.add_argument("--backend", choices=("auto", "native", "windows"), default="auto")
    parser.add_argument("--windows-python", default=str(DEFAULT_WINDOWS_PYTHON))
    parser.add_argument("--device", type=int)
    parser.add_argument("--blocksize", type=int, default=1024)
    parser.add_argument("--prefer-wasapi", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output-json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        result = probe_wav(
            args.wav,
            backend=args.backend,
            windows_python=args.windows_python,
            requested_device=args.device,
            blocksize=args.blocksize,
            prefer_wasapi=args.prefer_wasapi,
        )
    except Exception as exc:
        result = PlaybackProbeResult(
            passed=False,
            backend=args.backend,
            transport="unavailable",
            device_id=None,
            device_name="",
            host_api="",
            source_sample_rate=0,
            source_channels=0,
            sample_rate=0,
            channels=0,
            frame_count=0,
            audio_duration_s=0.0,
            playback_wall_s=0.0,
            stream_latency_s=0.0,
            callback_count=0,
            output_underflow_count=0,
            callback_status_count=0,
            callback_gap_max_ms=0.0,
            error=f"{type(exc).__name__}: {exc}",
        )
    payload = asdict(result)
    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
    print(json.dumps(payload, ensure_ascii=False) if args.json else payload)
    raise SystemExit(0 if result.passed else 2)


if __name__ == "__main__":
    main()
