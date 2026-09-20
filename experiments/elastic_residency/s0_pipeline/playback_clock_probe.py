"""设备时钟诊断探针：把 callback 的原始时间戳打出来，判断 DAC 时间能不能当「实际播放时刻」。

用途：验证不通过时，先分清是「我取错了字段」还是「这个设备的时间本身就不可信」。
只做诊断、不参与门禁；同一份探针在 WSL2 与原生 Linux 上都能跑，用于真机复跑时确认时钟。

判据复用 ``playback_clock.calibrate_pairs``（与 S0 正式运行同一套），并额外给出
后端/设备信息与 ``currentTime`` 对照，用于定位跨度比偏离的来源。
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from typing import Any

from experiments.elastic_residency.s0_pipeline.playback_clock import calibrate_pairs


def _device_report(device: Any) -> dict[str, Any]:
    import sounddevice as sd

    try:
        info = sd.query_devices(device, "output") if device is not None else sd.query_devices(kind="output")
    except Exception as exc:  # noqa: BLE001 - 诊断里只记录
        return {"error": f"{type(exc).__name__}: {exc}"}
    hostapi_index = info.get("hostapi")
    hostapi_name = None
    if isinstance(hostapi_index, int):
        try:
            hostapi_name = sd.query_hostapis(hostapi_index).get("name")
        except Exception:  # noqa: BLE001
            hostapi_name = None
    return {
        "name": info.get("name"),
        "hostapi_index": hostapi_index,
        "hostapi_name": hostapi_name,
        "default_samplerate": info.get("default_samplerate"),
        "max_output_channels": info.get("max_output_channels"),
        "default_low_output_latency": info.get("default_low_output_latency"),
        "default_high_output_latency": info.get("default_high_output_latency"),
    }


def probe(
    *,
    seconds: float,
    sample_rate: int,
    blocksize: int,
    device: Any,
    latency: str,
) -> dict[str, Any]:
    import numpy as np
    import sounddevice as sd

    rows: list[dict[str, int]] = []
    lock = threading.Lock()

    def callback(outdata: Any, frames: int, time_info: Any, status: Any) -> None:
        host_ns = time.monotonic_ns()
        dac_s = getattr(time_info, "outputBufferDacTime", None)
        current_s = getattr(time_info, "currentTime", None)
        with lock:
            rows.append(
                {
                    "host_ns": host_ns,
                    "dac_ns": int(float(dac_s) * 1_000_000_000)
                    if isinstance(dac_s, (int, float))
                    else None,
                    "current_ns": int(float(current_s) * 1_000_000_000)
                    if isinstance(current_s, (int, float))
                    else None,
                    "frames": int(frames),
                    "status": int(getattr(status, "_flags", 0) or 0),
                }
            )
        outdata[:] = 0.0

    stream = sd.OutputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        blocksize=blocksize,
        latency=latency,
        device=device,
        callback=callback,
    )
    started = time.monotonic()
    stream.start()
    try:
        time.sleep(seconds)
    finally:
        stream.stop()
        stream.close()
    with lock:
        snapshot = list(rows)

    usable = [
        (row["host_ns"], row["dac_ns"])
        for row in snapshot
        if row["dac_ns"] is not None
    ]
    currents = [
        row["current_ns"]
        for row in snapshot
        if row["current_ns"] is not None
    ]
    calibration = calibrate_pairs(usable, currents=currents or None).to_dict()
    hosts = [row["host_ns"] for row in snapshot]

    def _span(values: list[int]) -> int:
        return values[-1] - values[0] if len(values) > 1 else 0

    dac_values = [row["dac_ns"] for row in snapshot if row["dac_ns"] is not None]
    current_values = [row["current_ns"] for row in snapshot if row["current_ns"] is not None]
    host_span = _span(hosts)
    return {
        "device": _device_report(device),
        "portaudio_version": _portaudio_version(),
        "stream": {
            "sample_rate": sample_rate,
            "blocksize": blocksize,
            "latency": latency,
            "observed_s": round(time.monotonic() - started, 3),
        },
        "sample_count": len(snapshot),
        "host_span_ns": host_span,
        "dac_span_ns": _span(dac_values),
        "current_span_ns": _span(current_values),
        "dac_span_ratio": (
            round(_span(dac_values) / host_span, 6) if host_span else None
        ),
        "current_span_ratio": (
            round(_span(current_values) / host_span, 6) if host_span else None
        ),
        "dac_monotone": all(
            later > earlier for earlier, later in zip(dac_values, dac_values[1:])
        ),
        "calibration": calibration,
        "first_rows": snapshot[:3],
        "last_rows": snapshot[-3:],
        "numpy_present": bool(np is not None),
    }


def _portaudio_version() -> str | None:
    import sounddevice as sd

    try:
        return str(sd.get_portaudio_version()[1])
    except Exception:  # noqa: BLE001
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--blocksize", type=int, default=1024)
    parser.add_argument("--device", default=None)
    parser.add_argument("--latency", default="low", choices=("low", "high"))
    parser.add_argument("--output-json", default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = probe(
        seconds=args.seconds,
        sample_rate=args.sample_rate,
        blocksize=args.blocksize,
        device=args.device,
        latency=args.latency,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
