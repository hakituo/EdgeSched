"""S0 播放余量与共跑减速的离线复算（纯标准库，不加载模型）。

从 S0 批次的 ``events.jsonl`` 逐请求复算：

- LLM 实际交付速度（全程、TTS 执行期间、TTS 空闲期间三种口径）；
- TTS 执行总时长、音频总时长、首句/末句波形就绪；
- 名义播放结束时刻 = 首个 playback_start + 音频总时长（不含断流补零，
  只在供给缺口为 0 时等于实际播放结束）；
- 合成后余量 = 名义播放结束 - 末句就绪，即所有音频合成完后，
  GPU 不再被本轮回复需要、而用户仍在听的时间；
- 可选：与 C2/C3 隔离画像的 decode 速度比较，给出共跑减速倍数；
- 可选：给定突发任务时长与切换代价，判断能否塞进合成后余量。

断流段数 ``starve_episodes_incl_startup`` 含首句前的启动缺音，
与 S0 报告里只计窗口内的「应用层供给缺口段数」不是同一口径。

playback_start 是 callback 代理时刻（设备时钟未验证），因此余量只是调度层
估计，不是实际播放时刻；结论写入论文前必须注明该口径。

用法::

    python -m experiments.elastic_residency.analyze_s0_slack \\
        experiment_results/elastic_residency/s0_pipeline/s0_formal_20260920_223105_8510199 \\
        --isolated experiment_results/elastic_residency/c2/c2_9b_small_20260907_143917/iterations.jsonl \\
        --isolated experiment_results/elastic_residency/c3/c3_9b_small_20260907_143917/iterations.jsonl \\
        --burst-s 5.2 --evict-s 0.5 --restore-s 5.6
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable

NS = 1e9


def load_events(run_dir: Path) -> list[dict[str, Any]]:
    """读取一个批次的全部事件。"""

    path = run_dir / "events.jsonl"
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _stage(events: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    return [event for event in events if event.get("stage") == name]


def _exec_intervals(events: list[dict[str, Any]]) -> list[tuple[int, int]]:
    """按时间顺序把 tts_exec_start/end 配成区间。"""

    intervals: list[tuple[int, int]] = []
    start: int | None = None
    for event in sorted(events, key=lambda item: item["timestamp_ns"]):
        if event["stage"] == "tts_exec_start":
            start = event["timestamp_ns"]
        elif event["stage"] == "tts_exec_end" and start is not None:
            intervals.append((start, event["timestamp_ns"]))
            start = None
    return intervals


def _token_gaps(
    token_ns: list[int], intervals: list[tuple[int, int]]
) -> tuple[list[float], list[float]]:
    """把相邻 token 间隔按中点是否落在 TTS 执行区间内分两组。"""

    busy: list[float] = []
    idle: list[float] = []
    for left, right in zip(token_ns, token_ns[1:]):
        middle = (left + right) / 2
        inside = any(start <= middle <= end for start, end in intervals)
        (busy if inside else idle).append((right - left) / NS)
    return busy, idle


def _rate(gaps: list[float]) -> float | None:
    return round(1.0 / median(gaps), 3) if gaps else None


def request_metrics(events: list[dict[str, Any]]) -> dict[str, Any]:
    """单个请求的余量与交付速度指标。"""

    start_ns = _stage(events, "request_start")[0]["timestamp_ns"]
    token_ns = sorted(event["timestamp_ns"] for event in _stage(events, "token_delta"))
    intervals = _exec_intervals(events)
    ready_ns = [event["timestamp_ns"] for event in _stage(events, "waveform_ready")]
    playback_ns = [event["timestamp_ns"] for event in _stage(events, "playback_start")]
    audio_s = sum(
        event["attributes"]["audio_duration_s"] for event in _stage(events, "waveform_ready")
    )
    tts_exec_s = sum(end - start for start, end in intervals) / NS
    busy, idle = _token_gaps(token_ns, intervals)
    span_s = (token_ns[-1] - token_ns[0]) / NS if len(token_ns) > 1 else None
    first_ready = (min(ready_ns) - start_ns) / NS
    last_ready = (max(ready_ns) - start_ns) / NS
    playback_end = (min(playback_ns) - start_ns) / NS + audio_s
    episodes = {
        event.get("attributes", {}).get("episode_id")
        for event in _stage(events, "playback_starve")
    }
    return {
        "config": events[0].get("config"),
        "output_tokens": len(token_ns),
        "token_rate_overall": round((len(token_ns) - 1) / span_s, 3) if span_s else None,
        "token_rate_tts_busy": _rate(busy),
        "token_rate_tts_idle": _rate(idle),
        "token_end_s": round((token_ns[-1] - start_ns) / NS, 3) if token_ns else None,
        "tts_exec_s": round(tts_exec_s, 3),
        "audio_s": round(audio_s, 3),
        "first_ready_s": round(first_ready, 3),
        "last_ready_s": round(last_ready, 3),
        "nominal_playback_end_s": round(playback_end, 3),
        "post_synthesis_slack_s": round(playback_end - last_ready, 3),
        "tts_busy_fraction": round(tts_exec_s / playback_end, 3),
        "starve_episodes_incl_startup": len(episodes - {None}),
        "slack_semantics": "callback_proxy_nominal_playback",
    }


def isolated_decode_rates(paths: Iterable[Path], context: int) -> dict[str, float]:
    """C2/C3 隔离画像中指定 context 的逐档 decode 中位数（tok/s）。"""

    rates: dict[str, list[float]] = defaultdict(list)
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("context_tokens") != context:
                    continue
                rates[str(record["config"])].append(
                    record["inference"]["decode_tokens_per_s"]
                )
    return {config: round(median(values), 3) for config, values in rates.items()}


def burst_fit(slack_s: float, burst_s: float, evict_s: float, restore_s: float) -> dict[str, Any]:
    """判断一次突发任务能否放进合成后余量；margin<0 表示会造成断流。"""

    need = evict_s + burst_s + restore_s
    return {
        "need_s": round(need, 3),
        "slack_s": round(slack_s, 3),
        "margin_s": round(slack_s - need, 3),
        "fits": slack_s >= need,
    }


def summarize(rows: list[dict[str, Any]], isolated: dict[str, float]) -> dict[str, Any]:
    """按「批次|配置」聚合，不跨批次混合（不同批次可能换了后端或协议）。

    isolated 非空时给出共跑减速倍数（隔离速度 / 共跑速度）。供给缺口非 0 时，
    名义播放结束不等于实际播放结束，余量只能作下界参考。
    """

    by_config: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_config[f"{row.get('run')}|{row['config']}"].append(row)
    summary: dict[str, Any] = {}
    for group, items in sorted(by_config.items()):
        config = group.split("|", 1)[1]
        keys = [k for k, v in items[0].items() if isinstance(v, (int, float))]
        agg = {
            key: round(median(item[key] for item in items if item[key] is not None), 3)
            for key in keys
            if any(item[key] is not None for item in items)
        }
        agg["runs"] = len(items)
        llm_config = config.split(":")[0]
        if llm_config in isolated and agg.get("token_rate_overall"):
            agg["isolated_decode_rate"] = isolated[llm_config]
            agg["colocation_slowdown_x"] = round(
                isolated[llm_config] / agg["token_rate_overall"], 2
            )
        summary[group] = agg
    return summary


def analyze(
    run_dirs: list[Path],
    isolated_paths: list[Path],
    context: int,
    burst: tuple[float, float, float] | None,
) -> dict[str, Any]:
    """完整复算；返回可直接落盘的 JSON 结构。"""

    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for event in load_events(run_dir):
            grouped[event["request_id"]].append(event)
        for request_id, events in sorted(grouped.items()):
            if not _stage(events, "request_start") or not _stage(events, "waveform_ready"):
                continue
            row = request_metrics(events)
            row.update({"run": run_dir.name, "request_id": request_id})
            if burst is not None:
                row["burst_fit"] = burst_fit(row["post_synthesis_slack_s"], *burst)
            rows.append(row)
    isolated = isolated_decode_rates(isolated_paths, context) if isolated_paths else {}
    return {
        "schema": "edgesched.s0_slack_analysis.v1",
        "inputs": [str(path) for path in run_dirs],
        "isolated_inputs": [str(path) for path in isolated_paths],
        "isolated_context_tokens": context,
        "burst_assumption": (
            dict(zip(("burst_s", "evict_s", "restore_s"), burst)) if burst else None
        ),
        "requests": rows,
        "by_config": summarize(rows, isolated),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--isolated", action="append", type=Path, default=[])
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--burst-s", type=float)
    parser.add_argument("--evict-s", type=float, default=0.0)
    parser.add_argument("--restore-s", type=float, default=0.0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    burst = (args.burst_s, args.evict_s, args.restore_s) if args.burst_s is not None else None
    result = analyze(args.run_dirs, args.isolated, args.context, burst)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(json.dumps(result["by_config"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
