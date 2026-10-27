"""analyze_s0_slack 的合成事件验证（不依赖真实产物）。"""

from __future__ import annotations

import json
from pathlib import Path

from experiments.elastic_residency.analyze_s0_slack import (
    analyze,
    burst_fit,
    request_metrics,
)

S = 1_000_000_000


def _event(stage: str, t_s: float, **attributes) -> dict:
    return {
        "stage": stage,
        "timestamp_ns": int(t_s * S),
        "request_id": "K24-T0-first_sentence-r1",
        "config": "K24:T0",
        "attributes": attributes,
    }


def _synthetic_request() -> list[dict]:
    """两句：token 每 0.25 s 一个；TTS 在 [2,3] 与 [5,6] 执行；各 4 s 音频。"""

    events = [_event("request_start", 0.0)]
    events += [_event("token_delta", 1.0 + 0.25 * i, token_index=i) for i in range(21)]
    events += [
        _event("tts_exec_start", 2.0),
        _event("tts_exec_end", 3.0),
        _event("waveform_ready", 3.0, audio_duration_s=4.0),
        _event("playback_start", 3.1),
        _event("tts_exec_start", 5.0),
        _event("tts_exec_end", 6.0),
        _event("waveform_ready", 6.0, audio_duration_s=4.0),
        _event("playback_start", 7.1),
    ]
    return events


def test_request_metrics_slack_and_rates():
    row = request_metrics(_synthetic_request())
    assert row["output_tokens"] == 21
    assert row["token_rate_overall"] == 4.0
    assert row["token_rate_tts_idle"] == 4.0
    assert row["tts_exec_s"] == 2.0
    assert row["audio_s"] == 8.0
    assert row["first_ready_s"] == 3.0
    assert row["last_ready_s"] == 6.0
    # 首播 3.1 s + 8 s 音频 = 11.1 s；减末句就绪 6 s
    assert row["nominal_playback_end_s"] == 11.1
    assert row["post_synthesis_slack_s"] == 5.1
    assert row["starve_episodes_incl_startup"] == 0


def test_burst_fit_margin_sign():
    assert burst_fit(10.0, 5.0, 1.0, 3.0)["fits"] is True
    result = burst_fit(10.0, 5.0, 1.0, 5.0)
    assert result["fits"] is False and result["margin_s"] == -1.0


def test_analyze_with_isolated_profile(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "events.jsonl").write_text(
        "\n".join(json.dumps(e) for e in _synthetic_request()) + "\n", encoding="utf-8"
    )
    iso = tmp_path / "iso.jsonl"
    iso.write_text(
        "\n".join(
            json.dumps({"context_tokens": c, "config": "K24", "inference": {"decode_tokens_per_s": r}})
            for c, r in ((512, 14.0), (512, 16.0), (2048, 1.0))
        )
        + "\n",
        encoding="utf-8",
    )
    result = analyze([run], [iso], 512, (2.0, 0.5, 1.0))
    agg = result["by_config"]["run|K24:T0"]
    assert agg["isolated_decode_rate"] == 15.0
    assert agg["colocation_slowdown_x"] == 3.75
    assert result["requests"][0]["burst_fit"]["fits"] is True
