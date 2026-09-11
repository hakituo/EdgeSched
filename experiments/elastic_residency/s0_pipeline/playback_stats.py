"""播放侧的数据结构、事件发射与时间语义判定。

从 ``playback`` 拆出以保持单文件规模上限；这里不含音频设备依赖，
只负责可序列化数据与「该用什么时间语义」这类纯判定。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from experiments.elastic_residency.s0_pipeline.events import (
    ClockDomain,
    EventRecorder,
    Stage,
    TimestampSemantics,
)


@dataclass(frozen=True)
class SentenceSpan:
    """单句在全局样本流中的位置，用于把断流追溯到具体句子。"""

    chunk_id: int
    start_sample: int
    end_sample: int
    submit_ns: int
    audio_duration_s: float


@dataclass
class PlaybackStats:
    """播放侧统计；代理证据必须显式标记 pending。"""

    submitted_sentences: int = 0
    played_sentences: int = 0
    consumed_samples: int = 0
    starve_events: int = 0
    starve_samples: int = 0
    starve_callback_count: int = 0
    starve_episode_count: int = 0
    starve_samples_in_window: int = 0
    device_underflow_count: int = 0
    callback_count: int = 0
    max_callback_gap_ns: int = 0
    used_device_clock: bool = False
    backend: str = ""
    is_real: bool = False
    pending_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "submitted_sentences": self.submitted_sentences,
            "played_sentences": self.played_sentences,
            "consumed_samples": self.consumed_samples,
            "starve_events": self.starve_events,
            "starve_samples": self.starve_samples,
            "starve_callback_count": self.starve_callback_count,
            "starve_episode_count": self.starve_episode_count,
            "starve_samples_in_window": self.starve_samples_in_window,
            "device_underflow_count": self.device_underflow_count,
            "callback_count": self.callback_count,
            "max_callback_gap_ns": self.max_callback_gap_ns,
            "used_device_clock": self.used_device_clock,
            "backend": self.backend,
            "is_real": self.is_real,
            "pending_notes": list(self.pending_notes),
        }


def emit_event(
    recorder: EventRecorder | None,
    stage: Stage,
    *,
    chunk_id: int | None,
    domain: ClockDomain,
    timestamp_ns: int,
    semantics: TimestampSemantics,
    attributes: dict[str, Any] | None = None,
) -> None:
    """事件发射；recorder 为空时静默跳过（播放器允许无记录器运行）。"""

    if recorder is None:
        return
    recorder.emit(
        stage,
        chunk_id=chunk_id,
        clock_domain=domain,
        timestamp_ns=timestamp_ns,
        semantics=semantics,
        attributes=attributes,
    )


def clock_choice(
    used_device_clock: bool,
) -> tuple[ClockDomain, TimestampSemantics, str]:
    """决定播放事件用哪种时间语义；设备时钟未经验证时必须退回 proxy。"""

    if used_device_clock:
        return ClockDomain.AUDIO_DEVICE, TimestampSemantics.DEVICE_CONSUME, ""
    return (
        ClockDomain.AUDIO_CALLBACK,
        TimestampSemantics.PROXY,
        "callback 触发时刻代理设备实际播放时刻，实际播放时间待校准",
    )
