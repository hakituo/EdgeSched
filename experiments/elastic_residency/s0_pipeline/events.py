"""S0 流水线的可解释事件契约（G4）。

时间语义边界（对应论文指南第 07 模块的边界要求）：

- token 源交付 delta 的时间，不等于模型内部逐 token 生成时间；
- 线程提交时间，不等于实际合成开始；
- 播放 callback 或入队时间，不等于设备实际播放时间；
- WSL 与 Windows 的时间戳不能未经校准直接相减。

因此每个事件必须同时声明 ``clock_domain`` 与 ``timestamp_semantics``：
只有同一时钟域内的时间戳可以直接相减，跨域必须先给出 ``ClockMapping``。
语义为 ``PROXY`` 的事件必须带 ``note``，说明它代理的是什么。
"""

from __future__ import annotations

import datetime as dt
import threading
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

SCHEMA_VERSION = "edgesched.s0_pipeline.v1"


class ClockDomain(str, Enum):
    """时间戳由谁产生；不同域之间不可直接相减。"""

    HOST_MONOTONIC = "host_monotonic"
    HOST_WALL_UTC = "host_wall_utc"
    TOKEN_THREAD = "token_source_thread"
    AUDIO_CALLBACK = "audio_callback"
    AUDIO_DEVICE = "audio_device"


class TimestampSemantics(str, Enum):
    """时间戳代表什么时刻；用于区分生成时间与交付时间。"""

    SUBMIT = "submit"
    DELIVERY = "delivery"
    EXECUTION_START = "execution_start"
    EXECUTION_END = "execution_end"
    DEVICE_CONSUME = "device_consume"
    PROXY = "proxy"


class Stage(str, Enum):
    """流水线阶段；每个阶段只表达一个可判定的事实。"""

    REQUEST_START = "request_start"
    LLM_LOAD_END = "llm_load_end"
    TOKEN_DELTA = "token_delta"
    TOKEN_END = "token_end"
    SENTENCE_READY = "sentence_ready"
    CHUNK_ENQUEUE = "chunk_enqueue"
    TTS_SUBMIT = "tts_submit"
    TTS_EXEC_START = "tts_exec_start"
    TTS_EXEC_END = "tts_exec_end"
    WAVEFORM_READY = "waveform_ready"
    PLAYBACK_SUBMIT = "playback_submit"
    PLAYBACK_START = "playback_start"
    PLAYBACK_END = "playback_end"
    PLAYBACK_STARVE = "playback_starve"
    DEVICE_UNDERRUN = "device_underrun"
    QUEUE_SNAPSHOT = "queue_snapshot"
    REQUEST_END = "request_end"
    CANCEL = "cancel"


DEFAULT_SEMANTICS: dict[Stage, TimestampSemantics] = {
    Stage.REQUEST_START: TimestampSemantics.EXECUTION_START,
    Stage.LLM_LOAD_END: TimestampSemantics.EXECUTION_END,
    Stage.TOKEN_DELTA: TimestampSemantics.DELIVERY,
    Stage.TOKEN_END: TimestampSemantics.DELIVERY,
    Stage.SENTENCE_READY: TimestampSemantics.DELIVERY,
    Stage.CHUNK_ENQUEUE: TimestampSemantics.SUBMIT,
    Stage.TTS_SUBMIT: TimestampSemantics.SUBMIT,
    Stage.TTS_EXEC_START: TimestampSemantics.EXECUTION_START,
    Stage.TTS_EXEC_END: TimestampSemantics.EXECUTION_END,
    Stage.WAVEFORM_READY: TimestampSemantics.DELIVERY,
    Stage.PLAYBACK_SUBMIT: TimestampSemantics.SUBMIT,
    Stage.PLAYBACK_START: TimestampSemantics.DEVICE_CONSUME,
    Stage.PLAYBACK_END: TimestampSemantics.DEVICE_CONSUME,
    Stage.PLAYBACK_STARVE: TimestampSemantics.EXECUTION_START,
    Stage.DEVICE_UNDERRUN: TimestampSemantics.DEVICE_CONSUME,
    Stage.QUEUE_SNAPSHOT: TimestampSemantics.PROXY,
    Stage.REQUEST_END: TimestampSemantics.EXECUTION_END,
    Stage.CANCEL: TimestampSemantics.EXECUTION_END,
}

_DEVICE_ONLY_SEMANTICS = frozenset({TimestampSemantics.DEVICE_CONSUME})
_DEVICE_DOMAINS = frozenset({ClockDomain.AUDIO_CALLBACK, ClockDomain.AUDIO_DEVICE})


@dataclass(frozen=True)
class PipelineEvent:
    """单条流水线事件；字段全集可反向定位到一次真实执行。"""

    request_id: str
    stage: str
    timestamp_ns: int
    clock_domain: str = ClockDomain.HOST_MONOTONIC.value
    timestamp_semantics: str = TimestampSemantics.DELIVERY.value
    chunk_id: int | None = None
    config: str = ""
    wall_time_iso: str = ""
    attributes: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.request_id.strip():
            raise ValueError("request_id 不能为空")
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns 不能为负")
        if self.stage not in {item.value for item in Stage}:
            raise ValueError(f"未知阶段: {self.stage}")
        if self.clock_domain not in {item.value for item in ClockDomain}:
            raise ValueError(f"未知时钟域: {self.clock_domain}")
        if self.timestamp_semantics not in {
            item.value for item in TimestampSemantics
        }:
            raise ValueError(f"未知时间语义: {self.timestamp_semantics}")
        if (
            self.timestamp_semantics in _DEVICE_ONLY_SEMANTICS
            and self.clock_domain not in {item.value for item in _DEVICE_DOMAINS}
        ):
            raise ValueError("device_consume 语义只能用于音频设备时钟域")
        if self.timestamp_semantics == TimestampSemantics.PROXY.value and not str(
            self.attributes.get("proxy_note", "")
        ).strip():
            raise ValueError("PROXY 语义必须提供 attributes['proxy_note']")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "chunk_id": self.chunk_id,
            "stage": self.stage,
            "config": self.config,
            "clock_domain": self.clock_domain,
            "timestamp_ns": self.timestamp_ns,
            "timestamp_semantics": self.timestamp_semantics,
            "wall_time_iso": self.wall_time_iso,
            "attributes": dict(self.attributes),
        }


@dataclass(frozen=True)
class ClockMapping:
    """跨时钟域映射：``target_ns = source_ns + offset_ns``。"""

    source_domain: ClockDomain
    target_domain: ClockDomain
    offset_ns: int
    max_error_ns: int
    method: str
    sample_count: int = 0
    note: str = ""

    def __post_init__(self) -> None:
        if self.max_error_ns < 0:
            raise ValueError("max_error_ns 不能为负")
        if not self.method.strip():
            raise ValueError("时钟映射必须记录 method")

    def map_ns(self, source_ns: int) -> int:
        return source_ns + self.offset_ns

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_domain": self.source_domain.value,
            "target_domain": self.target_domain.value,
            "offset_ns": self.offset_ns,
            "max_error_ns": self.max_error_ns,
            "method": self.method,
            "sample_count": self.sample_count,
            "note": self.note,
        }


class EventRecorder:
    """线程安全的事件记录器；自动补全墙钟与默认语义。"""

    def __init__(self, *, request_id: str, config: str = "") -> None:
        self.request_id = request_id
        self.config = config
        self._lock = threading.Lock()
        self._events: list[PipelineEvent] = []
        self._mappings: list[ClockMapping] = []

    def emit(
        self,
        stage: Stage,
        *,
        chunk_id: int | None = None,
        clock_domain: ClockDomain = ClockDomain.HOST_MONOTONIC,
        timestamp_ns: int | None = None,
        semantics: TimestampSemantics | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> PipelineEvent:
        resolved = semantics or DEFAULT_SEMANTICS[stage]
        event = PipelineEvent(
            request_id=self.request_id,
            stage=stage.value,
            timestamp_ns=(
                time.monotonic_ns() if timestamp_ns is None else int(timestamp_ns)
            ),
            clock_domain=clock_domain.value,
            timestamp_semantics=resolved.value,
            chunk_id=chunk_id,
            config=self.config,
            wall_time_iso=dt.datetime.now(dt.timezone.utc).isoformat(),
            attributes=dict(attributes or {}),
        )
        with self._lock:
            self._events.append(event)
        return event

    def add_mapping(self, mapping: ClockMapping) -> None:
        with self._lock:
            self._mappings.append(mapping)

    def snapshot(self) -> list[PipelineEvent]:
        with self._lock:
            return list(self._events)

    def mappings(self) -> list[ClockMapping]:
        with self._lock:
            return list(self._mappings)

    def to_records(self) -> list[dict[str, Any]]:
        return [event.to_dict() for event in self.snapshot()]


@dataclass(frozen=True)
class ExecutionInterval:
    """由一对 EXECUTION_START / EXECUTION_END 组成的真实执行区间。"""

    key: str
    start_ns: int
    end_ns: int
    clock_domain: str
    chunk_id: int | None = None

    @property
    def duration_ns(self) -> int:
        return max(0, self.end_ns - self.start_ns)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "chunk_id": self.chunk_id,
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "duration_ns": self.duration_ns,
            "clock_domain": self.clock_domain,
        }


def execution_intervals(
    events: Iterable[PipelineEvent],
    *,
    start_stage: Stage,
    end_stage: Stage,
    key: str,
) -> list[ExecutionInterval]:
    """只接受 EXECUTION_START/END，拒绝用提交时间冒充执行区间。"""

    starts: dict[int | None, int] = {}
    results: list[ExecutionInterval] = []
    for event in events:
        if event.stage == start_stage.value:
            if event.timestamp_semantics != TimestampSemantics.EXECUTION_START.value:
                raise ValueError(
                    f"{start_stage.value} 必须带 execution_start 语义"
                )
            starts[event.chunk_id] = event.timestamp_ns
        elif event.stage == end_stage.value:
            if event.timestamp_semantics != TimestampSemantics.EXECUTION_END.value:
                raise ValueError(f"{end_stage.value} 必须带 execution_end 语义")
            begin = starts.pop(event.chunk_id, None)
            if begin is None:
                continue
            results.append(
                ExecutionInterval(
                    key=key,
                    start_ns=begin,
                    end_ns=event.timestamp_ns,
                    clock_domain=event.clock_domain,
                    chunk_id=event.chunk_id,
                )
            )
    return results


def overlap_ns(left: ExecutionInterval, right: ExecutionInterval) -> int:
    if left.clock_domain != right.clock_domain:
        raise ValueError("跨时钟域区间必须先建立 ClockMapping 才能比较")
    return max(
        0, min(left.end_ns, right.end_ns) - max(left.start_ns, right.start_ns)
    )


def semantics_coverage(events: Iterable[PipelineEvent]) -> dict[str, Any]:
    """统计语义分布；用于判断能否支撑重叠与延迟结论。"""

    counts: dict[str, int] = {}
    for event in events:
        counts[event.timestamp_semantics] = counts.get(event.timestamp_semantics, 0) + 1
    total = sum(counts.values())
    proxy = counts.get(TimestampSemantics.PROXY.value, 0)
    submit_only = counts.get(TimestampSemantics.SUBMIT.value, 0)
    execution = counts.get(
        TimestampSemantics.EXECUTION_START.value, 0
    ) + counts.get(TimestampSemantics.EXECUTION_END.value, 0)
    return {
        "event_count": total,
        "semantics_counts": counts,
        "proxy_event_count": proxy,
        "submit_event_count": submit_only,
        "execution_event_count": execution,
        "supports_execution_interval": execution > 0,
        "has_proxy_evidence": proxy > 0,
    }
