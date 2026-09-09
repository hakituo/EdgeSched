"""G1：句级连续播放器与断流记录。

两条断流口径严格分离：

- ``PLAYBACK_STARVE``：供给不足。回调需要样本但缓冲已耗尽，只能补零；
  补零样本数必须记录，不允许隐藏。
- ``DEVICE_UNDERRUN``：PortAudio 回调报告的 ``output_underflow``，属于设备侧。

自然静音不算断流：句内静音是真实样本，会被正常消费，不会产生 starve
事件；只有缓冲真的没有数据可给时才算供给不足。

时间口径：优先使用 ``outputBufferDacTime`` 作为设备播放时刻；取不到时只能
用 callback 触发时刻作为代理，语义标记为 ``proxy`` 并写入说明。
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from experiments.elastic_residency.s0_pipeline.events import (
    ClockDomain,
    ClockMapping,
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
            "device_underflow_count": self.device_underflow_count,
            "callback_count": self.callback_count,
            "max_callback_gap_ns": self.max_callback_gap_ns,
            "used_device_clock": self.used_device_clock,
            "backend": self.backend,
            "is_real": self.is_real,
            "pending_notes": list(self.pending_notes),
        }


class ContinuousPlayer:
    """按句追加、连续消费、可追溯样本位置的播放器。"""

    def __init__(
        self,
        *,
        sample_rate: int,
        recorder: EventRecorder | None = None,
        blocksize: int = 1024,
        device: int | None = None,
        max_pending_sentences: int = 4,
        backend: str = "auto",
    ) -> None:
        if sample_rate <= 0 or blocksize <= 0 or max_pending_sentences < 1:
            raise ValueError("播放器参数非法")
        self.sample_rate = sample_rate
        self.recorder = recorder
        self.blocksize = blocksize
        self.device = device
        self.max_pending_sentences = max_pending_sentences
        self.backend = backend
        self.stats = PlaybackStats()
        self._pending: deque[np.ndarray] = deque()
        self._offset = 0
        self._spans: deque[SentenceSpan] = deque()
        self._next_sample = 0
        self._consumed_samples = 0
        self._producer_done = False
        self._finished = threading.Event()
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._stream: Any = None
        self._last_callback_ns: int | None = None

    @property
    def is_opened(self) -> bool:
        return self._stream is not None

    @property
    def pending_sentences(self) -> int:
        with self._lock:
            return len(self._pending)

    def _emit(
        self,
        stage: Stage,
        *,
        chunk_id: int | None,
        domain: ClockDomain,
        timestamp_ns: int,
        semantics: TimestampSemantics,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        if self.recorder is None:
            return
        self.recorder.emit(
            stage,
            chunk_id=chunk_id,
            clock_domain=domain,
            timestamp_ns=timestamp_ns,
            semantics=semantics,
            attributes=attributes,
        )

    async def submit(self, chunk_id: int, samples: np.ndarray) -> SentenceSpan:
        """追加一句音频；待播放句数有界，从而把背压传回合成侧。"""

        array = np.asarray(samples, dtype=np.float32).reshape(-1)
        loop = asyncio.get_running_loop()
        span = await loop.run_in_executor(None, self._submit_blocking, chunk_id, array)
        self._emit(
            Stage.PLAYBACK_SUBMIT,
            chunk_id=chunk_id,
            domain=ClockDomain.HOST_MONOTONIC,
            timestamp_ns=span.submit_ns,
            semantics=TimestampSemantics.SUBMIT,
            attributes={
                "start_sample": span.start_sample,
                "end_sample": span.end_sample,
                "pending_sentences": self.pending_sentences,
            },
        )
        return span

    def _submit_blocking(self, chunk_id: int, array: np.ndarray) -> SentenceSpan:
        with self._condition:
            while (
                len(self._pending) >= self.max_pending_sentences
                and not self._producer_done
            ):
                self._condition.wait(timeout=1.0)
            start = self._next_sample
            end = start + len(array)
            self._next_sample = end
            self._pending.append(array)
            self._spans.append(
                SentenceSpan(
                    chunk_id=chunk_id,
                    start_sample=start,
                    end_sample=end,
                    submit_ns=time.monotonic_ns(),
                    audio_duration_s=len(array) / self.sample_rate,
                )
            )
            self.stats.submitted_sentences += 1
            self._condition.notify_all()
            return self._spans[-1]

    def _take(self, frames: int) -> tuple[np.ndarray, int]:
        """取最多 frames 个样本，返回 (数据, 缺数据而补零的样本数)。"""

        chunks: list[np.ndarray] = []
        remaining = frames
        while remaining > 0 and self._pending:
            head = self._pending[0]
            available = len(head) - self._offset
            take = min(available, remaining)
            chunks.append(head[self._offset : self._offset + take])
            self._offset += take
            remaining -= take
            if self._offset >= len(head):
                self._pending.popleft()
                self._offset = 0
                with self._condition:
                    self._condition.notify_all()
        if not chunks:
            return np.zeros(0, dtype=np.float32), frames
        data = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        return data, remaining

    def _current_chunk(self, sample_position: int) -> int | None:
        for span in self._spans:
            if span.start_sample <= sample_position < span.end_sample:
                return span.chunk_id
        return None

    def _consume(self, frames: int, status: Any, time_info: Any) -> np.ndarray:
        now_ns = time.monotonic_ns()
        with self._lock:
            data, missing = self._take(frames)
            position = self._consumed_samples
            self._consumed_samples += frames
            self.stats.consumed_samples = self._consumed_samples
            self.stats.callback_count += 1
            if self._last_callback_ns is not None:
                self.stats.max_callback_gap_ns = max(
                    self.stats.max_callback_gap_ns, now_ns - self._last_callback_ns
                )
            self._last_callback_ns = now_ns
            if status and bool(getattr(status, "output_underflow", False)):
                self.stats.device_underflow_count += 1
            starved = missing > 0 and not (
                self._producer_done and not self._pending
            )
            if starved:
                self.stats.starve_events += 1
                self.stats.starve_samples += missing

        if status and bool(getattr(status, "output_underflow", False)):
            self._emit(
                Stage.DEVICE_UNDERRUN,
                chunk_id=self._current_chunk(position),
                domain=ClockDomain.AUDIO_CALLBACK,
                timestamp_ns=now_ns,
                semantics=TimestampSemantics.DEVICE_CONSUME,
                attributes={"sample_position": position},
            )
        if starved:
            self._emit(
                Stage.PLAYBACK_STARVE,
                chunk_id=self._current_chunk(position),
                domain=ClockDomain.AUDIO_CALLBACK,
                timestamp_ns=now_ns,
                semantics=TimestampSemantics.EXECUTION_START,
                attributes={
                    "sample_position": position,
                    "missing_samples": missing,
                    "cause": "supply_shortage",
                    "note": "因缺数据而补零，不是自然静音",
                },
            )
        self._advance_spans(position)
        return data

    def _advance_spans(self, position: int) -> None:
        domain, semantics, note = self._clock_choice()
        attributes: dict[str, Any] = {}
        if semantics == TimestampSemantics.PROXY:
            attributes["proxy_note"] = note
        while self._spans:
            span = self._spans[0]
            if span.start_sample <= position:
                self._emit(
                    Stage.PLAYBACK_START,
                    chunk_id=span.chunk_id,
                    domain=domain,
                    timestamp_ns=time.monotonic_ns(),
                    semantics=semantics,
                    attributes={
                        **attributes,
                        "start_sample": span.start_sample,
                        "end_sample": span.end_sample,
                    },
                )
                self._spans.popleft()
                self.stats.played_sentences += 1
                continue
            break

    def _clock_choice(self) -> tuple[ClockDomain, TimestampSemantics, str]:
        if self.stats.used_device_clock:
            return (
                ClockDomain.AUDIO_DEVICE,
                TimestampSemantics.DEVICE_CONSUME,
                "",
            )
        return (
            ClockDomain.AUDIO_CALLBACK,
            TimestampSemantics.PROXY,
            "callback 触发时刻代理设备实际播放时刻，实际播放时间待校准",
        )

    def open(self) -> None:
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError(f"真实播放需要 sounddevice: {exc}") from exc
        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.blocksize,
            latency="low",
            device=self.device,
            callback=self._callback,
            finished_callback=self._finished.set,
        )
        self._stream.start()
        self.stats.backend = "sounddevice"
        self.stats.is_real = True
        self.stats.used_device_clock = self._probe_device_clock()
        if not self.stats.used_device_clock:
            self.stats.pending_notes.append(
                "未取得 outputBufferDacTime，实际播放时间为 callback 代理，"
                "零 underrun 验收保持 pending"
            )

    def _probe_device_clock(self) -> bool:
        stream = self._stream
        if stream is None:
            return False
        try:
            dac_time = stream.time
        except Exception:
            return False
        return isinstance(dac_time, (int, float))

    def calibrate_clock(self, *, samples: int = 5) -> ClockMapping | None:
        """用 host monotonic 与设备时钟估计偏移，给出误差上界。"""

        if self._stream is None or not self.stats.used_device_clock:
            return None
        offsets: list[int] = []
        for _ in range(max(1, samples)):
            host_ns = time.monotonic_ns()
            try:
                device_s = float(self._stream.time)
            except Exception:
                return None
            offsets.append(host_ns - int(device_s * 1_000_000_000))
            time.sleep(0.002)
        if not offsets:
            return None
        mapping = ClockMapping(
            source_domain=ClockDomain.AUDIO_DEVICE,
            target_domain=ClockDomain.HOST_MONOTONIC,
            offset_ns=int(sum(offsets) / len(offsets)),
            max_error_ns=max(offsets) - min(offsets),
            method="stream_time_vs_host_monotonic_roundtrip",
            sample_count=len(offsets),
        )
        if self.recorder is not None:
            self.recorder.add_mapping(mapping)
        return mapping

    def _callback(self, outdata: Any, frames: int, time_info: Any, status: Any) -> None:
        data = self._consume(frames, status, time_info)
        outdata.fill(0)
        if len(data) > 0:
            outdata[: len(data), 0] = data
        if self._producer_done and not self._pending and len(data) == 0:
            raise self._stop_exception()

    def _stop_exception(self) -> Exception:
        import sounddevice as sd

        return sd.CallbackStop()

    def mark_producer_done(self) -> None:
        with self._condition:
            self._producer_done = True
            self._condition.notify_all()

    def wait_finished(self, *, timeout_s: float = 60.0) -> bool:
        return bool(self._finished.wait(timeout_s))

    def close(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None
