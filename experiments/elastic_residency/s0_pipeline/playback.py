"""G1：句级连续播放器与断流记录。

两条断流口径严格分离：

- ``PLAYBACK_STARVE``：供给不足。回调需要样本但缓冲已耗尽，只能补零；
  补零样本数必须记录，不允许隐藏。
- ``DEVICE_UNDERRUN``：PortAudio 回调报告的 ``output_underflow``，属于设备侧。

自然静音不算断流：句内静音是真实样本，会被正常消费，不会产生 starve
事件；只有缓冲真的没有数据可给时才算供给不足。

时间口径：设备播放时刻取 callback 的 ``outputBufferDacTime``（缓冲首样本的
DAC 输出时刻），并必须与 host monotonic 通过配对采样**验证**后才能使用；
验证不通过时退回 callback 触发时刻作代理，语义标 ``proxy`` 并写明原因。
偏移量大小本身不是判据（PortAudio 未规定流时间起点）。

断流口径：缺供给的 callback 次数不等于独立断流次数，相邻缺音必须合并；
每段落再按启动/句间/句内/收尾/未知分阶段，窗口内外分开统计。
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

import numpy as np

from experiments.elastic_residency.s0_pipeline.events import (
    ClockDomain,
    ClockMapping,
    EventRecorder,
    Stage,
    TimestampSemantics,
)
from experiments.elastic_residency.s0_pipeline.playback_buffer import SentenceBuffer
from experiments.elastic_residency.s0_pipeline.playback_clock import (
    ClockCalibration,
    DeviceClockProbe,
)
from experiments.elastic_residency.s0_pipeline.playback_stats import (
    PlaybackStats,
    SentenceSpan,
    clock_choice,
    emit_event,
)
from experiments.elastic_residency.s0_pipeline.playback_starvation import (
    StarvationTracker,
)


def episode_chunk(tracker: StarvationTracker) -> int | None:
    """最近一段断流可归属的句子；不可归属时保留 None。"""

    episodes = tracker.episodes()
    return episodes[-1].chunk_id if episodes else None


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
        self._consumed_samples = 0
        self._producer_done = False
        self._finished = threading.Event()
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._buffer = SentenceBuffer(
            sample_rate=sample_rate,
            max_pending_sentences=max_pending_sentences,
            condition=self._condition,
        )
        self._stream: Any = None
        self._last_callback_ns: int | None = None
        self.tracker = StarvationTracker()
        self._probe = DeviceClockProbe()
        self._calibration: ClockCalibration | None = None

    @property
    def is_opened(self) -> bool:
        return self._stream is not None

    @property
    def pending_sentences(self) -> int:
        with self._lock:
            return self._buffer.pending_count

    @property
    def calibration(self) -> ClockCalibration | None:
        return self._calibration

    def span_tuples(self) -> list[tuple[int, int, int]]:
        """已开始播放句的**设备下标**区间；断流归属与窗口都用它。"""

        return self._buffer.device_spans()

    def starve_summary(self) -> dict[str, Any]:
        """断流汇总；窗口为「首句开播设备位置 → 末句结束设备位置」。"""

        spans = self.span_tuples()
        if not spans:
            window_start = window_end = None
        else:
            window_start = min(item[1] for item in spans)
            window_end = max(item[2] for item in spans)
        return self.tracker.summary(window_start=window_start, window_end=window_end)

    async def submit(self, chunk_id: int, samples: np.ndarray) -> SentenceSpan:
        """追加一句音频；待播放句数有界，从而把背压传回合成侧。"""

        array = np.asarray(samples, dtype=np.float32).reshape(-1)
        loop = asyncio.get_running_loop()
        span = await loop.run_in_executor(None, self._submit_blocking, chunk_id, array)
        emit_event(
            self.recorder,
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
        span = self._buffer.append_blocking(
            chunk_id, array, producer_done=self._producer_done
        )
        self.stats.submitted_sentences += 1
        return span

    def _consume(self, frames: int, status: Any, time_info: Any) -> np.ndarray:
        now_ns = time.monotonic_ns()
        with self._lock:
            data, missing = self._buffer.take(frames)
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

        # 先登记本块内开播的句，再判定断流归属：否则本块刚开始的句不在
        # 设备下标表里，断流会被误判成「在等尚未提交的下一句」。
        self._advance_spans(position, frames, missing)

        with self._lock:
            # 每一次 callback 都要上报：只在缺音时上报的话段落永远无法闭合。
            episode_id = self.tracker.observe(
                # 块内前 frames-missing 个样本是真实音频，缺音从其后开始；
                # 用块起点会把真实样本算进断流，也会把归属带偏一整块
                sample_position=position + frames - missing,
                end_position=position + frames,
                missing_samples=missing,
                spans=self._buffer.device_spans(),
                producer_done=self._producer_done,
            )
            self.stats.starve_events = len(self.tracker.episodes()) + (
                1 if self.tracker.has_open_episode() else 0
            )
            self.stats.starve_callback_count = self.tracker.callback_starve_count
            self.stats.starve_samples = self.tracker.missing_samples_total

        if status and bool(getattr(status, "output_underflow", False)):
            emit_event(
                self.recorder,
                Stage.DEVICE_UNDERRUN,
                chunk_id=episode_chunk(self.tracker),
                domain=ClockDomain.AUDIO_CALLBACK,
                timestamp_ns=now_ns,
                semantics=TimestampSemantics.DEVICE_CONSUME,
                attributes={"sample_position": position},
            )
        if missing > 0:
            emit_event(
                self.recorder,
                Stage.PLAYBACK_STARVE,
                chunk_id=episode_chunk(self.tracker),
                domain=ClockDomain.AUDIO_CALLBACK,
                timestamp_ns=now_ns,
                semantics=TimestampSemantics.EXECUTION_START,
                attributes={
                    "sample_position": position,
                    "missing_samples": missing,
                    "episode_id": episode_id,
                    "cause": "supply_shortage",
                    "note": "因缺数据而补零，不是自然静音",
                },
            )
        return data

    def _advance_spans(self, position: int, frames: int, missing: int) -> None:
        """登记本块内开始播放的句；开播判据是「已交付真实样本数」。

        设备下标与生产者下标之间差着「已补零样本数」，用设备下标直接比
        producer ``start_sample`` 会让首播事件在启动补零期间提前触发。
        """

        domain, semantics, note = self._clock_choice()
        attributes: dict[str, Any] = {}
        if semantics == TimestampSemantics.PROXY:
            attributes["proxy_note"] = note
        # 本块内真实数据占据设备区间 [position, position+frames-missing)，
        # 对应生产者区间 [delivered-real_len, delivered)，因此
        # 生产者下标 x 的设备位置 = position + (x - (delivered - real_len))
        delivered = self._buffer.delivered_samples
        real_len = frames - missing
        block_producer_start = delivered - real_len
        for span in self._buffer.pop_started(
            delivered_samples=delivered,
            block_producer_start=block_producer_start,
            block_device_start=position,
        ):
            device_start = position + max(0, span.start_sample - block_producer_start)
            emit_event(
                self.recorder,
                Stage.PLAYBACK_START,
                chunk_id=span.chunk_id,
                domain=domain,
                timestamp_ns=time.monotonic_ns(),
                semantics=semantics,
                attributes={
                    **attributes,
                    "start_sample": span.start_sample,
                    "end_sample": span.end_sample,
                    "device_start_sample": device_start,
                },
            )
            self.stats.played_sentences += 1

    def _clock_choice(self) -> tuple[ClockDomain, TimestampSemantics, str]:
        return clock_choice(self.stats.used_device_clock)

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
        self.stats.used_device_clock = False

    def calibrate_clock(
        self,
        *,
        samples: int = 32,
        min_span_s: float = 1.0,
        timeout_s: float = 10.0,
    ) -> ClockCalibration | None:
        """等待 callback 配对样本并验证映射；不按偏移量大小判定可用性。

        device 时间取 callback 的 ``outputBufferDacTime``（缓冲首样本的 DAC
        输出时刻），不是 ``currentTime``，也不是 ``stream.time``。
        验证不通过时保留 proxy 并给出原因，不生成映射。
        """

        if self._stream is None:
            return None
        calibration = self._probe.calibrate(
            samples=samples, min_span_s=min_span_s, timeout_s=timeout_s
        )
        self._calibration = calibration
        self.stats.used_device_clock = bool(calibration.verified)
        if not calibration.verified:
            # 未通过也要返回：残差/漂移/采样耗时是判断依据，必须落盘
            self.stats.pending_notes.append(
                "设备时钟映射未通过验证（"
                + "；".join(calibration.reasons)
                + "），实际播放时刻保持 callback 代理"
            )
            return calibration
        mapping = ClockMapping(
            source_domain=ClockDomain.AUDIO_DEVICE,
            target_domain=ClockDomain.HOST_MONOTONIC,
            offset_ns=calibration.offset_ns,
            max_error_ns=calibration.residual_max_ns,
            method=calibration.method,
            sample_count=calibration.sample_count,
            note=f"drift_ns_per_s={calibration.drift_ns_per_s:.3f}",
        )
        if self.recorder is not None:
            self.recorder.add_mapping(mapping)
        return calibration

    def _callback(self, outdata: Any, frames: int, time_info: Any, status: Any) -> None:
        self._probe.observe(time_info, time.monotonic_ns())
        data = self._consume(frames, status, time_info)
        outdata.fill(0)
        if len(data) > 0:
            outdata[: len(data), 0] = data
        if self._producer_done and self._buffer.is_empty and len(data) == 0:
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

    def finalize_starvation(self) -> dict[str, Any]:
        """结算断流段落并回写统计；幂等，可在取结果前与 close 时各调一次。"""

        self.tracker.finalize(producer_done=self._producer_done)
        summary = self.starve_summary()
        self.stats.starve_events = summary["starve_episode_count"]
        self.stats.starve_episode_count = summary["starve_episode_count"]
        self.stats.starve_samples = summary["starve_samples_total"]
        self.stats.starve_callback_count = self.tracker.callback_starve_count
        self.stats.starve_samples_in_window = summary["starve_samples_in_window"]
        return summary

    def close(self) -> None:
        self.finalize_starvation()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None
