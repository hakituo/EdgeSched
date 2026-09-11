"""按句追加的样本缓冲：取块、背压，以及两套样本下标的换算。

**两套下标必须分开**，否则启动补零之后所有归属都会错位：

- **生产者下标**（``start_sample``/``end_sample``）：句子在「已交付真实样本」
  流里的位置，从 0 开始，只随真实样本前进；
- **设备下标**：输出流的绝对消费位置，从 0 开始，**每个 callback 都前进**，
  补零也会让它前进。

二者之差就是「到目前为止补零了多少样本」。因此：

- 句子开始播放的条件是「已交付真实样本数越过该句起点」，而不是设备下标；
- 句子在设备下标上的起点可以精确换算：
  ``device_start = device_offset + start_sample``，其中
  ``device_offset = 本块设备起点 + 本块帧数 - 已交付真实样本数``。

断流与统计窗口都用设备下标，因为那是「播放时刻」的坐标系。
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any

import numpy as np

from experiments.elastic_residency.s0_pipeline.playback_stats import SentenceSpan

SpanTuple = tuple[int, int, int]


class SentenceBuffer:
    """句级样本缓冲；共享状态由调用方提供的条件变量保护。"""

    def __init__(
        self,
        *,
        sample_rate: int,
        max_pending_sentences: int,
        condition: threading.Condition,
    ) -> None:
        if max_pending_sentences < 1:
            raise ValueError("待播句数上限必须为正")
        self.sample_rate = sample_rate
        self.max_pending_sentences = max_pending_sentences
        self._condition = condition
        self._pending: deque[np.ndarray] = deque()
        self._offset = 0
        self._pending_spans: deque[SentenceSpan] = deque()
        self._producer_spans: list[SpanTuple] = []
        self._device_spans: list[SpanTuple] = []
        self._next_producer_sample = 0
        self.delivered_samples = 0

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def is_empty(self) -> bool:
        return not self._pending

    def producer_spans(self) -> list[SpanTuple]:
        """全部已提交句的生产者下标区间。"""

        return list(self._producer_spans)

    def device_spans(self) -> list[SpanTuple]:
        """**已开始播放**句的设备下标区间；未开播的句子不在这里。"""

        return list(self._device_spans)

    def append_blocking(
        self,
        chunk_id: int,
        array: np.ndarray,
        *,
        producer_done: bool,
    ) -> SentenceSpan:
        """追加一句；待播句数达到上限时阻塞，把背压传回合成侧。"""

        with self._condition:
            while (
                len(self._pending) >= self.max_pending_sentences
                and not producer_done
            ):
                self._condition.wait(timeout=1.0)
            start = self._next_producer_sample
            end = start + len(array)
            self._next_producer_sample = end
            self._pending.append(array)
            span = SentenceSpan(
                chunk_id=chunk_id,
                start_sample=start,
                end_sample=end,
                submit_ns=time.monotonic_ns(),
                audio_duration_s=len(array) / self.sample_rate,
            )
            self._pending_spans.append(span)
            self._producer_spans.append((span.chunk_id, start, end))
            self._condition.notify_all()
            return span

    def take(self, frames: int) -> tuple[np.ndarray, int]:
        """取最多 frames 个样本，返回 (数据, 因缺数据而补零的样本数)。"""

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
        self.delivered_samples += len(data)
        return data, remaining

    def pop_started(
        self,
        *,
        delivered_samples: int,
        block_producer_start: int,
        block_device_start: int,
    ) -> list[SentenceSpan]:
        """弹出本块内开始播放的句，并登记其设备下标区间。

        块内真实数据占据设备 ``[block_device_start, block_device_start+real_len)``，
        对应生产者 ``[block_producer_start, block_producer_start+real_len)``。
        若某句的生产者起点早于本块起点（它是在句尚未提交时被跳过、现在才
        真正出声的），它的设备起点就是本块真实数据的起点，而不是反推出来的
        更早位置。
        """

        started: list[SentenceSpan] = []
        while self._pending_spans:
            span = self._pending_spans[0]
            if span.start_sample >= delivered_samples:
                break
            self._pending_spans.popleft()
            device_start = block_device_start + max(
                0, span.start_sample - block_producer_start
            )
            self._device_spans.append(
                (
                    span.chunk_id,
                    device_start,
                    device_start + (span.end_sample - span.start_sample),
                )
            )
            started.append(span)
        return started

    def snapshot(self) -> dict[str, Any]:
        return {
            "pending_sentences": self.pending_count,
            "next_producer_sample": self._next_producer_sample,
            "delivered_samples": self.delivered_samples,
            "submitted_sentences": len(self._producer_spans),
        }
