"""支持原子请求取消的 TTS 有界队列。"""

from __future__ import annotations

import asyncio
from collections import deque

from domain.models.tts import TTSChunkRequest
from services.tts.errors import QueueClosedError, TTSRequestCancelled


class BoundedTTSQueue:
    """使用 ``deque + asyncio.Condition`` 实现背压与指定请求移除。"""

    def __init__(self, maxsize: int) -> None:
        if maxsize < 1:
            raise ValueError("maxsize 必须大于 0")
        self.maxsize = maxsize
        self._items: deque[TTSChunkRequest] = deque()
        self._condition = asyncio.Condition()
        self._cancelled: set[tuple[str, int]] = set()
        self._closed = False

    @property
    def qsize(self) -> int:
        return len(self._items)

    async def put(self, item: TTSChunkRequest) -> None:
        async with self._condition:
            while len(self._items) >= self.maxsize and not self._closed:
                if item.request_key in self._cancelled:
                    raise TTSRequestCancelled(f"请求已取消: {item.request_key}")
                await self._condition.wait()
            if self._closed:
                raise QueueClosedError("TTS 队列已经关闭")
            if item.request_key in self._cancelled:
                raise TTSRequestCancelled(f"请求已取消: {item.request_key}")
            self._items.append(item)
            self._condition.notify_all()

    async def get(self) -> TTSChunkRequest:
        async with self._condition:
            while not self._items and not self._closed:
                await self._condition.wait()
            if not self._items:
                raise QueueClosedError("TTS 队列已经关闭")
            item = self._items.popleft()
            self._condition.notify_all()
            return item

    async def cancel(self, request_key: tuple[str, int]) -> list[TTSChunkRequest]:
        """原子标记取消并移除尚未派发的块。"""

        async with self._condition:
            self._cancelled.add(request_key)
            removed = [item for item in self._items if item.request_key == request_key]
            if removed:
                self._items = deque(
                    item for item in self._items if item.request_key != request_key
                )
            self._condition.notify_all()
            return removed

    async def close(self) -> None:
        async with self._condition:
            self._closed = True
            self._condition.notify_all()

    def is_cancelled(self, request_key: tuple[str, int]) -> bool:
        return request_key in self._cancelled
