"""单 Worker、句子级流式 TTS 编排。"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from domain.models.tts import TTSChunkRequest, TTSChunkResult, TTSOutputMode
from services.tts.errors import (
    QueueClosedError,
    TTSRequestCancelled,
    UnsupportedFeatureError,
    WorkerProtocolError,
)
from services.tts.queue import BoundedTTSQueue
from services.tts.qwen3_tts_worker import Qwen3TTSWorker
from services.tts.sentence_splitter import DeterministicSentenceSplitter


_END = object()


@dataclass
class _RequestContext:
    request_id: str
    request_epoch: int
    outputs: asyncio.Queue[Any] = field(default_factory=asyncio.Queue)
    next_emit_id: int = 0
    pending_count: int = 0
    producer_done: bool = False
    cancelled: bool = False
    output_closed: bool = False
    reorder_buffer: dict[int, TTSChunkResult] = field(default_factory=dict)
    late_result_discarded: int = 0

    @property
    def key(self) -> tuple[str, int]:
        return self.request_id, self.request_epoch


class TTSPipeline:
    """把 token 流变成有背压、可取消且有序的句级音频结果流。"""

    def __init__(
        self,
        worker: Qwen3TTSWorker,
        *,
        max_queue_size: int = 4,
        min_chars: int = 12,
        max_chars: int = 80,
    ) -> None:
        self.worker = worker
        self.queue = BoundedTTSQueue(max_queue_size)
        self.min_chars = min_chars
        self.max_chars = max_chars
        self._contexts: dict[tuple[str, int], _RequestContext] = {}
        self._dispatcher: asyncio.Task[None] | None = None
        self._started = False
        self._context_lock = asyncio.Lock()

    async def start(self) -> None:
        if self._started:
            return
        await self.worker.start()
        self._started = True
        self._dispatcher = asyncio.create_task(
            self._dispatch_loop(), name="tts-pipeline-dispatcher"
        )

    async def stop(self) -> None:
        if not self._started:
            return
        await self.queue.close()
        if self._dispatcher is not None:
            await self._dispatcher
        await self.worker.stop()
        self._started = False

    async def synthesize_chunk(
        self,
        request: TTSChunkRequest,
        *,
        emotion: str | None = None,
    ) -> TTSChunkResult:
        if emotion:
            raise UnsupportedFeatureError(
                "Qwen3-TTS Base 不支持 CustomVoice emotion/instruct"
            )
        await self._ensure_started()
        context = await self._register_context(request.request_id, request.request_epoch)
        context.producer_done = True
        context.pending_count = 1
        try:
            await self.queue.put(request)
            output = await context.outputs.get()
            if isinstance(output, Exception):
                raise output
            if output is _END:
                raise TTSRequestCancelled(f"请求已取消: {context.key}")
            return output
        finally:
            await self._release_context_when_idle(context)

    async def stream_synthesize(
        self,
        request_id: str,
        token_stream: AsyncIterator[str],
        *,
        voice_profile_id: str,
        language: str,
        request_epoch: int = 0,
        emotion: str | None = None,
        output_mode: TTSOutputMode = TTSOutputMode.BOTH,
        seed: int = 42,
    ) -> AsyncIterator[TTSChunkResult]:
        if emotion:
            raise UnsupportedFeatureError(
                "Qwen3-TTS Base 不支持 CustomVoice emotion/instruct"
            )
        await self._ensure_started()
        context = await self._register_context(request_id, request_epoch)
        producer = asyncio.create_task(
            self._produce_chunks(
                context,
                token_stream,
                voice_profile_id=voice_profile_id,
                language=language,
                output_mode=output_mode,
                seed=seed,
            ),
            name=f"tts-producer-{request_id}-{request_epoch}",
        )
        try:
            while True:
                output = await context.outputs.get()
                if output is _END:
                    break
                if isinstance(output, Exception):
                    raise output
                yield output
        finally:
            if not producer.done():
                await self.cancel(request_id, request_epoch=request_epoch)
            await asyncio.gather(producer, return_exceptions=True)
            await self._release_context_when_idle(context)

    async def cancel(self, request_id: str, *, request_epoch: int = 0) -> int:
        key = (request_id, request_epoch)
        async with self._context_lock:
            context = self._contexts.get(key)
            if context is not None:
                context.cancelled = True
        removed = await self.queue.cancel(key)
        if context is None:
            return len(removed)
        context.pending_count = max(0, context.pending_count - len(removed))
        await self._close_output(context)
        await self._release_context_when_idle(context)
        return len(removed)

    async def _produce_chunks(
        self,
        context: _RequestContext,
        token_stream: AsyncIterator[str],
        *,
        voice_profile_id: str,
        language: str,
        output_mode: TTSOutputMode,
        seed: int,
    ) -> None:
        splitter = DeterministicSentenceSplitter(
            min_chars=self.min_chars, max_chars=self.max_chars
        )
        chunk_id = 0
        try:
            async for token in token_stream:
                for chunk in splitter.feed(token):
                    if chunk.synthesis_text:
                        await self._enqueue_chunk(
                            context,
                            chunk_id,
                            chunk.raw_text,
                            chunk.synthesis_text,
                            voice_profile_id,
                            language,
                            output_mode,
                            seed,
                        )
                        chunk_id += 1
            for chunk in splitter.finish():
                if chunk.synthesis_text:
                    await self._enqueue_chunk(
                        context,
                        chunk_id,
                        chunk.raw_text,
                        chunk.synthesis_text,
                        voice_profile_id,
                        language,
                        output_mode,
                        seed,
                    )
                    chunk_id += 1
        except TTSRequestCancelled:
            pass
        except Exception as exc:
            if not context.output_closed:
                await context.outputs.put(exc)
        finally:
            context.producer_done = True
            await self._maybe_finish(context)

    async def _enqueue_chunk(
        self,
        context: _RequestContext,
        chunk_id: int,
        raw_text: str,
        synthesis_text: str,
        voice_profile_id: str,
        language: str,
        output_mode: TTSOutputMode,
        seed: int,
    ) -> None:
        if context.cancelled:
            raise TTSRequestCancelled(f"请求已取消: {context.key}")
        request = TTSChunkRequest(
            request_id=context.request_id,
            request_epoch=context.request_epoch,
            chunk_id=chunk_id,
            raw_text=raw_text,
            synthesis_text=synthesis_text,
            language=language,
            voice_profile_id=voice_profile_id,
            output_mode=output_mode,
            seed=seed + chunk_id,
        )
        await self.queue.put(request)
        context.pending_count += 1

    async def _dispatch_loop(self) -> None:
        while True:
            try:
                request = await self.queue.get()
            except QueueClosedError:
                return

            context = self._contexts.get(request.request_key)
            if context is None or context.cancelled:
                if context is not None:
                    context.pending_count = max(0, context.pending_count - 1)
                    await self._maybe_finish(context)
                continue

            try:
                result = await self.worker.synthesize(request)
            except Exception as exc:
                context.pending_count = max(0, context.pending_count - 1)
                if not context.cancelled and not context.output_closed:
                    await context.outputs.put(exc)
                    await self._close_output(context)
                await self._maybe_finish(context)
                continue

            context.pending_count = max(0, context.pending_count - 1)
            if context.cancelled:
                context.late_result_discarded += 1
            else:
                await self._accept_ordered_result(context, result)
            await self._maybe_finish(context)

    async def _accept_ordered_result(
        self, context: _RequestContext, result: TTSChunkResult
    ) -> None:
        if result.chunk_id < context.next_emit_id:
            error = WorkerProtocolError(
                f"重复或倒退的 chunk_id: {result.chunk_id} < {context.next_emit_id}"
            )
            await context.outputs.put(error)
            await self._close_output(context)
            return
        if result.chunk_id > context.next_emit_id:
            if result.chunk_id in context.reorder_buffer:
                await context.outputs.put(
                    WorkerProtocolError(f"重复的乱序 chunk_id: {result.chunk_id}")
                )
                await self._close_output(context)
                return
            context.reorder_buffer[result.chunk_id] = result
            return

        await context.outputs.put(result)
        context.next_emit_id += 1
        while context.next_emit_id in context.reorder_buffer:
            buffered = context.reorder_buffer.pop(context.next_emit_id)
            await context.outputs.put(buffered)
            context.next_emit_id += 1

    async def _maybe_finish(self, context: _RequestContext) -> None:
        if context.producer_done and context.pending_count == 0:
            if context.reorder_buffer and not context.cancelled:
                await context.outputs.put(
                    WorkerProtocolError(
                        f"结果流结束时仍有乱序缺口: {sorted(context.reorder_buffer)}"
                    )
                )
            await self._close_output(context)
            await self._release_context_when_idle(context)

    async def _close_output(self, context: _RequestContext) -> None:
        if not context.output_closed:
            context.output_closed = True
            await context.outputs.put(_END)

    async def _register_context(
        self, request_id: str, request_epoch: int
    ) -> _RequestContext:
        key = (request_id, request_epoch)
        async with self._context_lock:
            if key in self._contexts:
                raise ValueError(f"请求键正在使用: {key}")
            context = _RequestContext(request_id=request_id, request_epoch=request_epoch)
            self._contexts[key] = context
            return context

    async def _release_context_when_idle(self, context: _RequestContext) -> None:
        if context.pending_count != 0:
            return
        async with self._context_lock:
            if self._contexts.get(context.key) is context:
                self._contexts.pop(context.key, None)

    async def _ensure_started(self) -> None:
        if not self._started:
            await self.start()
