"""Qwen3-TTS Base 生产适配器。"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

from domain.interfaces.base_interfaces import TTSInterface
from domain.models.tts import TTSChunkRequest, TTSChunkResult, TTSOutputMode
from services.tts.errors import UnsupportedFeatureError
from services.tts.pipeline import TTSPipeline
from services.tts.qwen3_tts_worker import Qwen3TTSWorker
from services.tts.worker_protocol import DeviceTransitionMetrics, TTSWorkerConfig


class Qwen3TTSAdapter(TTSInterface):
    """真实 Base Worker 的生命周期与兼容接口门面。"""

    def __init__(
        self,
        config: TTSWorkerConfig,
        *,
        max_queue_size: int = 4,
        min_chars: int = 12,
        max_chars: int = 80,
    ) -> None:
        if config.backend_kind != "real_qwen3_tts":
            raise ValueError("Qwen3TTSAdapter 只允许 real_qwen3_tts 后端")
        self.worker = Qwen3TTSWorker(config)
        self.pipeline = TTSPipeline(
            self.worker,
            max_queue_size=max_queue_size,
            min_chars=min_chars,
            max_chars=max_chars,
        )

    async def start(self) -> None:
        await self.pipeline.start()

    async def stop(self) -> None:
        await self.pipeline.stop()

    async def synthesize_chunk(
        self,
        request: TTSChunkRequest,
        *,
        emotion: str | None = None,
    ) -> TTSChunkResult:
        return await self.pipeline.synthesize_chunk(request, emotion=emotion)

    async def stream_synthesize(
        self,
        request_id: str,
        token_stream: AsyncIterator[str],
        *,
        voice_profile_id: str,
        language: str,
        **kwargs,
    ) -> AsyncIterator[TTSChunkResult]:
        async for result in self.pipeline.stream_synthesize(
            request_id,
            token_stream,
            voice_profile_id=voice_profile_id,
            language=language,
            **kwargs,
        ):
            yield result

    async def synthesize(
        self,
        text: str,
        emotion: str | None = None,
        **kwargs,
    ) -> str:
        """兼容旧接口；仍走真实 Worker，不允许静默忽略 emotion。"""

        if emotion:
            raise UnsupportedFeatureError(
                "Qwen3-TTS Base 不支持 CustomVoice emotion/instruct"
            )
        profile_id = kwargs.get("voice_profile_id")
        language = kwargs.get("language", "Chinese")
        if not profile_id:
            raise ValueError("兼容 synthesize() 必须提供 voice_profile_id")
        request = TTSChunkRequest(
            request_id=kwargs.get("request_id", uuid.uuid4().hex),
            request_epoch=int(kwargs.get("request_epoch", 0)),
            chunk_id=0,
            raw_text=text,
            synthesis_text=text,
            language=language,
            voice_profile_id=profile_id,
            output_mode=TTSOutputMode.FILE,
            seed=int(kwargs.get("seed", 42)),
        )
        result = await self.synthesize_chunk(request)
        return result.wav_path

    async def cancel(self, request_id: str, **kwargs) -> None:
        await self.pipeline.cancel(
            request_id, request_epoch=int(kwargs.get("request_epoch", 0))
        )

    async def reload_on_device(self, target_device: str) -> DeviceTransitionMetrics:
        return await self.worker.reload_on_device(target_device)
