"""同进程、逐句、真实 TTS 合成执行器。

与 ``services/tts/qwen3_tts_worker.py`` 的 spawn 子进程方案不同，本执行器
与 LLM 共享同一进程，从而沿用已冻结的共置放置、核租约与 direct NVML
采样口径。它只实现 ``TTSPipeline`` 所需的 ``synthesize`` 契约。

时间语义：提交到执行线程的时刻记为 ``submit``，线程内真正开始合成的时刻
记为 ``execution_start``，两者不允许混用。
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from pathlib import Path
from typing import Any

from domain.models.tts import TTSChunkRequest, TTSChunkResult, TTSOutputMode
from services.tts.wav_codec import pcm16_wav_bytes

from experiments.elastic_residency.s0_pipeline.events import (
    ClockDomain,
    EventRecorder,
    Stage,
    TimestampSemantics,
)


class SentenceSynthesizer:
    """把已放置好的 TTS 模型包装成逐句合成执行器。"""

    def __init__(
        self,
        *,
        model: Any,
        voice_prompt: Any,
        language: str,
        generation_kwargs: dict[str, Any],
        recorder: EventRecorder | None = None,
        output_dir: Path | None = None,
        device: str = "cuda:0",
        backend_kind: str = "qwen3_tts_local",
        queue_depth_provider: Any = None,
    ) -> None:
        self.model = model
        self.voice_prompt = voice_prompt
        self.language = language
        self.generation_kwargs = dict(generation_kwargs)
        self.recorder = recorder
        self.output_dir = output_dir
        self.device = device
        self.backend_kind = backend_kind
        self._queue_depth = queue_depth_provider

    def _emit(
        self,
        stage: Stage,
        *,
        chunk_id: int,
        timestamp_ns: int,
        semantics: TimestampSemantics,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        if self.recorder is None:
            return
        self.recorder.emit(
            stage,
            chunk_id=chunk_id,
            clock_domain=ClockDomain.HOST_MONOTONIC,
            timestamp_ns=timestamp_ns,
            semantics=semantics,
            attributes=attributes,
        )

    async def start(self) -> None:
        """同进程执行器无需启动子进程；仅为满足 pipeline 契约。"""

        return None

    async def stop(self) -> None:
        return None

    def _generate(self, text: str) -> tuple[Any, int]:
        """真实模型调用；测试替身只需覆盖本方法。"""

        return self.model.generate_voice_clone(
            text=text,
            language=self.language,
            voice_clone_prompt=self.voice_prompt,
            non_streaming_mode=True,
            **self.generation_kwargs,
        )

    async def synthesize(self, request: TTSChunkRequest) -> TTSChunkResult:
        submit_ns = time.monotonic_ns()
        self._emit(
            Stage.TTS_SUBMIT,
            chunk_id=request.chunk_id,
            timestamp_ns=submit_ns,
            semantics=TimestampSemantics.SUBMIT,
            attributes={
                "queue_depth": (
                    int(self._queue_depth()) if callable(self._queue_depth) else 0
                ),
                "submit_semantics_note": (
                    "提交到执行线程的时刻，不等于实际合成开始"
                ),
            },
        )
        return await asyncio.to_thread(self._synthesize_blocking, request, submit_ns)

    def _synthesize_blocking(
        self, request: TTSChunkRequest, submit_ns: int
    ) -> TTSChunkResult:
        start_ns = time.monotonic_ns()
        self._emit(
            Stage.TTS_EXEC_START,
            chunk_id=request.chunk_id,
            timestamp_ns=start_ns,
            semantics=TimestampSemantics.EXECUTION_START,
            attributes={"submit_to_start_ns": start_ns - submit_ns},
        )
        try:
            generated, sample_rate = self._generate(request.synthesis_text)
        except Exception as exc:
            end_ns = time.monotonic_ns()
            self._emit(
                Stage.TTS_EXEC_END,
                chunk_id=request.chunk_id,
                timestamp_ns=end_ns,
                semantics=TimestampSemantics.EXECUTION_END,
                attributes={"error": f"{type(exc).__name__}: {exc}"},
            )
            raise

        samples = generated[0]
        end_ns = time.monotonic_ns()
        audio_duration_s = float(len(samples)) / float(sample_rate)
        self._emit(
            Stage.TTS_EXEC_END,
            chunk_id=request.chunk_id,
            timestamp_ns=end_ns,
            semantics=TimestampSemantics.EXECUTION_END,
            attributes={
                "execution_ns": end_ns - start_ns,
                "audio_duration_s": audio_duration_s,
            },
        )

        serialization_started = time.perf_counter()
        audio_bytes = pcm16_wav_bytes(samples, sample_rate)
        serialization_s = time.perf_counter() - serialization_started
        wav_path = ""
        write_s = 0.0
        if request.output_mode in {TTSOutputMode.FILE, TTSOutputMode.BOTH}: 
            if self.output_dir is None:
                raise ValueError("FILE/BOTH 输出模式必须提供 output_dir")
            started = time.perf_counter()
            audio_dir = self.output_dir / "audio"
            audio_dir.mkdir(parents=True, exist_ok=True)
            target = audio_dir / f"{request.request_id}_{request.chunk_id:04d}.wav"
            target.write_bytes(audio_bytes)
            wav_path = str(target.resolve())
            write_s = time.perf_counter() - started

        self._emit(
            Stage.WAVEFORM_READY,
            chunk_id=request.chunk_id,
            timestamp_ns=time.monotonic_ns(),
            semantics=TimestampSemantics.DELIVERY,
            attributes={
                "audio_duration_s": audio_duration_s,
                "sample_rate": int(sample_rate),
                "audio_sha256": hashlib.sha256(audio_bytes).hexdigest(),
            },
        )
        synthesis_s = (end_ns - start_ns) / 1_000_000_000
        return TTSChunkResult(
            request_id=request.request_id,
            request_epoch=request.request_epoch,
            chunk_id=request.chunk_id,
            raw_text=request.raw_text,
            synthesis_text=request.synthesis_text,
            audio_bytes=(
                audio_bytes
                if request.output_mode in {TTSOutputMode.BYTES, TTSOutputMode.BOTH}
                else b""
            ),
            wav_path=wav_path,
            audio_sha256=hashlib.sha256(audio_bytes).hexdigest(),
            sample_rate=int(sample_rate),
            audio_duration_s=audio_duration_s,
            queue_wait_s=max(0.0, (start_ns - request.enqueue_time_ns) / 1_000_000_000),
            synthesis_s=synthesis_s,
            serialization_s=serialization_s,
            output_write_s=write_s,
            rtf=synthesis_s / audio_duration_s if audio_duration_s > 0 else 0.0,
            device=self.device,
            backend_kind=self.backend_kind,
        )
