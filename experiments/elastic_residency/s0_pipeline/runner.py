"""G2：S0 流水线的集成编排。

只负责把 token 源、有界队列、句合成执行器和播放器串成一条真实依赖链，
并记录可解释事件；资源采样、门禁与产物写入由 ``s0_child`` 负责。

播放器为空时仍可完整跑通生成与合成链路，用于 Windows 侧链路验证。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from domain.models.tts import TTSOutputMode
from services.tts.pipeline import TTSPipeline
from services.tts.wav_codec import decode_pcm16_wav

from experiments.elastic_residency.s0_pipeline.clock import (
    measure_clock_resolution_ns,
)
from experiments.elastic_residency.s0_pipeline.config import S0PipelineConfig
from experiments.elastic_residency.s0_pipeline.events import (
    EventRecorder,
    Stage,
    TimestampSemantics,
    execution_intervals,
    semantics_coverage,
)
from experiments.elastic_residency.s0_pipeline.output_contract import (
    evaluate_text_contract,
)
from experiments.elastic_residency.s0_pipeline.startup import (
    StartupPlan,
    ensure_player_started,
)
from experiments.elastic_residency.s0_pipeline.text_contract import (
    StreamingTextNormalizer,
)


@dataclass
class PipelineRunResult:
    """一次流水线运行的完整可序列化结果。"""

    request_id: str
    placement: str
    finished: bool
    error: str = ""
    sentence_count: int = 0
    sentences: list[dict[str, Any]] = field(default_factory=list)
    token_stats: dict[str, Any] = field(default_factory=dict)
    playback_stats: dict[str, Any] = field(default_factory=dict)
    playback_starvation: dict[str, Any] = field(default_factory=dict)
    playback_calibration: dict[str, Any] | None = None
    tts_execution_intervals: list[dict[str, Any]] = field(default_factory=list)
    semantics: dict[str, Any] = field(default_factory=dict)
    clock_resolution_ns: int = 0
    text_contract: dict[str, Any] = field(default_factory=dict)
    startup: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    clock_mappings: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "placement": self.placement,
            "finished": self.finished,
            "error": self.error,
            "sentence_count": self.sentence_count,
            "sentences": self.sentences,
            "token_stats": self.token_stats,
            "playback_stats": self.playback_stats,
            "playback_starvation": self.playback_starvation,
            "playback_calibration": self.playback_calibration,
            "startup": self.startup,
            "tts_execution_intervals": self.tts_execution_intervals,
            "semantics": self.semantics,
            "clock_resolution_ns": self.clock_resolution_ns,
            "text_contract": self.text_contract,
            "clock_mappings": self.clock_mappings,
            # 事件必须落盘：只记 event_count 无法回溯首句就绪、提交与消费时间线
            "events": self.events,
            "event_count": len(self.events),
        }


class PipelineRunner:
    """执行 LLM 流式 → 分句 → 有界队列 → 逐句合成 → 连续播放。"""

    def __init__(
        self,
        *,
        config: S0PipelineConfig,
        token_source: Any,
        synthesizer: Any,
        recorder: EventRecorder,
        player: Any = None,
        request_id: str = "s0-request",
        voice_profile_id: str = "s0-default",
    ) -> None:
        self.config = config
        self.token_source = token_source
        self.synthesizer = synthesizer
        self.recorder = recorder
        self.player = player
        self.request_id = request_id
        self.voice_profile_id = voice_profile_id
        self.normalizer = StreamingTextNormalizer()
        self.startup = StartupPlan(policy=config.startup_policy)
        if getattr(self.synthesizer, "recorder", None) is None:
            self.synthesizer.recorder = self.recorder

    def _normalized_token_stream(self) -> Any:
        """输入适配层：token 进入分句器之前完成确定性格式规范化。"""

        async def gen() -> Any:
            async for token in self.token_source.stream():
                out = self.normalizer.feed(token)
                if out:
                    yield out
            tail = self.normalizer.finish()
            if tail:
                yield tail

        return gen()

    def _snapshot_attributes(self, pipeline: TTSPipeline) -> dict[str, Any]:
        return {
            "proxy_note": "周期性队列深度观测，不是事件发生时刻",
            "token_bridge_depth": int(
                getattr(self.token_source, "bridge_depth", 0)
            ),
            "tts_queue_depth": int(pipeline.queue.qsize),
            "tts_queue_maxsize": int(pipeline.queue.maxsize),
            "playback_pending_sentences": (
                int(self.player.pending_sentences) if self.player is not None else 0
            ),
        }

    async def _snapshot_loop(self, pipeline: TTSPipeline) -> None:
        while True:
            await asyncio.sleep(self.config.sample_interval_s)
            self.recorder.emit(
                Stage.QUEUE_SNAPSHOT,
                semantics=TimestampSemantics.PROXY,
                attributes=self._snapshot_attributes(pipeline),
            )

    async def run(self) -> PipelineRunResult:
        resolution_ns = measure_clock_resolution_ns()
        self.recorder.emit(
            Stage.REQUEST_START,
            attributes={
                "placement": self.config.placement,
                "clock_resolution_ns": resolution_ns,
            },
        )
        pipeline = TTSPipeline(
            worker=self.synthesizer,
            max_queue_size=self.config.tts_queue_maxsize,
            min_chars=self.config.min_chars,
            max_chars=self.config.max_chars,
        )
        snapshot_task = asyncio.create_task(self._snapshot_loop(pipeline))
        sentences: list[dict[str, Any]] = []
        error = ""
        try:
            if self.player is not None and not self.startup.is_prebuffer:
                # 首句即播：消费前打开输出流；启动等待落在统计窗口之外
                ensure_player_started(self.player)
            await pipeline.start()
            stream = pipeline.stream_synthesize(
                self.request_id,
                self._normalized_token_stream(),
                voice_profile_id=self.voice_profile_id,
                language=self.config.language,
                output_mode=TTSOutputMode.BYTES,
                seed=self.config.seed,
            )
            async for result in stream:
                sentence = {
                    "chunk_id": result.chunk_id,
                    "text": result.synthesis_text,
                    "raw_text": result.raw_text,
                    "audio_duration_s": result.audio_duration_s,
                    "queue_wait_s": result.queue_wait_s,
                    "synthesis_s": result.synthesis_s,
                    "rtf": result.rtf,
                    "audio_sha256": result.audio_sha256,
                }
                if self.player is not None:
                    samples, _ = decode_pcm16_wav(result.audio_bytes)
                    if self.startup.is_prebuffer:
                        # 全量预缓冲：先留在内存，等整段回复合成完再开播
                        self.startup.hold(
                            result.chunk_id, samples, result.audio_duration_s
                        )
                    else:
                        span = await self.player.submit(result.chunk_id, samples)
                        sentence["start_sample"] = span.start_sample
                        sentence["end_sample"] = span.end_sample
                sentences.append(sentence)
        except Exception as exc:  # noqa: BLE001 - 异常必须落盘而不是静默
            error = f"{type(exc).__name__}: {exc}"
            self.recorder.emit(
                Stage.CANCEL,
                attributes={"error": error, "stage": "pipeline_stream"},
            )
        finally:
            snapshot_task.cancel()
            if self.player is not None:
                # 预缓冲模式在这里才开播；首句即播模式 flush 为空操作
                await self.startup.flush(self.player, sentences)
                self.player.mark_producer_done()
                if self.player.is_opened:
                    self.player.wait_finished(timeout_s=120.0)
                # 先结算断流再取结果，否则统计与段落汇总会不一致
                self.player.finalize_starvation()
            await pipeline.stop()
            contract = evaluate_text_contract(
                normalizer=self.normalizer,
                sentences=[item["raw_text"] for item in sentences],
            )
            contract["synthesis_sentences"] = [
                item["text"] for item in sentences
            ]
            self.recorder.emit(
                Stage.REQUEST_END,
                attributes={
                    "sentence_count": len(sentences),
                    "error": error,
                    "output_contract_failed": bool(contract["failed"]),
                },
            )

        events = self.recorder.snapshot()
        return PipelineRunResult(
            request_id=self.request_id,
            placement=self.config.placement,
            finished=not error and not contract["failed"],
            error=error,
            sentence_count=len(sentences),
            sentences=sentences,
            token_stats=getattr(self.token_source, "stats", None).to_dict()
            if hasattr(self.token_source, "stats")
            else {},
            playback_stats=self.player.stats.to_dict()
            if self.player is not None
            else {"backend": "disabled", "is_real": False},
            playback_starvation=(
                self.player.starve_summary() if self.player is not None else {}
            ),
            playback_calibration=(
                self.player.calibration.to_dict()
                if self.player is not None and self.player.calibration is not None
                else None
            ),
            tts_execution_intervals=[
                item.to_dict()
                for item in execution_intervals(
                    events,
                    start_stage=Stage.TTS_EXEC_START,
                    end_stage=Stage.TTS_EXEC_END,
                    key="tts_sentence",
                )
            ],
            semantics=semantics_coverage(events),
            clock_resolution_ns=resolution_ns,
            text_contract=contract,
            startup=self.startup.evidence(),
            events=[item.to_dict() for item in events],
            clock_mappings=[item.to_dict() for item in self.recorder.mappings()],
        )
