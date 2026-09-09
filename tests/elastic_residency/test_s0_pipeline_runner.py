"""S0 链路测试：多句、保序、尾句 flush、真实执行区间、异常与取消清理。"""

import asyncio

from domain.models.tts import TTSOutputMode
from services.tts.pipeline import TTSPipeline

from experiments.elastic_residency.s0_pipeline.config import S0PipelineConfig
from experiments.elastic_residency.s0_pipeline.events import EventRecorder, Stage
from experiments.elastic_residency.s0_pipeline.fake_sources import (
    ScriptedSynthesizer,
    ScriptedTokenSource,
)
from experiments.elastic_residency.s0_pipeline.runner import PipelineRunner

_MULTI_SENTENCE = "第一句话在这里。第二句话在这里。最后一句话"


def _build(synthesizer: ScriptedSynthesizer, *, queue_size: int = 2):
    recorder = EventRecorder(request_id="s0-unit", config="K24:T0")
    source = ScriptedTokenSource(list(_MULTI_SENTENCE), bridge_maxsize=16)
    config = S0PipelineConfig(
        placement="K24:T0",
        tts_queue_maxsize=queue_size,
        min_chars=4,
        max_chars=40,
        sample_interval_s=0.01,
    )
    runner = PipelineRunner(
        config=config,
        token_source=source,
        synthesizer=synthesizer,
        recorder=recorder,
        player=None,
        request_id="s0-unit",
    )
    return runner, source


def test_pipeline_emits_ordered_sentences_and_flushes_tail() -> None:
    synthesizer = ScriptedSynthesizer(
        base_duration_s=0.01, per_char_duration_s=0.0, recorder=None
    )
    runner, source = _build(synthesizer)
    source.start()
    result = asyncio.run(runner.run())
    source.close()

    assert result.finished, result.error
    assert result.sentence_count == 3
    assert [item["chunk_id"] for item in result.sentences] == [0, 1, 2]
    assert result.sentences[-1]["text"] == "最后一句话"
    assert all(item["audio_duration_s"] > 0 for item in result.sentences)


def test_pipeline_records_real_tts_execution_intervals() -> None:
    synthesizer = ScriptedSynthesizer(
        base_duration_s=0.02,
        per_char_duration_s=0.0,
        synthesis_delay_s=0.01,
        recorder=None,
    )
    runner, source = _build(synthesizer)
    source.start()
    result = asyncio.run(runner.run())
    source.close()

    durations = [item["duration_ns"] for item in result.tts_execution_intervals]
    assert len(result.tts_execution_intervals) == 3
    assert max(durations) > 0
    assert all(duration >= 0 for duration in durations)
    # Windows 的 monotonic 粒度约 15.6 ms，短于一个 tick 的区间会测成 0；
    # 分辨率作为事实记录在 clock_resolution_ns 与 REQUEST_START 事件里，
    # 正式测量在 WSL2 执行（约 550 ns），不在测试里对 0 值做强解释断言
    assert result.clock_resolution_ns > 0
    assert result.semantics["supports_execution_interval"] is True
    assert result.semantics["execution_event_count"] >= 6


def test_synthesis_failure_closes_stream_and_records_error() -> None:
    synthesizer = ScriptedSynthesizer(
        base_duration_s=0.01,
        per_char_duration_s=0.0,
        fail_on_chunk=1,
        recorder=None,
    )
    runner, source = _build(synthesizer)
    source.start()
    result = asyncio.run(runner.run())
    source.close()

    assert result.finished is False
    assert "scripted synthesis failure" in result.error
    stages = {item["stage"] for item in result.events}
    assert Stage.CANCEL.value in stages
    assert Stage.REQUEST_END.value in stages


def test_cancel_drains_queue_without_pending_leak() -> None:
    recorder = EventRecorder(request_id="cancel-unit", config="K24:T0")
    synthesizer = ScriptedSynthesizer(
        base_duration_s=0.01, per_char_duration_s=0.0, recorder=None
    )
    source = ScriptedTokenSource(list("甲乙丙丁戊己庚辛壬癸") * 20, bridge_maxsize=4)
    pipeline = TTSPipeline(
        worker=synthesizer, max_queue_size=2, min_chars=4, max_chars=12
    )

    async def run() -> tuple[int, int]:
        await pipeline.start()
        received = 0
        async for _ in pipeline.stream_synthesize(
            "cancel-unit",
            source.stream(),
            voice_profile_id="calm",
            language="Chinese",
            output_mode=TTSOutputMode.BYTES,
            seed=42,
        ):
            received += 1
            if received == 1:
                await pipeline.cancel("cancel-unit")
        return received, pipeline.queue.qsize

    source.start()
    try:
        received, depth = asyncio.run(run())
    finally:
        source.close()

    assert received == 1
    assert depth == 0
    assert pipeline._contexts == {}
    assert recorder.snapshot() == []
