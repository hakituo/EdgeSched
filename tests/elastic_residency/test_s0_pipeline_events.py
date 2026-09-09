"""S0 事件契约测试：生成时间与交付时间不得混用。"""

import pytest

from experiments.elastic_residency.s0_pipeline.events import (
    ClockDomain,
    ClockMapping,
    EventRecorder,
    ExecutionInterval,
    PipelineEvent,
    Stage,
    TimestampSemantics,
    execution_intervals,
    overlap_ns,
    semantics_coverage,
)


def test_proxy_semantics_requires_note() -> None:
    with pytest.raises(ValueError, match="proxy_note"):
        PipelineEvent(
            request_id="r1",
            stage=Stage.QUEUE_SNAPSHOT.value,
            timestamp_ns=1,
            timestamp_semantics=TimestampSemantics.PROXY.value,
        )


def test_device_consume_rejected_on_host_clock() -> None:
    with pytest.raises(ValueError, match="音频设备时钟域"):
        PipelineEvent(
            request_id="r1",
            stage=Stage.PLAYBACK_START.value,
            timestamp_ns=1,
            clock_domain=ClockDomain.HOST_MONOTONIC.value,
            timestamp_semantics=TimestampSemantics.DEVICE_CONSUME.value,
        )


def test_default_semantics_separate_submit_and_execution() -> None:
    recorder = EventRecorder(request_id="r1", config="K24:T0")
    recorder.emit(Stage.TTS_SUBMIT)
    recorder.emit(Stage.TTS_EXEC_START)
    recorder.emit(Stage.TTS_EXEC_END)
    events = recorder.snapshot()
    assert [item.timestamp_semantics for item in events] == [
        TimestampSemantics.SUBMIT.value,
        TimestampSemantics.EXECUTION_START.value,
        TimestampSemantics.EXECUTION_END.value,
    ]


def test_execution_intervals_ignore_submit_only_events() -> None:
    recorder = EventRecorder(request_id="r1", config="K24:T0")
    recorder.emit(Stage.TTS_SUBMIT, chunk_id=0, timestamp_ns=10)
    recorder.emit(Stage.TTS_SUBMIT, chunk_id=0, timestamp_ns=20)
    intervals = execution_intervals(
        recorder.snapshot(),
        start_stage=Stage.TTS_EXEC_START,
        end_stage=Stage.TTS_EXEC_END,
        key="tts_sentence",
    )
    assert intervals == []


def test_execution_intervals_and_overlap_detect_real_concurrency() -> None:
    recorder = EventRecorder(request_id="r1", config="K24:T0")
    recorder.emit(Stage.REQUEST_START, timestamp_ns=100)
    recorder.emit(Stage.TTS_EXEC_START, chunk_id=0, timestamp_ns=150)
    recorder.emit(Stage.TTS_EXEC_END, chunk_id=0, timestamp_ns=250)
    recorder.emit(Stage.REQUEST_END, timestamp_ns=400)
    events = recorder.snapshot()

    llm_intervals = execution_intervals(
        events,
        start_stage=Stage.REQUEST_START,
        end_stage=Stage.REQUEST_END,
        key="llm_generation",
    )
    tts_intervals = execution_intervals(
        events,
        start_stage=Stage.TTS_EXEC_START,
        end_stage=Stage.TTS_EXEC_END,
        key="tts_sentence",
    )
    assert len(llm_intervals) == 1 and len(tts_intervals) == 1
    assert overlap_ns(llm_intervals[0], tts_intervals[0]) == 100


def test_overlap_rejects_cross_clock_domain() -> None:
    left = ExecutionInterval(key="a", start_ns=0, end_ns=100, clock_domain="host_monotonic")
    right = ExecutionInterval(
        key="b", start_ns=50, end_ns=150, clock_domain=ClockDomain.AUDIO_DEVICE.value
    )
    with pytest.raises(ValueError, match="ClockMapping"):
        overlap_ns(left, right)


def test_clock_mapping_records_error_bound() -> None:
    mapping = ClockMapping(
        source_domain=ClockDomain.AUDIO_DEVICE,
        target_domain=ClockDomain.HOST_MONOTONIC,
        offset_ns=123,
        max_error_ns=7,
        method="stream_time_vs_host_monotonic_roundtrip",
        sample_count=5,
    )
    assert mapping.map_ns(1000) == 1123
    assert mapping.to_dict()["max_error_ns"] == 7
    with pytest.raises(ValueError):
        ClockMapping(
            source_domain=ClockDomain.AUDIO_DEVICE,
            target_domain=ClockDomain.HOST_MONOTONIC,
            offset_ns=0,
            max_error_ns=-1,
            method="",
        )


def test_semantics_coverage_flags_execution_evidence() -> None:
    recorder = EventRecorder(request_id="r1", config="K24:T0")
    recorder.emit(Stage.TOKEN_DELTA)
    recorder.emit(Stage.TTS_EXEC_START, chunk_id=0)
    recorder.emit(Stage.TTS_EXEC_END, chunk_id=0)
    coverage = semantics_coverage(recorder.snapshot())
    assert coverage["supports_execution_interval"] is True
    assert coverage["has_proxy_evidence"] is False
    assert coverage["semantics_counts"][TimestampSemantics.DELIVERY.value] == 1
