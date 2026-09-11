"""播放测量口径测试：断流单位/阶段/窗口，以及时钟映射的验证判据。"""

import threading

import numpy as np

from experiments.elastic_residency.s0_pipeline.playback_buffer import (
    SentenceBuffer,
)
from experiments.elastic_residency.s0_pipeline.playback_clock import (
    calibrate_pairs,
)
from experiments.elastic_residency.s0_pipeline.playback_starvation import (
    PHASE_INTER_SENTENCE,
    PHASE_INTRA_SENTENCE,
    PHASE_STARTUP,
    PHASE_TAIL,
    StarvationTracker,
)

_SPANS = [(0, 0, 24000), (1, 24000, 48000)]


def test_contiguous_starvation_merges_into_one_episode() -> None:
    tracker = StarvationTracker()
    for start in (0, 1024, 2048):
        tracker.observe(
            sample_position=start,
            end_position=start + 1024,
            missing_samples=1024,
            spans=_SPANS,
            producer_done=False,
        )
    tracker.finalize(producer_done=False)
    summary = tracker.summary(window_start=0, window_end=48000)
    assert summary["starve_episode_count"] == 1
    assert summary["starve_callback_count"] == 3
    assert summary["starve_samples_total"] == 3072


def test_non_contiguous_starvation_is_two_episodes() -> None:
    tracker = StarvationTracker()
    for start, missing in ((0, 1024), (1024, 0), (2048, 1024)):
        tracker.observe(
            sample_position=start,
            end_position=start + 1024,
            missing_samples=missing,
            spans=_SPANS,
            producer_done=False,
        )
    tracker.finalize(producer_done=False)
    assert tracker.summary(window_start=0, window_end=48000)[
        "starve_episode_count"
    ] == 2


def _episode(**kwargs: object) -> object:
    tracker = StarvationTracker()
    tracker.observe(
        sample_position=int(kwargs["position"]),
        end_position=int(kwargs["position"]) + 1024,
        missing_samples=1024,
        spans=list(kwargs.get("spans", _SPANS)),  # type: ignore[arg-type]
        producer_done=bool(kwargs.get("producer_done", False)),
    )
    tracker.finalize(producer_done=bool(kwargs.get("producer_done", False)))
    return tracker.episodes()[0]


def test_startup_wait_is_classified_as_startup() -> None:
    # 首句尚未提交：归属必须用首次观测时的快照，不能被后来提交的句子改写
    first = _episode(position=0, spans=[])
    assert first.phase == PHASE_STARTUP  # type: ignore[attr-defined]
    assert first.chunk_id is None  # type: ignore[attr-defined]


def test_within_sentence_is_attributed_to_that_sentence() -> None:
    first = _episode(position=1000)
    assert first.phase == PHASE_INTRA_SENTENCE  # type: ignore[attr-defined]
    assert first.chunk_id == 0  # type: ignore[attr-defined]


def test_beyond_last_submitted_is_waiting_or_tail() -> None:
    waiting = _episode(position=48000, producer_done=False)
    assert waiting.phase == PHASE_INTER_SENTENCE  # type: ignore[attr-defined]
    assert waiting.chunk_id is None, "不可归属时保留 None"  # type: ignore[attr-defined]
    tail = _episode(position=48000, producer_done=True)
    assert tail.phase == PHASE_TAIL  # type: ignore[attr-defined]


def test_startup_is_excluded_from_window_ratio() -> None:
    tracker = StarvationTracker()
    tracker.observe(
        sample_position=0,
        end_position=1024,
        missing_samples=1024,
        spans=[],
        producer_done=False,
    )
    tracker.finalize(producer_done=False)
    summary = tracker.summary(window_start=0, window_end=48000)
    assert summary["starve_samples_total"] == 1024
    assert summary["by_phase"][PHASE_STARTUP]["samples"] == 1024
    # 启动等待落在窗口起点之前，不计入窗口内缺音
    assert summary["starve_samples_in_window"] == 0


def test_consistent_pairs_with_huge_offset_are_still_verified() -> None:
    # PortAudio 未规定流时间起点：偏移量巨大本身不是失效证据
    base = 1_700_000_000_000_000_000
    pairs = [(i * 10_000_000, base + i * 10_000_000) for i in range(8)]
    calibration = calibrate_pairs(pairs)
    assert calibration.verified is True
    assert calibration.offset_ns == base
    assert calibration.residual_max_ns == 0
    assert calibration.map_ns(base + 5) == 5


def test_unit_mismatch_fails_verification() -> None:
    pairs = [(i * 10_000_000, i * 5_000_000) for i in range(8)]
    calibration = calibrate_pairs(pairs)
    assert calibration.verified is False
    assert any("跨度比" in reason for reason in calibration.reasons)


def test_non_monotonic_device_time_fails_verification() -> None:
    pairs = [(i * 10_000_000, (8 - i) * 10_000_000) for i in range(8)]
    calibration = calibrate_pairs(pairs)
    assert calibration.verified is False
    assert any("递增" in reason for reason in calibration.reasons)


def test_large_residual_fails_verification() -> None:
    jitter = [0, 40_000_000, 0, 40_000_000, 0, 40_000_000, 0, 0]
    pairs = [
        (i * 10_000_000, i * 10_000_000 + jitter[i]) for i in range(8)
    ]
    calibration = calibrate_pairs(pairs)
    assert calibration.verified is False
    assert calibration.residual_max_ns > 5_000_000


def test_drift_is_detected_only_on_long_enough_span() -> None:
    # 2 s 跨度：漂移 2 ms/s，可判定
    pairs = [
        (i * 200_000_000, i * 200_000_000 + i * 400_000) for i in range(11)
    ]
    calibration = calibrate_pairs(pairs)
    assert calibration.drift_evaluated is True
    assert abs(calibration.drift_ns_per_s) > 1_000_000
    assert calibration.verified is False


def test_short_span_does_not_judge_drift() -> None:
    # 70 ms 跨度不足以估计漂移：只记录，不作为失效理由
    pairs = [(i * 10_000_000, i * 10_000_000) for i in range(8)]
    calibration = calibrate_pairs(pairs)
    assert calibration.drift_evaluated is False
    assert calibration.verified is True
    assert calibration.sample_cost_ns == 10_000_000


def _buffer() -> SentenceBuffer:
    return SentenceBuffer(
        sample_rate=1000, max_pending_sentences=4, condition=threading.Condition()
    )


def test_device_span_accounts_for_startup_padding() -> None:
    buf = _buffer()
    # 启动等待：缓冲还没有数据，设备先消费 300 个补零
    _, missing = buf.take(300)
    assert missing == 300
    assert buf.delivered_samples == 0
    buf.append_blocking(0, np.ones(1000, dtype=np.float32), producer_done=False)
    data, missing = buf.take(1200)
    started = buf.pop_started(
        delivered_samples=buf.delivered_samples,
        block_producer_start=buf.delivered_samples - (1200 - missing),
        block_device_start=300,
    )
    assert len(data) == 1000
    assert missing == 200
    assert [item.chunk_id for item in started] == [0]
    # 首句真实音频从设备位置 300 开始，不是 0，也不是被反推出来的更早位置
    assert buf.device_spans() == [(0, 300, 1300)]


def test_span_not_yet_started_is_absent_from_device_spans() -> None:
    buf = _buffer()
    buf.append_blocking(0, np.zeros(100, dtype=np.float32), producer_done=False)
    buf.append_blocking(1, np.zeros(100, dtype=np.float32), producer_done=False)
    buf.take(50)
    started = buf.pop_started(
        delivered_samples=buf.delivered_samples,
        block_producer_start=0,
        block_device_start=0,
    )
    assert [item.chunk_id for item in started] == [0]
    # 只交付了 50 个真实样本，第二句（生产者起点 100）还没开播
    assert buf.device_spans() == [(0, 0, 100)]


def test_span_submitted_after_gap_starts_where_it_really_plays() -> None:
    # 第二句在设备已经跑过一段时才提交：它的设备起点是真正出声的那一块，
    # 不能用「设备位置 - 已交付样本数」反推出一个更早的位置
    buf = _buffer()
    buf.append_blocking(0, np.ones(100, dtype=np.float32), producer_done=False)
    buf.take(100)
    started = buf.pop_started(
        delivered_samples=buf.delivered_samples,
        block_producer_start=0,
        block_device_start=0,
    )
    assert [item.chunk_id for item in started] == [0]
    _, missing = buf.take(200)  # 设备 [100,300) 全部补零
    assert missing == 200
    buf.append_blocking(1, np.ones(100, dtype=np.float32), producer_done=False)
    buf.take(100)  # 设备 [300,400) 是第二句的真实音频
    started = buf.pop_started(
        delivered_samples=buf.delivered_samples,
        block_producer_start=buf.delivered_samples - 100,
        block_device_start=300,
    )
    assert [item.chunk_id for item in started] == [1]
    assert buf.device_spans() == [(0, 0, 100), (1, 300, 400)]


def test_insufficient_samples_are_not_verified() -> None:
    pairs = [(i * 10_000_000, i * 10_000_000) for i in range(3)]
    calibration = calibrate_pairs(pairs)
    assert calibration.verified is False
    assert calibration.sample_count == 3
