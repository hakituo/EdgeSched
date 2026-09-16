"""启动策略测试：首句即播 vs 全量预缓冲必须共用同一播放路径与统计口径。"""

import asyncio

import numpy as np
import pytest

from experiments.elastic_residency.s0_pipeline.config import (
    STARTUP_FIRST_SENTENCE,
    STARTUP_FULL_PREBUFFER,
    S0PipelineConfig,
)
from experiments.elastic_residency.s0_pipeline.events import EventRecorder
from experiments.elastic_residency.s0_pipeline.fake_sources import (
    ScriptedSynthesizer,
    ScriptedTokenSource,
)
from experiments.elastic_residency.s0_pipeline.playback_stats import SentenceSpan
from experiments.elastic_residency.s0_pipeline.runner import PipelineRunner
from experiments.elastic_residency.s0_pipeline.startup import StartupPlan

_MULTI_SENTENCE = "第一句话在这里。第二句话在这里。最后一句话"


class _FakeStats:
    def to_dict(self) -> dict:
        return {"backend": "fake", "is_real": False, "device_underflow_count": 0}


class _FakePlayer:
    """记录调用顺序的最小播放器替身；不进任何真实结论。"""

    def __init__(self) -> None:
        self.is_opened = False
        self.open_count = 0
        self.calls: list[str] = []
        self.submitted: list[int] = []
        self.producer_done = False
        self.stats = _FakeStats()
        self.calibration = None

    def open(self) -> None:
        self.is_opened = True
        self.open_count += 1
        self.calls.append("open")

    def calibrate_clock(self):
        self.calls.append("calibrate")
        return None

    async def submit(self, chunk_id: int, samples: np.ndarray) -> SentenceSpan:
        self.submitted.append(chunk_id)
        self.calls.append(f"submit:{chunk_id}")
        return SentenceSpan(
            chunk_id=chunk_id,
            start_sample=0,
            end_sample=len(samples),
            submit_ns=0,
            audio_duration_s=len(samples) / 24000.0,
        )

    def mark_producer_done(self) -> None:
        self.producer_done = True

    def wait_finished(self, *, timeout_s: float = 60.0) -> bool:
        return True

    def finalize_starvation(self) -> dict:
        return {}

    def starve_summary(self) -> dict:
        return {}

    @property
    def pending_sentences(self) -> int:
        return 0


def _run(policy: str, player: _FakePlayer):
    recorder = EventRecorder(request_id="s0-startup", config="K24:T0")
    source = ScriptedTokenSource(list(_MULTI_SENTENCE), bridge_maxsize=16)
    config = S0PipelineConfig(
        placement="K24:T0",
        startup_policy=policy,
        tts_queue_maxsize=2,
        min_chars=4,
        max_chars=40,
        sample_interval_s=0.01,
    )
    runner = PipelineRunner(
        config=config,
        token_source=source,
        synthesizer=ScriptedSynthesizer(
            base_duration_s=0.02, per_char_duration_s=0.0, recorder=None
        ),
        recorder=recorder,
        player=player,
        request_id="s0-startup",
    )
    source.start()
    try:
        return asyncio.run(runner.run())
    finally:
        source.close()


def test_first_sentence_opens_before_consuming_and_submits_inline() -> None:
    player = _FakePlayer()
    result = _run(STARTUP_FIRST_SENTENCE, player)
    assert result.finished, result.error
    assert player.calls[0] == "open"
    assert player.submitted == [0, 1, 2]
    assert result.startup["policy"] == STARTUP_FIRST_SENTENCE
    assert result.startup["prebuffered_sentences"] == 0
    assert player.open_count == 1


def test_full_prebuffer_holds_all_audio_then_plays_once() -> None:
    player = _FakePlayer()
    result = _run(STARTUP_FULL_PREBUFFER, player)
    assert result.finished, result.error
    # 预缓冲：所有句子在开播后才提交，且只开一次流
    assert player.open_count == 1
    assert player.calls[0] == "open"
    assert player.submitted == [0, 1, 2]
    assert result.startup["policy"] == STARTUP_FULL_PREBUFFER
    assert result.startup["prebuffered_sentences"] == 3
    assert result.startup["prebuffered_audio_s"] > 0
    # 设备下标位置被回填，两种策略的句级证据结构一致
    assert all("start_sample" in item for item in result.sentences)


def test_startup_plan_rejects_unknown_policy() -> None:
    with pytest.raises(ValueError, match="启动策略"):
        StartupPlan(policy="no_such_policy")


def test_config_rejects_unknown_startup_policy() -> None:
    with pytest.raises(ValueError, match="启动策略"):
        S0PipelineConfig(placement="K24:T0", startup_policy="no_such_policy")


def test_startup_plan_flush_is_noop_for_first_sentence() -> None:
    plan = StartupPlan(policy=STARTUP_FIRST_SENTENCE)
    player = _FakePlayer()
    assert asyncio.run(plan.flush(player, [])) == 0
    assert player.is_opened is False
