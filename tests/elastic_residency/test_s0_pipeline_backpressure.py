"""S0 背压测试：token 桥接、TTS 队列与播放待播队列都必须真正有界。"""

import asyncio
import threading

import numpy as np

from experiments.elastic_residency.s0_pipeline.fake_sources import (
    ScriptedTokenSource,
)
from experiments.elastic_residency.s0_pipeline.playback import ContinuousPlayer


def test_token_bridge_is_bounded_and_blocks_upstream() -> None:
    source = ScriptedTokenSource(
        ["词"] * 80, bridge_maxsize=4, per_token_delay_s=0.001
    )
    source.start()

    async def consume() -> int:
        await asyncio.sleep(0.15)
        count = 0
        async for _ in source.stream():
            count += 1
            if count == 4:
                await asyncio.sleep(0.05)
        return count

    try:
        count = asyncio.run(consume())
    finally:
        source.close()

    assert count == 80
    assert source.stats.bridge_max_depth <= 4
    assert source.stats.bridge_blocked_count > 0, "背压未发生，桥接没有真正有界"


def test_token_source_failure_propagates_to_stream() -> None:
    source = ScriptedTokenSource(["词"] * 8, bridge_maxsize=4, fail_after=3)
    source.start()

    async def consume() -> int:
        count = 0
        async for _ in source.stream():
            count += 1
        return count

    try:
        with_assert = False
        try:
            asyncio.run(consume())
        except RuntimeError as exc:
            with_assert = "scripted token source failure" in str(exc)
        assert with_assert
    finally:
        source.close()


def test_playback_blocks_when_pending_sentences_exceed_limit() -> None:
    player = ContinuousPlayer(
        sample_rate=24000, blocksize=256, max_pending_sentences=2
    )
    player._submit_blocking(0, np.zeros(500, dtype=np.float32))
    player._submit_blocking(1, np.zeros(500, dtype=np.float32))
    finished = threading.Event()

    def submit_third() -> None:
        player._submit_blocking(2, np.zeros(500, dtype=np.float32))
        finished.set()

    worker = threading.Thread(target=submit_third, daemon=True)
    worker.start()
    assert not finished.wait(0.2), "第三句未被背压阻塞"
    player._consume(1000, None, None)
    assert finished.wait(2.0)
    worker.join(timeout=2.0)


def test_playback_starve_is_recorded_with_sample_position() -> None:
    player = ContinuousPlayer(sample_rate=24000, blocksize=100)
    player._submit_blocking(0, np.zeros(50, dtype=np.float32))
    player._consume(100, None, None)
    assert player.stats.starve_events == 1
    assert player.stats.starve_samples == 50
    assert player.stats.consumed_samples == 100


def test_natural_silence_is_not_counted_as_starve() -> None:
    player = ContinuousPlayer(sample_rate=24000, blocksize=100)
    player._submit_blocking(0, np.zeros(100, dtype=np.float32))
    player._consume(100, None, None)
    assert player.stats.starve_events == 0
    assert player.stats.consumed_samples == 100
