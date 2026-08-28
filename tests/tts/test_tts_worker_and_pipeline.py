import asyncio
import os
from pathlib import Path

import pytest

from domain.models.tts import (
    TTSChunkRequest,
    TTSChunkResult,
    TTSOutputMode,
    VoiceCloneProfile,
)
from services.tts.errors import UnsupportedFeatureError, WorkerProtocolError
from services.tts.pipeline import TTSPipeline, _RequestContext
from services.tts.qwen3_tts_worker import Qwen3TTSWorker
from services.tts.worker_protocol import TTSWorkerConfig, WorkerState


def _worker(tmp_path: Path, *, synthesis_s: float = 0.01) -> Qwen3TTSWorker:
    profile = VoiceCloneProfile(
        profile_id="fake-profile",
        ref_audio_path="not-used-by-fake.wav",
        ref_text="",
        language="Chinese",
        x_vector_only_mode=True,
    )
    return Qwen3TTSWorker(
        TTSWorkerConfig(
            model_path="not-used-by-fake",
            device="cpu",
            voice_profiles=(profile,),
            output_dir=str(tmp_path),
            backend_kind="test_fake_tts",
            fake_synthesis_s=synthesis_s,
        )
    )


def _request(chunk_id: int = 0) -> TTSChunkRequest:
    return TTSChunkRequest(
        request_id="spawn",
        request_epoch=0,
        chunk_id=chunk_id,
        raw_text=f"第{chunk_id}句。",
        synthesis_text=f"第{chunk_id}句。",
        language="Chinese",
        voice_profile_id="fake-profile",
        output_mode=TTSOutputMode.BOTH,
    )


def test_spawn_worker_protocol_and_profile_cache(tmp_path: Path) -> None:
    async def run() -> None:
        worker = _worker(tmp_path)
        health = await worker.start()
        assert health.state == WorkerState.READY
        assert health.model_load_count == 1
        assert health.voice_prompt_build_count == 1

        results = [await worker.synthesize(_request(index)) for index in range(10)]
        assert [result.chunk_id for result in results] == list(range(10))
        assert {result.worker_epoch for result in results} == {1}
        assert {result.model_load_count for result in results} == {1}
        assert {result.voice_prompt_build_count for result in results} == {1}
        assert {result.backend_kind for result in results} == {"test_fake_tts"}
        assert all(result.first_chunk_ready_s > 0 for result in results)
        assert all(result.audio_bytes.startswith(b"RIFF") for result in results)
        assert all(Path(result.wav_path).is_file() for result in results)
        await worker.stop()
        assert worker.health.state == WorkerState.STOPPED

    asyncio.run(run())


def test_missing_real_model_path_fails_without_mock_fallback(tmp_path: Path) -> None:
    async def run() -> None:
        profile = VoiceCloneProfile(
            profile_id="missing",
            ref_audio_path="missing.wav",
            ref_text="准确转写",
            language="Chinese",
        )
        worker = Qwen3TTSWorker(
            TTSWorkerConfig(
                model_path=str(tmp_path / "missing-model"),
                device="cpu",
                voice_profiles=(profile,),
                output_dir=str(tmp_path),
                backend_kind="real_qwen3_tts",
                start_timeout_s=10,
            )
        )
        with pytest.raises(WorkerProtocolError, match="找不到 Qwen3-TTS 模型"):
            await worker.start()
        assert worker.health.state == WorkerState.ERROR

    asyncio.run(run())


def test_device_reload_restarts_child_and_increments_epoch(tmp_path: Path) -> None:
    async def run() -> None:
        worker = _worker(tmp_path)
        first = await worker.start()
        metrics = await worker.reload_on_device("cpu")
        second = worker.health
        assert second.worker_pid != first.worker_pid
        assert second.worker_epoch == 2
        assert metrics.worker_epoch == 2
        assert metrics.total_transition_s >= metrics.process_stop_s
        await worker.stop()

    asyncio.run(run())


def test_linux_worker_reports_enforced_cpu_affinity(tmp_path: Path) -> None:
    if not hasattr(os, "sched_getaffinity"):
        pytest.skip("Windows 不提供 sched_getaffinity，权威绑核验证在 WSL 执行")

    async def run() -> None:
        profile = VoiceCloneProfile(
            profile_id="affinity-profile",
            ref_audio_path="not-used-by-fake.wav",
            ref_text="",
            language="Chinese",
            x_vector_only_mode=True,
        )
        worker = Qwen3TTSWorker(
            TTSWorkerConfig(
                model_path="not-used-by-fake",
                device="cpu",
                voice_profiles=(profile,),
                output_dir=str(tmp_path),
                cpu_threads=2,
                cpu_affinity=(0, 2),
                backend_kind="test_fake_tts",
            )
        )
        health = await worker.start()
        try:
            assert health.cpu_threads == 2
            assert health.cpu_affinity == (0, 2)
            result = await worker.synthesize(
                TTSChunkRequest(
                    request_id="affinity",
                    request_epoch=0,
                    chunk_id=0,
                    raw_text="绑核测试。",
                    synthesis_text="绑核测试。",
                    language="Chinese",
                    voice_profile_id="affinity-profile",
                )
            )
            assert result.cpu_affinity == (0, 2)
        finally:
            await worker.stop()

    asyncio.run(run())


def test_pipeline_cancellation_discards_running_and_removes_pending(
    tmp_path: Path,
) -> None:
    async def tokens():
        yield "第一句。第二句。第三句。"

    async def run() -> None:
        pipeline = TTSPipeline(
            _worker(tmp_path, synthesis_s=0.2),
            max_queue_size=4,
            min_chars=2,
            max_chars=20,
        )
        results: list[TTSChunkResult] = []

        async def consume() -> None:
            async for result in pipeline.stream_synthesize(
                "cancel",
                tokens(),
                voice_profile_id="fake-profile",
                language="Chinese",
            ):
                results.append(result)

        consumer = asyncio.create_task(consume())
        try:
            for _ in range(100):
                if (
                    ("cancel", 0) in pipeline._contexts
                    and pipeline.queue.qsize
                    and pipeline.worker.health.state == WorkerState.BUSY
                ):
                    break
                await asyncio.sleep(0.005)
            context = pipeline._contexts[("cancel", 0)]
            assert pipeline.worker.health.state == WorkerState.BUSY
            removed = await pipeline.cancel("cancel")
            await asyncio.wait_for(consumer, timeout=2)
            for _ in range(200):
                if context.pending_count == 0:
                    break
                await asyncio.sleep(0.01)
            assert removed >= 1
            assert results == []
            assert context.late_result_discarded == 1
        finally:
            await pipeline.stop()

    asyncio.run(run())


def test_reorder_buffer_emits_in_order_and_detects_duplicate(tmp_path: Path) -> None:
    async def run() -> None:
        pipeline = TTSPipeline(_worker(tmp_path))
        context = _RequestContext("order", 0)
        result_1 = TTSChunkResult("order", 0, 1, "二", "二")
        result_0 = TTSChunkResult("order", 0, 0, "一", "一")
        await pipeline._accept_ordered_result(context, result_1)
        assert context.outputs.empty()
        await pipeline._accept_ordered_result(context, result_0)
        assert (await context.outputs.get()).chunk_id == 0
        assert (await context.outputs.get()).chunk_id == 1
        await pipeline._accept_ordered_result(context, result_0)
        assert isinstance(await context.outputs.get(), WorkerProtocolError)

    asyncio.run(run())


def test_base_emotion_is_explicit_error(tmp_path: Path) -> None:
    async def run() -> None:
        pipeline = TTSPipeline(_worker(tmp_path))
        with pytest.raises(UnsupportedFeatureError):
            await pipeline.synthesize_chunk(_request(), emotion="happy")

    asyncio.run(run())
