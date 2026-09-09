"""Qwen3-TTS Base 的 spawn 子进程 Worker。"""

from __future__ import annotations

import asyncio
import hashlib
import io
import multiprocessing as mp
import os
import queue
import random
import sys
import time
import traceback
import wave
from dataclasses import replace
from pathlib import Path
from typing import Any

from domain.models.tts import (
    TTSChunkRequest,
    TTSChunkResult,
    TTSChunkStatus,
    TTSOutputMode,
)
from services.tts.errors import WorkerCrashedError, WorkerProtocolError
from services.tts.worker_protocol import (
    DeviceTransitionMetrics,
    TTSWorkerConfig,
    TTSWorkerHealth,
    WorkerState,
)
from services.tts.wav_codec import pcm16_wav_bytes


def _health_from_dict(data: dict[str, Any]) -> TTSWorkerHealth:
    copied = data.copy()
    copied["state"] = WorkerState(copied["state"])
    copied["cpu_affinity"] = tuple(copied.get("cpu_affinity", ()))
    return TTSWorkerHealth(**copied)


def _result_from_dict(data: dict[str, Any]) -> TTSChunkResult:
    copied = data.copy()
    copied["status"] = TTSChunkStatus(copied["status"])
    copied["cpu_affinity"] = tuple(copied.get("cpu_affinity", ()))
    return TTSChunkResult(**copied)


def _actual_cpu_affinity() -> tuple[int, ...]:
    if hasattr(os, "sched_getaffinity"):
        return tuple(sorted(os.sched_getaffinity(0)))
    return ()


def _pcm_wav_bytes(samples: Any, sample_rate: int) -> bytes:
    """将模型浮点输出转换为标准单声道 PCM16 WAV。"""

    return pcm16_wav_bytes(samples, sample_rate)


def _fake_wav_bytes(chunk_id: int, sample_rate: int = 24000) -> bytes:
    import numpy as np

    duration_s = 0.08
    count = int(sample_rate * duration_s)
    timeline = np.arange(count, dtype=np.float32) / sample_rate
    samples = 0.05 * np.sin(2 * np.pi * (220 + chunk_id) * timeline)
    return _pcm_wav_bytes(samples, sample_rate)


def _peak_rss_mb() -> float:
    try:
        import resource

        rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return rss / 1024.0 if sys.platform != "darwin" else rss / (1024.0**2)
    except Exception:
        return 0.0


def _write_audio(
    config: TTSWorkerConfig,
    request: TTSChunkRequest,
    audio_bytes: bytes,
) -> tuple[str, float]:
    if request.output_mode == TTSOutputMode.BYTES:
        return "", 0.0
    started = time.perf_counter()
    audio_dir = Path(config.output_dir) / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    path = audio_dir / (
        f"{request.request_id}_{request.request_epoch}_{request.chunk_id:04d}.wav"
    )
    path.write_bytes(audio_bytes)
    return str(path.resolve()), time.perf_counter() - started


def _prepare_runtime(config: TTSWorkerConfig) -> tuple[Any, dict[str, Any], dict[str, float]]:
    """仅在子进程中加载模型并构建所有音色 prompt。"""

    if config.cpu_threads > 0:
        os.environ["OMP_NUM_THREADS"] = str(config.cpu_threads)
        os.environ["MKL_NUM_THREADS"] = str(config.cpu_threads)
    if config.cpu_affinity and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(config.cpu_affinity))

    prompt_started = 0.0
    if config.backend_kind == "test_fake_tts":
        time.sleep(min(config.fake_synthesis_s, 0.05))
        prompts = {profile.profile_id: object() for profile in config.voice_profiles}
        return object(), prompts, {
            "model_load_s": min(config.fake_synthesis_s, 0.05),
            "voice_prompt_build_s": 0.0,
        }

    model_path = Path(config.model_path).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"找不到 Qwen3-TTS 模型: {model_path}")

    import torch
    from qwen_tts import Qwen3TTSModel

    if config.cpu_threads > 0:
        torch.set_num_threads(config.cpu_threads)
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("请求了 CUDA TTS Worker，但当前子进程不可用 CUDA")

    if config.dtype == "auto":
        dtype = torch.bfloat16 if config.device.startswith("cuda") else torch.float32
    else:
        dtype = getattr(torch, config.dtype)

    load_started = time.perf_counter()
    model = Qwen3TTSModel.from_pretrained(
        str(model_path),
        device_map=config.device,
        dtype=dtype,
        attn_implementation=config.attention_backend,
    )
    model_load_s = time.perf_counter() - load_started

    prompt_started = time.perf_counter()
    prompts: dict[str, Any] = {}
    for profile in config.voice_profiles:
        validated = profile.validate_audio()
        prompts[profile.profile_id] = model.create_voice_clone_prompt(
            ref_audio=validated.ref_audio_path,
            ref_text=validated.ref_text or None,
            x_vector_only_mode=validated.x_vector_only_mode,
        )
    return model, prompts, {
        "model_load_s": model_load_s,
        "voice_prompt_build_s": time.perf_counter() - prompt_started,
    }


def _synthesize_in_child(
    config: TTSWorkerConfig,
    model: Any,
    prompts: dict[str, Any],
    request: TTSChunkRequest,
    health: TTSWorkerHealth,
) -> TTSChunkResult:
    if request.voice_profile_id not in prompts:
        raise KeyError(f"未注册音色画像: {request.voice_profile_id}")

    random.seed(request.seed)
    queue_wait_s = max(
        0.0, (time.monotonic_ns() - request.enqueue_time_ns) / 1_000_000_000
    )
    started = time.perf_counter()
    peak_vram_mb = 0.0

    if config.backend_kind == "test_fake_tts":
        time.sleep(config.fake_synthesis_s)
        sample_rate = 24000
        audio_bytes = _fake_wav_bytes(request.chunk_id, sample_rate)
        audio_duration_s = 0.08
    else:
        import numpy as np
        import torch

        np.random.seed(request.seed)
        torch.manual_seed(request.seed)
        if config.device.startswith("cuda"):
            torch.cuda.manual_seed_all(request.seed)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        generated, sample_rate = model.generate_voice_clone(
            text=request.synthesis_text,
            language=request.language,
            voice_clone_prompt=prompts[request.voice_profile_id],
            non_streaming_mode=True,
            **config.generation_kwargs,
        )
        if config.device.startswith("cuda"):
            torch.cuda.synchronize()
            peak_vram_mb = torch.cuda.max_memory_allocated() / (1024.0**2)
        samples = generated[0]
        audio_duration_s = float(len(samples)) / float(sample_rate)
        serialization_started = time.perf_counter()
        audio_bytes = _pcm_wav_bytes(samples, sample_rate)
        serialization_s = time.perf_counter() - serialization_started

    synthesis_s = time.perf_counter() - started
    if config.backend_kind == "test_fake_tts":
        serialization_s = 0.0
    wav_path, output_write_s = _write_audio(config, request, audio_bytes)
    digest = hashlib.sha256(audio_bytes).hexdigest()
    returned_bytes = (
        audio_bytes
        if request.output_mode in {TTSOutputMode.BYTES, TTSOutputMode.BOTH}
        else b""
    )
    return TTSChunkResult(
        request_id=request.request_id,
        request_epoch=request.request_epoch,
        chunk_id=request.chunk_id,
        raw_text=request.raw_text,
        synthesis_text=request.synthesis_text,
        audio_bytes=returned_bytes,
        wav_path=wav_path,
        audio_sha256=digest,
        sample_rate=sample_rate,
        audio_duration_s=audio_duration_s,
        queue_wait_s=queue_wait_s,
        synthesis_s=synthesis_s,
        serialization_s=serialization_s,
        output_write_s=output_write_s,
        rtf=synthesis_s / audio_duration_s if audio_duration_s > 0 else 0.0,
        device=config.device,
        cpu_threads=config.cpu_threads,
        cpu_affinity=_actual_cpu_affinity(),
        backend_kind=config.backend_kind,
        worker_pid=os.getpid(),
        worker_epoch=config.worker_epoch,
        model_load_count=health.model_load_count,
        voice_prompt_build_count=health.voice_prompt_build_count,
        peak_rss_mb=_peak_rss_mb(),
        peak_vram_allocated_mb=peak_vram_mb,
    )


def _worker_process_main(
    config: TTSWorkerConfig,
    command_queue: Any,
    response_queue: Any,
) -> None:
    """spawn 入口；Torch 和模型不得泄漏到主进程。"""

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = (output_dir / "worker.log").open("a", encoding="utf-8", buffering=1)
    sys.stdout = log_file
    sys.stderr = log_file
    health = TTSWorkerHealth(
        state=WorkerState.STARTING,
        worker_pid=os.getpid(),
        worker_epoch=config.worker_epoch,
        device=config.device,
        backend_kind=config.backend_kind,
        cpu_threads=config.cpu_threads,
    )
    try:
        model, prompts, timings = _prepare_runtime(config)
        health.cpu_affinity = _actual_cpu_affinity()
        health.model_load_count = 1
        health.voice_prompt_build_count = len(prompts)
        health.model_load_s = timings["model_load_s"]
        health.voice_prompt_build_s = timings["voice_prompt_build_s"]
        health.state = WorkerState.READY
        response_queue.put({"kind": "ready", "health": health.to_dict()})

        while True:
            command = command_queue.get()
            kind = command.get("kind")
            if kind == "stop":
                health.state = WorkerState.STOPPING
                response_queue.put({"kind": "stopped", "health": health.to_dict()})
                return
            if kind != "synthesize":
                raise ValueError(f"未知 Worker 命令: {kind}")

            request = command["request"]
            health.state = WorkerState.BUSY
            try:
                result = _synthesize_in_child(config, model, prompts, request, health)
                health.state = WorkerState.READY
                response_queue.put(
                    {
                        "kind": "result",
                        "result": result.to_record() | {"audio_bytes": result.audio_bytes},
                        "health": health.to_dict(),
                    }
                )
            except Exception as exc:
                health.state = WorkerState.READY
                health.last_error = f"{type(exc).__name__}: {exc}"
                response_queue.put(
                    {
                        "kind": "request_error",
                        "error": health.last_error,
                        "traceback": traceback.format_exc(),
                        "health": health.to_dict(),
                    }
                )
    except Exception as exc:
        health.state = WorkerState.ERROR
        health.last_error = f"{type(exc).__name__}: {exc}"
        response_queue.put(
            {
                "kind": "boot_error",
                "error": health.last_error,
                "traceback": traceback.format_exc(),
                "health": health.to_dict(),
            }
        )
    finally:
        log_file.close()


class Qwen3TTSWorker:
    """主进程代理；子进程一次只执行一个 chunk。"""

    def __init__(self, config: TTSWorkerConfig) -> None:
        self.config = config
        self._context = mp.get_context("spawn")
        self._commands: Any | None = None
        self._responses: Any | None = None
        self._process: mp.Process | None = None
        self._operation_lock = asyncio.Lock()
        self._health = TTSWorkerHealth(
            state=WorkerState.STOPPED,
            worker_epoch=config.worker_epoch,
            device=config.device,
            backend_kind=config.backend_kind,
            cpu_threads=config.cpu_threads,
        )

    @property
    def health(self) -> TTSWorkerHealth:
        return replace(self._health)

    async def start(self) -> TTSWorkerHealth:
        if self._process is not None and self._process.is_alive():
            return self.health
        self._health.state = WorkerState.STARTING
        self._commands = self._context.Queue(maxsize=1)
        self._responses = self._context.Queue(maxsize=1)
        self._process = self._context.Process(
            target=_worker_process_main,
            args=(self.config, self._commands, self._responses),
            name=f"qwen3-tts-{self.config.worker_epoch}",
        )
        self._process.start()
        message = await self._receive(timeout_s=self.config.start_timeout_s)
        if message["kind"] == "boot_error":
            self._health = _health_from_dict(message["health"])
            await self._join_process()
            raise WorkerProtocolError(
                f"TTS Worker 启动失败: {message['error']}\n{message['traceback']}"
            )
        if message["kind"] != "ready":
            raise WorkerProtocolError(f"期待 ready，实际收到: {message['kind']}")
        self._health = _health_from_dict(message["health"])
        return self.health

    async def synthesize(self, request: TTSChunkRequest) -> TTSChunkResult:
        async with self._operation_lock:
            if self._health.state != WorkerState.READY:
                raise WorkerProtocolError(f"Worker 当前不可执行: {self._health.state.value}")
            if self._commands is None:
                raise WorkerCrashedError("Worker 命令队列不存在")
            request_started_ns = request.enqueue_time_ns
            self._health.state = WorkerState.BUSY
            await asyncio.to_thread(
                self._commands.put, {"kind": "synthesize", "request": request}
            )
            message = await self._receive(timeout_s=600.0)
            if "health" in message:
                self._health = _health_from_dict(message["health"])
            if message["kind"] == "request_error":
                raise WorkerProtocolError(
                    f"TTS 合成失败: {message['error']}\n{message['traceback']}"
                )
            if message["kind"] != "result":
                raise WorkerProtocolError(f"期待 result，实际收到: {message['kind']}")
            result = _result_from_dict(message["result"])
            result.first_chunk_ready_s = max(
                0.0, (time.monotonic_ns() - request_started_ns) / 1_000_000_000
            )
            return result

    async def stop(self) -> None:
        async with self._operation_lock:
            await self._stop_locked()

    async def reload_on_device(self, target_device: str) -> DeviceTransitionMetrics:
        """自然排空当前块后重启子进程，确保旧 CUDA context 被释放。"""

        requested_at = time.perf_counter()
        async with self._operation_lock:
            locked_at = time.perf_counter()
            source_device = self.config.device
            old_load_s = self._health.model_load_s
            old_prompt_s = self._health.voice_prompt_build_s
            stop_started = time.perf_counter()
            self._health.state = WorkerState.RELOADING
            await self._stop_locked()
            stopped_at = time.perf_counter()
            self.config = replace(
                self.config,
                device=target_device,
                worker_epoch=self.config.worker_epoch + 1,
            )
            ready_started = time.perf_counter()
            health = await self.start()
            ready_at = time.perf_counter()
            return DeviceTransitionMetrics(
                source_device=source_device,
                target_device=target_device,
                drain_s=locked_at - requested_at,
                process_stop_s=stopped_at - stop_started,
                model_load_s=health.model_load_s or old_load_s,
                voice_prompt_build_s=health.voice_prompt_build_s or old_prompt_s,
                ready_s=ready_at - ready_started,
                total_transition_s=ready_at - requested_at,
                worker_epoch=health.worker_epoch,
            )

    async def _stop_locked(self) -> None:
        process = self._process
        if process is None:
            self._health.state = WorkerState.STOPPED
            return
        if process.is_alive() and self._commands is not None:
            self._health.state = WorkerState.STOPPING
            await asyncio.to_thread(self._commands.put, {"kind": "stop"})
            try:
                await self._receive(timeout_s=self.config.stop_timeout_s)
            except (TimeoutError, WorkerCrashedError):
                process.terminate()
        await self._join_process()
        self._health.state = WorkerState.STOPPED

    async def _join_process(self) -> None:
        if self._process is not None:
            await asyncio.to_thread(self._process.join, self.config.stop_timeout_s)
            if self._process.is_alive():
                self._process.terminate()
                await asyncio.to_thread(self._process.join, 5.0)
        self._process = None
        self._commands = None
        self._responses = None

    async def _receive(self, *, timeout_s: float) -> dict[str, Any]:
        if self._responses is None:
            raise WorkerCrashedError("Worker 响应队列不存在")
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"等待 TTS Worker 超时: {timeout_s:.1f}s")
            try:
                return await asyncio.to_thread(
                    self._responses.get, True, min(0.5, remaining)
                )
            except queue.Empty:
                if self._process is None or not self._process.is_alive():
                    exit_code = self._process.exitcode if self._process else None
                    raise WorkerCrashedError(
                        f"TTS Worker 已退出，exit_code={exit_code}"
                    )
