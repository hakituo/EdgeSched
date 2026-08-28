"""GPU、DDR、CPU 与调度 jitter 的统一时间线采样。"""

from __future__ import annotations

import asyncio
import datetime as dt
import os
import statistics
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import psutil

from .contracts import ExperimentArtifactWriter, TimelineSample


def _wall_time_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


class NvidiaSmiProbe:
    """低频读取整卡与目标进程显存，失败时显式返回 unavailable。"""

    def __init__(
        self,
        command_runner: Callable[[list[str]], str] | None = None,
    ) -> None:
        self._run = command_runner or self._run_command

    @staticmethod
    def _run_command(command: list[str]) -> str:
        try:
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=2.0,
            )
        except (OSError, subprocess.SubprocessError):
            return ""
        return completed.stdout.strip()

    def sample(self, pid: int) -> dict[str, float | bool]:
        card_output = self._run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu",
                "--format=csv,noheader,nounits",
            ]
        )
        if not card_output:
            return {"available": False}
        try:
            card_parts = [
                float(value.strip())
                for value in card_output.splitlines()[0].split(",")
            ]
            if len(card_parts) != 5:
                return {"available": False}
        except ValueError:
            return {"available": False}
        process_output = self._run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ]
        )
        process_used_mb = 0.0
        for line in process_output.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 2:
                continue
            try:
                if int(parts[0]) == pid:
                    process_used_mb += float(parts[1])
            except ValueError:
                continue
        return {
            "available": True,
            "card_used_mb": card_parts[0],
            "card_total_mb": card_parts[1],
            "utilization_percent": card_parts[2],
            "power_w": card_parts[3],
            "temperature_c": card_parts[4],
            "process_used_mb": process_used_mb,
        }


class ResourceSampler:
    """后台线程采样资源，并用 deadline 偏差记录调度 jitter。"""

    def __init__(
        self,
        *,
        process_pid: int | None = None,
        interval_s: float = 0.05,
        gpu_interval_s: float = 0.25,
        writer: ExperimentArtifactWriter | None = None,
        gpu_probe: NvidiaSmiProbe | None = None,
        framework_gpu_probe: Callable[[], Mapping[str, float]] | None = None,
        thread_affinity: Sequence[int] | None = None,
    ) -> None:
        if interval_s <= 0 or gpu_interval_s < interval_s:
            raise ValueError("采样周期必须为正，且 GPU 周期不能小于基础周期")
        self.process_pid = process_pid or os.getpid()
        self.interval_s = interval_s
        self.gpu_interval_s = gpu_interval_s
        self.writer = writer
        self.gpu_probe = gpu_probe or NvidiaSmiProbe()
        self.framework_gpu_probe = framework_gpu_probe
        self.thread_affinity = (
            tuple(sorted(set(int(cpu_id) for cpu_id in thread_affinity)))
            if thread_affinity is not None
            else ()
        )
        if thread_affinity is not None and not self.thread_affinity:
            raise ValueError("thread_affinity 至少需要一个逻辑 CPU")
        self.samples: list[TimelineSample] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._gpu_thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._gpu_error: BaseException | None = None
        self._gpu_lock = threading.Lock()
        self._gpu_data: dict[str, float | bool] = {"available": False}
        self._gpu_sample_time = 0.0
        self._gpu_sample_count = 0
        self._sampler_thread_affinity: tuple[int, ...] = ()
        self._gpu_thread_affinity: tuple[int, ...] = ()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("ResourceSampler 已启动")
        self._stop_event.clear()
        self._error = None
        self._gpu_error = None
        self._gpu_thread = threading.Thread(
            target=self._run_gpu,
            name="edgesched-gpu-sampler",
            daemon=True,
        )
        self._thread = threading.Thread(
            target=self._run,
            name="edgesched-resource-sampler",
            daemon=True,
        )
        self._gpu_thread.start()
        self._thread.start()

    def stop(self, timeout_s: float = 5.0) -> dict[str, Any]:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout_s)
            if self._thread.is_alive():
                raise TimeoutError("ResourceSampler 未在期限内停止")
        if self._gpu_thread is not None:
            self._gpu_thread.join(timeout_s)
            if self._gpu_thread.is_alive():
                raise TimeoutError("GPU sampler 未在期限内停止")
        if self._error is not None:
            raise RuntimeError("ResourceSampler 后台失败") from self._error
        if self._gpu_error is not None:
            raise RuntimeError("GPU sampler 后台失败") from self._gpu_error
        jitters = [sample.scheduler_jitter_ms for sample in self.samples]
        return {
            "sample_count": len(self.samples),
            "interval_s": self.interval_s,
            "gpu_interval_s": self.gpu_interval_s,
            "gpu_sample_count": self._gpu_sample_count,
            "sampler_thread_affinity": list(self._sampler_thread_affinity),
            "gpu_thread_affinity": list(self._gpu_thread_affinity),
            "jitter_mean_ms": statistics.fmean(jitters) if jitters else 0.0,
            "jitter_p95_ms": _percentile(jitters, 0.95),
            "jitter_max_ms": max(jitters, default=0.0),
        }

    def __enter__(self) -> "ResourceSampler":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()

    def _run(self) -> None:
        try:
            self._sampler_thread_affinity = self._apply_current_thread_affinity()
            process = psutil.Process(self.process_pid)
            process.cpu_percent(interval=None)
            psutil.cpu_percent(interval=None, percpu=True)
            next_deadline = time.monotonic()
            while not self._stop_event.is_set():
                now = time.monotonic()
                wait_s = next_deadline - now
                if wait_s > 0 and self._stop_event.wait(wait_s):
                    break
                actual = time.monotonic()
                jitter_ms = max(0.0, (actual - next_deadline) * 1000.0)
                with self._gpu_lock:
                    gpu_data = dict(self._gpu_data)
                    gpu_sample_time = self._gpu_sample_time
                memory = psutil.virtual_memory()
                swap = psutil.swap_memory()
                try:
                    affinity = tuple(process.cpu_affinity())
                except (AttributeError, psutil.Error):
                    affinity = ()
                framework_gpu = (
                    dict(self.framework_gpu_probe())
                    if self.framework_gpu_probe is not None
                    else {}
                )
                sample = TimelineSample(
                    monotonic_ns=time.monotonic_ns(),
                    wall_time_iso=_wall_time_iso(),
                    scheduler_jitter_ms=jitter_ms,
                    process_pid=self.process_pid,
                    process_cpu_percent=process.cpu_percent(interval=None),
                    process_rss_bytes=process.memory_info().rss,
                    system_available_bytes=memory.available,
                    system_swap_used_bytes=swap.used,
                    per_cpu_percent=tuple(psutil.cpu_percent(interval=None, percpu=True)),
                    cpu_affinity=affinity,
                    gpu_available=bool(gpu_data.get("available", False)),
                    gpu_sample_age_ms=(
                        max(0.0, (time.monotonic() - gpu_sample_time) * 1000.0)
                        if gpu_sample_time
                        else 0.0
                    ),
                    gpu_card_used_mb=float(gpu_data.get("card_used_mb", 0.0)),
                    gpu_card_total_mb=float(gpu_data.get("card_total_mb", 0.0)),
                    gpu_utilization_percent=float(
                        gpu_data.get("utilization_percent", 0.0)
                    ),
                    gpu_power_w=float(gpu_data.get("power_w", 0.0)),
                    gpu_temperature_c=float(gpu_data.get("temperature_c", 0.0)),
                    gpu_process_used_mb=float(
                        gpu_data.get("process_used_mb", 0.0)
                    ),
                    framework_gpu=framework_gpu,
                )
                self.samples.append(sample)
                if self.writer is not None:
                    self.writer.append_timeline(sample)
                next_deadline += self.interval_s
                if next_deadline < actual:
                    missed = int((actual - next_deadline) / self.interval_s) + 1
                    next_deadline += missed * self.interval_s
        except BaseException as exc:
            self._error = exc

    def _run_gpu(self) -> None:
        try:
            self._gpu_thread_affinity = self._apply_current_thread_affinity()
            while not self._stop_event.is_set():
                sampled = self.gpu_probe.sample(self.process_pid)
                with self._gpu_lock:
                    self._gpu_data = sampled
                    self._gpu_sample_time = time.monotonic()
                    self._gpu_sample_count += 1
                if self._stop_event.wait(self.gpu_interval_s):
                    break
        except BaseException as exc:
            self._gpu_error = exc

    def _apply_current_thread_affinity(self) -> tuple[int, ...]:
        """Linux 下把当前采样线程放到系统预留核；其他平台显式回报未绑定。"""

        if not self.thread_affinity or not hasattr(os, "sched_setaffinity"):
            return ()
        os.sched_setaffinity(0, set(self.thread_affinity))
        return tuple(sorted(os.sched_getaffinity(0)))


class EventLoopJitterProbe:
    """独立测量 asyncio 调度 wake-up 延迟。"""

    def __init__(self, interval_s: float = 0.01) -> None:
        if interval_s <= 0:
            raise ValueError("interval_s 必须大于 0")
        self.interval_s = interval_s
        self.jitter_ms: list[float] = []

    async def run_for(self, duration_s: float) -> dict[str, float | int]:
        if duration_s <= 0:
            raise ValueError("duration_s 必须大于 0")
        self.jitter_ms.clear()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.interval_s
        end_time = loop.time() + duration_s
        while loop.time() < end_time:
            await asyncio.sleep(max(0.0, deadline - loop.time()))
            now = loop.time()
            self.jitter_ms.append(max(0.0, (now - deadline) * 1000.0))
            deadline += self.interval_s
            if deadline < now:
                missed = int((now - deadline) / self.interval_s) + 1
                deadline += missed * self.interval_s
        return {
            "sample_count": len(self.jitter_ms),
            "mean_ms": statistics.fmean(self.jitter_ms),
            "p95_ms": _percentile(self.jitter_ms, 0.95),
            "max_ms": max(self.jitter_ms),
        }
