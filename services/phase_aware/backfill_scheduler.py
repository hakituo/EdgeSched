"""
Resource Complementarity Backfilling

当 GPU 被 non-preemptible lease（Image）占用时，调度器主动 backfill CPU 阶段任务
（TTS、RAG、OCR、document parsing），从而隐藏 GPU 独占任务带来的等待时间。

核心思想：利用阶段级资源互补性
- GPU 任务（Image）和 CPU 任务（TTS/RAG）资源不冲突
- 可以并行执行，隐藏 GPU 独占任务的延迟

借鉴 HPC 调度的 backfilling 思想，但应用到边缘多模型 workflow。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Callable, Awaitable

from .phase_model import PhaseTask, PhaseRole, DeviceAffinity


@dataclass
class BackfillStats:
    """Backfill 统计（用于实验分析）"""
    total_submitted: int = 0
    total_executed: int = 0
    total_backfilled: int = 0          # 在 GPU 占用时 backfill 的数量
    total_backfill_time_hidden_s: float = 0.0  # backfill 隐藏的 GPU 等待时间
    max_concurrent: int = 0
    avg_wait_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_submitted": self.total_submitted,
            "total_executed": self.total_executed,
            "total_backfilled": self.total_backfilled,
            "total_backfill_time_hidden_s": self.total_backfill_time_hidden_s,
            "max_concurrent": self.max_concurrent,
            "avg_wait_s": self.avg_wait_s,
        }


class BackfillScheduler:
    """CPU 任务 Backfill 调度器

    管理 CPU-affine 任务队列，在 GPU 被占用时主动执行这些任务。

    核心机制：
    1. CPU 任务进入队列等待
    2. Worker 循环从队列拉取任务执行
    3. 如果 GPU 正被占用（burst 在跑），标记为 backfill 执行
    4. 记录 backfill 隐藏的等待时间
    """

    def __init__(self, max_concurrent_cpu: int = 2):
        self._max_concurrent = max_concurrent_cpu
        self._cpu_queue: asyncio.Queue = asyncio.Queue()
        self._running_tasks: Set[asyncio.Task] = set()
        self._workers: List[asyncio.Task] = []
        self._running = False
        self._start_lock = asyncio.Lock()

        # GPU 占用状态（由 phase_scheduler 更新）
        self._gpu_occupied: bool = False
        self._gpu_occupied_since: float = 0.0

        # 统计
        self._stats = BackfillStats()
        self._wait_times: List[float] = []
        self._completed_tasks: List[PhaseTask] = []

    async def start(self) -> None:
        """启动 backfill worker"""
        async with self._start_lock:
            if self._running:
                return
            self._running = True
            for i in range(self._max_concurrent):
                worker = asyncio.create_task(self._worker_loop(f"backfill-{i}"))
                self._workers.append(worker)

    async def stop(self) -> None:
        """停止 backfill worker"""
        self._running = False
        for w in self._workers:
            w.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

    def set_gpu_occupied(self, occupied: bool) -> None:
        """更新 GPU 占用状态（由 phase_scheduler 调用）"""
        if occupied and not self._gpu_occupied:
            self._gpu_occupied = True
            self._gpu_occupied_since = time.time()
        elif not occupied and self._gpu_occupied:
            self._gpu_occupied = False

    @property
    def is_gpu_occupied(self) -> bool:
        return self._gpu_occupied

    async def submit(self, task: PhaseTask) -> asyncio.Future:
        """提交 CPU 阶段任务

        Returns:
            Future，任务完成后可获取结果
        """
        self._stats.total_submitted += 1
        loop = asyncio.get_running_loop()
        future = loop.create_future()

        await self._cpu_queue.put((task, future, time.time()))
        return future

    async def _worker_loop(self, worker_name: str) -> None:
        """CPU 任务执行循环"""
        while self._running:
            try:
                task, future, enqueue_time = await asyncio.wait_for(
                    self._cpu_queue.get(), timeout=0.5
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            # 记录等待时间
            wait_s = time.time() - enqueue_time
            self._wait_times.append(wait_s)

            # 判断是否是 backfill（GPU 占用时执行）
            is_backfill_at_start = self._gpu_occupied
            gpu_occupied_duration_at_start = (
                time.time() - self._gpu_occupied_since if self._gpu_occupied else 0
            )

            task.start_time = time.time()
            task.status = "running"
            task.executed_on_device = "cpu"

            try:
                if task.execute_fn is not None:
                    result = await task.execute_fn()
                    task.result = result
                task.status = "completed"
                if not future.done():
                    future.set_result(task.result)
            except Exception as e:
                task.status = "failed"
                task.error = str(e)
                if not future.done():
                    future.set_exception(e)
            finally:
                task.end_time = time.time()
                task.actual_duration_s = task.end_time - task.start_time
                self._stats.total_executed += 1
                self._completed_tasks.append(task)

                # backfill 判定：任务执行期间 GPU 被占用过就算 backfill
                # 这包括"GPU 占用时开始"和"执行期间 GPU 被占用"两种情况
                was_backfill = is_backfill_at_start or self._gpu_occupied
                if was_backfill:
                    self._stats.total_backfilled += 1
                    # backfill 隐藏的 GPU 等待时间 = 任务执行时长
                    # （这部分时间用户不用干等 GPU，CPU 任务在并行）
                    self._stats.total_backfill_time_hidden_s += task.actual_duration_s

                # 更新最大并发
                self._stats.max_concurrent = max(
                    self._stats.max_concurrent,
                    len([t for t in self._workers if not t.done()]),
                )

    def get_stats(self) -> BackfillStats:
        """获取统计"""
        if self._wait_times:
            import statistics
            self._stats.avg_wait_s = statistics.mean(self._wait_times)
        return self._stats

    def get_completed_tasks(self) -> List[PhaseTask]:
        """获取已完成任务列表"""
        return list(self._completed_tasks)

    def get_queue_size(self) -> int:
        return self._cpu_queue.qsize()
