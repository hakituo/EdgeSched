"""
Phase-Aware Scheduler 主调度器

串联 phase_model / gpu_lease / cost_model / backfill_scheduler，
实现完整的多阶段 workflow 调度。

支持三种策略模式（用于实验对比）：
- SEQUENTIAL: 所有阶段串行执行（baseline）
- V1_DUAL_ROLE: 二元互斥（v1，非此即彼，来了就迁）
- V2_PHASE_AWARE: phase-aware + cost-aware + backfilling（v2）

核心决策流程（V2）：
1. burst 任务到达
2. 决策引擎判断是否迁移 LLM
3. 如需迁移：LLM GPU→CPU，获取 GPU 租约给 burst
4. 如不迁移：burst 排队等待，或 CPU-affine 任务旁路
5. GPU 被占用时，backfill CPU 任务（TTS/RAG/OCR）
6. burst 完成，释放租约，LLM 回迁 GPU
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Awaitable

from .phase_model import (
    PhaseTask,
    PhaseAttributes,
    PhaseRole,
    DeviceAffinity,
    Workflow,
    LLMState,
    LLMServiceLevel,
    PHASE_PRESETS,
    make_phase_task,
)
from .gpu_lease import GPULease, LeaseType, GPULeaseManager
from .cost_model import (
    MigrationCost,
    MigrationDecision,
    MigrationDecisionEngine,
    AlwaysMigrateEngine,
)
from .backfill_scheduler import BackfillScheduler


class SchedulerPolicy(Enum):
    """调度策略"""
    SEQUENTIAL = "sequential"            # 全串行 baseline
    V1_DUAL_ROLE = "v1_dual_role"        # v1：二元互斥，来了就迁
    V2_PHASE_AWARE = "v2_phase_aware"    # v2：phase-aware + cost-aware + backfill


class MigrationPolicy(Enum):
    """迁移策略（用于实验 1 对比）"""
    ALWAYS_MIGRATE = "always_migrate"    # 来了就迁
    COST_AWARE = "cost_aware"            # 代价感知


@dataclass
class SchedulerConfig:
    """调度器配置"""
    policy: SchedulerPolicy = SchedulerPolicy.V2_PHASE_AWARE
    migration_policy: MigrationPolicy = MigrationPolicy.COST_AWARE
    vram_total_mb: int = 8192
    max_concurrent_cpu: int = 2
    # 迁移代价（可由实验校准）
    migration_cost: MigrationCost = field(default_factory=MigrationCost)
    # 决策引擎参数
    short_burst_threshold_s: float = 2.0
    benefit_threshold_s: float = 0.5
    queue_pressure_threshold: int = 2


@dataclass
class ExecutionTrace:
    """执行轨迹（用于实验分析）"""
    task_id: str
    phase_name: str
    strategy: str
    submit_time: float
    start_time: float
    end_time: float
    duration_s: float
    executed_on_device: str
    waited_for_gpu: bool
    backfilled: bool
    decision: Optional[Dict[str, Any]] = None


class PhaseAwareScheduler:
    """Phase-Aware 主调度器

    使用方式：
        scheduler = PhaseAwareScheduler(config)
        await scheduler.start()
        # 提交任务
        await scheduler.submit_phase(task)
        # 或提交 workflow
        await scheduler.submit_workflow(workflow)
        await scheduler.stop()
    """

    def __init__(self, config: Optional[SchedulerConfig] = None):
        self.config = config or SchedulerConfig()

        # 核心组件
        self._lease_manager = GPULeaseManager(vram_total_mb=self.config.vram_total_mb)
        self._backfill = BackfillScheduler(max_concurrent_cpu=self.config.max_concurrent_cpu)

        # 迁移决策引擎
        # SEQUENTIAL 策略不需要迁移，但为接口统一仍创建一个
        if self.config.migration_policy == MigrationPolicy.ALWAYS_MIGRATE:
            self._decision_engine = AlwaysMigrateEngine(self.config.migration_cost)
        else:
            self._decision_engine = MigrationDecisionEngine(
                migration_cost=self.config.migration_cost,
                short_burst_threshold_s=self.config.short_burst_threshold_s,
                benefit_threshold_s=self.config.benefit_threshold_s,
                queue_pressure_threshold=self.config.queue_pressure_threshold,
            )

        # LLM 状态
        self._llm_state = LLMState(level=LLMServiceLevel.GPU_NORMAL, current_device="gpu")

        # GPU 任务队列（burst 任务排队等待）
        self._gpu_burst_queue: asyncio.Queue = asyncio.Queue()
        self._gpu_worker_running = False
        self._gpu_worker_task: Optional[asyncio.Task] = None

        # SEQUENTIAL 策略用的串行锁
        self._sequential_lock: Optional[asyncio.Lock] = None

        # 执行轨迹
        self._traces: List[ExecutionTrace] = []
        self._lock = asyncio.Lock()

        # 运行状态
        self._running = False

        # 迁移回调（实际迁移逻辑，可由外部注入；默认用 mock）
        self._migrate_llm_fn: Optional[Callable[[], Awaitable[None]]] = None
        self._restore_llm_fn: Optional[Callable[[], Awaitable[None]]] = None

    def set_migration_callbacks(
        self,
        migrate_fn: Optional[Callable[[], Awaitable[None]]] = None,
        restore_fn: Optional[Callable[[], Awaitable[None]]] = None,
    ) -> None:
        """设置实际迁移回调（对接真实 CPPSchedulerEngine）"""
        self._migrate_llm_fn = migrate_fn
        self._restore_llm_fn = restore_fn

    async def start(self) -> None:
        """启动调度器"""
        if self._running:
            return
        self._running = True

        # SEQUENTIAL 策略：不启动 backfill，用串行锁
        if self.config.policy == SchedulerPolicy.SEQUENTIAL:
            self._sequential_lock = asyncio.Lock()
        else:
            await self._backfill.start()

        # 启动 GPU burst worker（SEQUENTIAL 也用，保证 GPU 任务串行）
        self._gpu_worker_running = True
        self._gpu_worker_task = asyncio.create_task(self._gpu_burst_worker_loop())

        # 初始 LLM 持有 RESIDENT 租约
        resident_lease = GPULease(
            holder="llm_resident",
            holder_role=PhaseRole.RESIDENT_MIGRATABLE,
            lease_type=LeaseType.RESIDENT,
            preemptible=True,
            duration_pred_s=0.0,  # 常驻
            vram_required_mb=PHASE_PRESETS["llm"].vram_usage_mb,
        )
        await self._lease_manager.acquire(resident_lease)

    async def stop(self) -> None:
        """停止调度器"""
        self._running = False
        self._gpu_worker_running = False
        if self._gpu_worker_task:
            self._gpu_worker_task.cancel()
            try:
                await self._gpu_worker_task
            except asyncio.CancelledError:
                pass
        await self._backfill.stop()

    @property
    def llm_state(self) -> LLMState:
        return self._llm_state

    @property
    def decision_engine(self):
        return self._decision_engine

    @property
    def lease_manager(self) -> GPULeaseManager:
        return self._lease_manager

    @property
    def backfill_scheduler(self) -> BackfillScheduler:
        return self._backfill

    async def submit_phase(self, task: PhaseTask) -> Any:
        """提交单个阶段任务

        根据调度策略和任务属性路由：
        - RESIDENT_MIGRATABLE（LLM）：所有策略下都直接在 GPU 执行（常驻）
        - SEQUENTIAL: 其他任务串行执行（全局锁），不并行不 backfill
        - V1_DUAL_ROLE: CPU 任务也进 GPU burst 队列串行，不 backfill
        - V2_PHASE_AWARE:
          - CPU-affine 任务 → backfill scheduler
          - GPU_PREFERRED 任务 → 查询决策，bypass 则 backfill，否则 GPU 队列
          - GPU_REQUIRED 任务 → GPU burst 队列（触发迁移决策）
        """
        if task.attributes is None:
            raise ValueError("PhaseTask 必须有 attributes")

        attrs = task.attributes

        # RESIDENT_MIGRATABLE 任务（LLM）：所有策略下都直接在 GPU 执行
        # LLM 是常驻任务，不是 burst，不进 burst 队列，不迁移自己
        if attrs.role == PhaseRole.RESIDENT_MIGRATABLE:
            # SEQUENTIAL 策略下走串行路径（保持锁语义）
            if self.config.policy == SchedulerPolicy.SEQUENTIAL:
                async with self._sequential_lock:
                    return await self._execute_resident(task)
            return await self._execute_resident(task)

        # SEQUENTIAL 策略：所有任务串行执行，不并行不 backfill
        if self.config.policy == SchedulerPolicy.SEQUENTIAL:
            async with self._sequential_lock:
                return await self._execute_sequential(task)

        # V1_DUAL_ROLE 策略：CPU 任务也进 GPU burst 队列，不 backfill
        # 这样 GPU 任务和 CPU 任务都串行执行，体现 v1 的"非此即彼"特点
        if self.config.policy == SchedulerPolicy.V1_DUAL_ROLE:
            future = asyncio.get_running_loop().create_future()
            await self._gpu_burst_queue.put((task, future))
            return await future

        # V2_PHASE_AWARE 策略：phase-aware + cost-aware + backfill
        # CPU-affine 任务：走 backfill
        if task.is_cpu_affine:
            future = await self._backfill.submit(task)
            return await future

        # GPU_PREFERRED 任务：查询决策是否可降级 CPU
        if task.attributes.device_affinity == DeviceAffinity.GPU_PREFERRED:
            # 路由决策：不记录历史（避免与执行决策重复计数）
            decision = self._decision_engine.decide(
                burst_task=task,
                llm_state=self._llm_state,
                queue_depth=self._gpu_burst_queue.qsize(),
                record_history=False,
            )
            # cost-aware：GPU_PREFERRED 降级 CPU，走 backfill
            if decision.strategy == "bypass":
                future = await self._backfill.submit(task)
                return await future

        # GPU 任务：进入 burst 队列
        future = asyncio.get_running_loop().create_future()
        await self._gpu_burst_queue.put((task, future))
        return await future

    async def _execute_resident(self, task: PhaseTask) -> Any:
        """执行 RESIDENT_MIGRATABLE 任务（LLM 推理）

        LLM 常驻 GPU，直接执行推理，不迁移不获取 BURST 租约。
        如果 LLM 当前在 CPU（被前一个 burst 迁移了），需要先回迁 GPU。
        """
        submit_time = task.submit_time

        # 如果 LLM 当前在 CPU（被前一个 burst 迁移了），先回迁 GPU
        if self._llm_state.level == LLMServiceLevel.CPU_DEGRADED:
            await self._restore_llm_to_gpu()

        task.start_time = time.time()
        task.status = "running"
        task.executed_on_device = "gpu"

        try:
            if task.execute_fn is not None:
                result = await task.execute_fn()
                task.result = result
            task.status = "completed"
        except Exception as e:
            task.status = "failed"
            task.error = str(e)
            raise
        finally:
            task.end_time = time.time()
            task.actual_duration_s = task.end_time - task.start_time

            trace = ExecutionTrace(
                task_id=task.task_id,
                phase_name=task.phase_name,
                strategy="resident_gpu",
                submit_time=submit_time,
                start_time=task.start_time,
                end_time=task.end_time,
                duration_s=task.actual_duration_s,
                executed_on_device="gpu",
                waited_for_gpu=False,
                backfilled=False,
            )
            async with self._lock:
                self._traces.append(trace)

        return task.result

    async def _execute_sequential(self, task: PhaseTask) -> Any:
        """SEQUENTIAL 策略：直接串行执行任务，不并行不 backfill

        所有任务（GPU/CPU）都在当前协程中执行，
        用 _sequential_lock 保证一次只执行一个。

        GPU_REQUIRED 任务也需要迁移 LLM（释放显存），否则会 OOM。
        这与 v1/v2 的区别在于：sequential 不并行，不 backfill。
        """
        attrs = task.attributes
        submit_time = task.submit_time

        # GPU_REQUIRED 任务需要迁移 LLM 释放显存（物理约束）
        llm_was_on_gpu = self._llm_state.level == LLMServiceLevel.GPU_NORMAL
        migrated = False
        if attrs.device_affinity == DeviceAffinity.GPU_REQUIRED and llm_was_on_gpu:
            await self._migrate_llm_to_cpu()
            migrated = True

        task.start_time = time.time()
        task.status = "running"
        task.executed_on_device = "gpu" if task.needs_gpu else "cpu"

        try:
            if task.execute_fn is not None:
                result = await task.execute_fn()
                task.result = result
            task.status = "completed"
        except Exception as e:
            task.status = "failed"
            task.error = str(e)
            raise
        finally:
            task.end_time = time.time()
            task.actual_duration_s = task.end_time - task.start_time

            # 记录执行轨迹
            trace = ExecutionTrace(
                task_id=task.task_id,
                phase_name=task.phase_name,
                strategy="sequential",
                submit_time=submit_time,
                start_time=task.start_time,
                end_time=task.end_time,
                duration_s=task.actual_duration_s,
                executed_on_device=task.executed_on_device,
                waited_for_gpu=False,
                backfilled=False,
            )
            async with self._lock:
                self._traces.append(trace)

        # 回迁 LLM
        if migrated:
            await self._restore_llm_to_gpu()

        return task.result

    async def submit_workflow(self, workflow: Workflow) -> Dict[str, Any]:
        """提交完整 workflow

        根据 workflow 的阶段依赖关系执行。
        支持并行阶段（如 TTS + Image 同时）。
        """
        workflow.start_time = time.time()
        workflow.status = "running"

        results: Dict[str, Any] = {}
        # 按依赖关系分组：同组的可并行
        completed_phases: set = set()
        remaining = list(workflow.phases)

        while remaining:
            # 找出当前可执行的阶段（依赖已满足）
            ready: List[PhaseTask] = []
            not_ready: List[PhaseTask] = []
            for task in remaining:
                if all(dep in completed_phases for dep in task.depends_on):
                    ready.append(task)
                else:
                    not_ready.append(task)

            if not ready:
                # 死锁检测
                not_ready_names = [t.phase_name for t in not_ready]
                raise RuntimeError(f"workflow 依赖死锁: {not_ready_names}")

            remaining = not_ready

            # SEQUENTIAL 策略：即使有多个 ready 阶段也串行执行
            if self.config.policy == SchedulerPolicy.SEQUENTIAL:
                for task in ready:
                    result = await self.submit_phase(task)
                    results[task.phase_name] = result
                    completed_phases.add(task.phase_name)
            elif len(ready) == 1:
                # 单个阶段直接执行
                task = ready[0]
                result = await self.submit_phase(task)
                results[task.phase_name] = result
                completed_phases.add(task.phase_name)
            else:
                # 多个阶段并行
                tasks_coro = [self.submit_phase(t) for t in ready]
                done_results = await asyncio.gather(*tasks_coro, return_exceptions=True)
                for t, r in zip(ready, done_results):
                    if isinstance(r, Exception):
                        results[t.phase_name] = {"error": str(r)}
                    else:
                        results[t.phase_name] = r
                    completed_phases.add(t.phase_name)

        workflow.end_time = time.time()
        workflow.status = "completed"
        return {
            "workflow_id": workflow.workflow_id,
            "total_duration_s": workflow.total_duration_s,
            "phase_results": results,
        }

    async def _gpu_burst_worker_loop(self) -> None:
        """GPU burst 任务执行循环

        从队列拉取 burst 任务，决策是否迁移 LLM，执行任务。
        """
        while self._gpu_worker_running:
            try:
                task, future = await asyncio.wait_for(
                    self._gpu_burst_queue.get(), timeout=0.5
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            try:
                result = await self._execute_gpu_burst(task)
                if not future.done():
                    future.set_result(result)
            except Exception as e:
                if not future.done():
                    future.set_exception(e)

    async def _execute_gpu_burst(self, task: PhaseTask) -> Any:
        """执行一个 GPU burst 任务

        决策流程：
        1. V1_DUAL_ROLE 下 CPU 任务直接执行（不迁移不获取租约）
        2. 调用决策引擎判断是否迁移
        3. 如需迁移：执行 LLM GPU→CPU
        4. 获取 GPU 租约
        5. 执行 burst 任务
        6. 释放租约
        7. 回迁 LLM 到 GPU
        """
        attrs = task.attributes
        submit_time = task.submit_time

        # V1_DUAL_ROLE 下 CPU 任务也进了这个队列，直接执行不迁移
        if task.is_cpu_affine:
            task.start_time = time.time()
            task.status = "running"
            task.executed_on_device = "cpu"
            try:
                if task.execute_fn is not None:
                    result = await task.execute_fn()
                    task.result = result
                task.status = "completed"
            except Exception as e:
                task.status = "failed"
                task.error = str(e)
                raise
            finally:
                task.end_time = time.time()
                task.actual_duration_s = task.end_time - task.start_time
                trace = ExecutionTrace(
                    task_id=task.task_id,
                    phase_name=task.phase_name,
                    strategy="v1_serial_cpu",
                    submit_time=submit_time,
                    start_time=task.start_time,
                    end_time=task.end_time,
                    duration_s=task.actual_duration_s,
                    executed_on_device="cpu",
                    waited_for_gpu=True,
                    backfilled=False,
                )
                async with self._lock:
                    self._traces.append(trace)
            return task.result

        # 决策
        queue_depth = self._gpu_burst_queue.qsize()
        decision = self._decision_engine.decide(
            burst_task=task,
            llm_state=self._llm_state,
            queue_depth=queue_depth,
            llm_remaining_work_s=0.0,  # 简化：假设 LLM 空闲
        )

        # 记录决策前的状态
        llm_was_on_gpu = self._llm_state.level == LLMServiceLevel.GPU_NORMAL
        migrated = False

        if decision.should_migrate and llm_was_on_gpu:
            # 执行迁移
            await self._migrate_llm_to_cpu()
            migrated = True

        # 获取 GPU 租约
        burst_lease = GPULease(
            holder=task.task_id,
            holder_role=attrs.role,
            lease_type=LeaseType.BURST,
            preemptible=False,
            duration_pred_s=decision.predicted_burst_duration_s,
            vram_required_mb=attrs.vram_usage_mb,
        )

        acquired = await self._lease_manager.acquire(burst_lease)
        if not acquired:
            # 无法获取租约，等待
            await asyncio.sleep(0.1)
            # 重试
            acquired = await self._lease_manager.acquire(burst_lease)
            if not acquired:
                raise RuntimeError("无法获取 GPU 租约")

        # 标记 GPU 被占用，触发 backfill
        self._backfill.set_gpu_occupied(True)

        # 执行 burst 任务
        task.start_time = time.time()
        task.status = "running"
        task.executed_on_device = "gpu"

        try:
            if task.execute_fn is not None:
                result = await task.execute_fn()
                task.result = result
            task.status = "completed"
        except Exception as e:
            task.status = "failed"
            task.error = str(e)
            raise
        finally:
            task.end_time = time.time()
            task.actual_duration_s = task.end_time - task.start_time

            # 释放租约
            await self._lease_manager.release(burst_lease.lease_id)
            self._backfill.set_gpu_occupied(False)

            # 记录执行轨迹
            trace = ExecutionTrace(
                task_id=task.task_id,
                phase_name=task.phase_name,
                strategy=decision.strategy,
                submit_time=submit_time,
                start_time=task.start_time,
                end_time=task.end_time,
                duration_s=task.actual_duration_s,
                executed_on_device="gpu",
                waited_for_gpu=not migrated,
                backfilled=False,
                decision=decision.to_dict(),
            )
            async with self._lock:
                self._traces.append(trace)

            # 记录时长到决策引擎（用于未来预测）
            self._decision_engine.record_duration(task.phase_name, task.actual_duration_s)

        # 回迁 LLM
        if migrated:
            await self._restore_llm_to_gpu()

        return task.result

    async def _migrate_llm_to_cpu(self) -> None:
        """迁移 LLM 从 GPU 到 CPU"""
        self._llm_state.enter_degraded()

        if self._migrate_llm_fn is not None:
            await self._migrate_llm_fn()
        else:
            # Mock 迁移延迟（用于实验）
            cost = self.config.migration_cost
            await asyncio.sleep(cost.total_downlink_ms / 1000.0)

        # 释放 RESIDENT 租约
        resident = self._lease_manager.get_resident_lease()
        if resident is not None:
            await self._lease_manager.release(resident.lease_id)

    async def _restore_llm_to_gpu(self) -> None:
        """回迁 LLM 从 CPU 到 GPU"""
        self._llm_state.enter_restoring()

        if self._restore_llm_fn is not None:
            await self._restore_llm_fn()
        else:
            # Mock 回迁延迟
            cost = self.config.migration_cost
            await asyncio.sleep(cost.restore_ms / 1000.0)

        # 重新获取 RESIDENT 租约
        resident_lease = GPULease(
            holder="llm_resident",
            holder_role=PhaseRole.RESIDENT_MIGRATABLE,
            lease_type=LeaseType.RESIDENT,
            preemptible=True,
            duration_pred_s=0.0,
            vram_required_mb=PHASE_PRESETS["llm"].vram_usage_mb,
        )
        await self._lease_manager.acquire(resident_lease)
        self._llm_state.enter_normal()

    def get_traces(self) -> List[ExecutionTrace]:
        """获取所有执行轨迹"""
        return list(self._traces)

    def get_backfill_traces(self) -> List[ExecutionTrace]:
        """获取 backfill 执行轨迹"""
        backfill_tasks = self._backfill.get_completed_tasks()
        traces = []
        for task in backfill_tasks:
            traces.append(ExecutionTrace(
                task_id=task.task_id,
                phase_name=task.phase_name,
                strategy="backfill",
                submit_time=task.submit_time,
                start_time=task.start_time or 0,
                end_time=task.end_time or 0,
                duration_s=task.actual_duration_s,
                executed_on_device="cpu",
                waited_for_gpu=False,
                backfilled=True,
            ))
        return traces

    def get_all_traces(self) -> List[ExecutionTrace]:
        """获取所有轨迹（GPU + backfill）"""
        return self.get_traces() + self.get_backfill_traces()

    def get_stats(self) -> Dict[str, Any]:
        """获取完整统计"""
        all_traces = self.get_all_traces()
        gpu_traces = [t for t in all_traces if t.executed_on_device == "gpu"]
        cpu_traces = [t for t in all_traces if t.executed_on_device == "cpu"]

        # 端到端时间
        if all_traces:
            min_start = min(t.start_time for t in all_traces if t.start_time > 0)
            max_end = max(t.end_time for t in all_traces if t.end_time > 0)
            total_span_s = max_end - min_start
        else:
            total_span_s = 0.0

        return {
            "policy": self.config.policy.value,
            "migration_policy": self.config.migration_policy.value,
            "total_tasks": len(all_traces),
            "gpu_tasks": len(gpu_traces),
            "cpu_tasks": len(cpu_traces),
            "backfilled_tasks": len([t for t in cpu_traces if t.backfilled]),
            "total_span_s": total_span_s,
            "llm_migration_count": self._llm_state.migration_count,
            "llm_total_degraded_time_s": self._llm_state.total_degraded_time_s,
            "decision_stats": self._decision_engine.get_stats(),
            "backfill_stats": self._backfill.get_stats().to_dict(),
            "lease_snapshot": self._lease_manager.get_state_snapshot(),
        }


# ============================================================
# 便捷工厂函数
# ============================================================

def create_sequential_scheduler() -> PhaseAwareScheduler:
    """创建全串行调度器（baseline）"""
    config = SchedulerConfig(policy=SchedulerPolicy.SEQUENTIAL)
    return PhaseAwareScheduler(config)


def create_v1_scheduler() -> PhaseAwareScheduler:
    """创建 v1 二元互斥调度器（来了就迁）"""
    config = SchedulerConfig(
        policy=SchedulerPolicy.V1_DUAL_ROLE,
        migration_policy=MigrationPolicy.ALWAYS_MIGRATE,
    )
    return PhaseAwareScheduler(config)


def create_v2_scheduler() -> PhaseAwareScheduler:
    """创建 v2 phase-aware 调度器（cost-aware + backfill）"""
    config = SchedulerConfig(
        policy=SchedulerPolicy.V2_PHASE_AWARE,
        migration_policy=MigrationPolicy.COST_AWARE,
    )
    return PhaseAwareScheduler(config)
