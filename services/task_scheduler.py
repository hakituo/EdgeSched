"""统一全局任务调度器（L1 Router）。

结构说明（2026-08 解耦拆分）：
- 数据模型 TaskInfo/枚举 → services/task_models.py
- L2 执行器 PythonWorker → services/python_worker.py
- 图像队列/背压/生成路由 → services/scheduler_image_pipeline.py (ImagePipelineMixin)
- LLM 取消/回退/生成路由 → services/scheduler_llm_pipeline.py (LLMPipelineMixin)
- 本文件只保留核心：生命周期、任务调度、C++ 路由、恢复与查询

兼容性：原模块公开符号（TaskInfo/TaskPriority/TaskStatus/TaskType/PythonWorker/
get_current_task_id）在本文件顶部 re-export，外部导入路径无需修改。
"""
import asyncio
import os
import time
import uuid
from typing import Dict, Optional, Callable, Any, Union

from mvp_core.utils.trace_context import TraceContext
from mvp_core.utils.logger import logger
try:
    from mvp_core.config import get_settings
except Exception:
    from config import get_settings

try:
    from mvp_core.services.task_models import (
        TaskInfo,
        TaskPriority,
        TaskStatus,
        TaskType,
        get_current_task_id,
    )
    from mvp_core.services.python_worker import PythonWorker
    from mvp_core.services.scheduler_image_pipeline import ImagePipelineMixin
    from mvp_core.services.scheduler_llm_pipeline import LLMPipelineMixin
except ImportError:
    from services.task_models import (  # type: ignore
        TaskInfo,
        TaskPriority,
        TaskStatus,
        TaskType,
        get_current_task_id,
    )
    from services.python_worker import PythonWorker  # type: ignore
    from services.scheduler_image_pipeline import ImagePipelineMixin  # type: ignore
    from services.scheduler_llm_pipeline import LLMPipelineMixin  # type: ignore

__all__ = [
    "GlobalTaskScheduler",
    "TaskInfo",
    "TaskPriority",
    "TaskStatus",
    "TaskType",
    "PythonWorker",
    "get_current_task_id",
    "get_global_scheduler",
    "initialize_scheduler",
    "shutdown_scheduler",
]


class GlobalTaskScheduler(ImagePipelineMixin, LLMPipelineMixin):
    """
    Unified Global Task Scheduler (L1 Router)
    Routes tasks between PythonWorker (Legacy) and CPPSchedulerEngine (High-Performance).
    """
    def __init__(self):
        self._python_worker = PythonWorker()
        self._next_task_id = 0
        self._lock = asyncio.Lock() # For ID generation
        self._running = False
        self._start_lock = asyncio.Lock()
        self._background_tasks = set()
        self._fallback_llm_adapter = None
        self._fallback_llm_lock = asyncio.Lock()

        self._llm_cancel_events: Dict[str, asyncio.Event] = {}
        self._llm_cancel_lock = asyncio.Lock()

        self._image_queue = None
        self._image_workers_started = False
        self._image_workers_lock = asyncio.Lock()
        self._image_workers = set()
        self._image_total_slots = 0
        self._image_job_lock = asyncio.Lock()
        self._image_jobs: Dict[str, dict] = {}
        self._image_jobs_by_request: Dict[str, set] = {}

    async def _recover_tasks_from_backplane(self) -> None:
        sb = getattr(self._python_worker, "state_backplane", None)
        if sb is None:
            return

        try:
            active = sb.get_active_tasks()
        except Exception:
            return

        running_ids: list = []
        pending_items: list = []

        for t in active or []:
            try:
                tid = str(t.get("task_id", "") or "")
                st = int(t.get("status", 0) or 0)
            except Exception:
                continue
            if not tid:
                continue
            if st == 2:
                running_ids.append(tid)
            elif st == 1:
                pending_items.append(t)

        for tid in running_ids:
            try:
                sb.update_task_status(tid, 4)
            except Exception:
                pass

        async def _recovered_task_stub(task_id: str) -> None:
            raise RuntimeError(f"任务已从崩溃中恢复，但原始执行上下文已丢失: {task_id}")

        recovered_pending = 0
        for t in pending_items:
            tid = str(t.get("task_id", "") or "")
            if not tid:
                continue
            try:
                priority = TaskPriority(int(t.get("priority", TaskPriority.MEDIUM.value) or TaskPriority.MEDIUM.value))
            except Exception:
                priority = TaskPriority.MEDIUM

            task_info = TaskInfo(
                task_id=tid,
                name="recovered_task",
                priority=priority,
                task_type=TaskType.DEFAULT,
                created_at=time.time(),
                status=TaskStatus.PENDING,
                trace_id="system",
            )
            try:
                fut = await self._python_worker.submit_task(task_info, _recovered_task_stub, (tid,), {})
                try:
                    def _consume_future(f):
                        try:
                            f.exception()
                        except asyncio.CancelledError:
                            pass
                        except Exception:
                            pass

                    fut.add_done_callback(_consume_future)
                except Exception:
                    pass
                recovered_pending += 1
            except Exception:
                pass

        if running_ids or recovered_pending:
            logger.info(
                "Backplane 恢复完成: closed_running=%s, requeued_pending=%s",
                len(running_ids),
                recovered_pending,
            )

    async def cancel_request(self, request_id: str) -> None:
        await self.cancel_llm_request(request_id)
        await self.cancel_image_request(request_id)

    async def schedule_task(
        self,
        func: Callable,
        name: str = "unnamed_task",
        priority: Union[TaskPriority, int] = TaskPriority.MEDIUM,
        task_type: TaskType = TaskType.DEFAULT,
        args: tuple = (),
        kwargs: dict = None,
        trace_id: Optional[str] = None
    ) -> str:
        """Schedule a generic task"""
        if not self._running:
            raise RuntimeError("Scheduler not started")

        if kwargs is None:
            kwargs = {}

        if isinstance(priority, int):
            priority = TaskPriority(priority)

        task_id = f"task_{uuid.uuid4().hex[:8]}_{self._next_task_id}"
        self._next_task_id += 1

        if trace_id:
            current_trace = trace_id
        else:
            current_trace = TraceContext.get_trace_id()

        task_info = TaskInfo(
            task_id=task_id,
            name=name,
            priority=priority,
            task_type=task_type,
            created_at=time.time(),
            status=TaskStatus.PENDING,
            trace_id=current_trace
        )

        try:
            sb = getattr(self._python_worker, "state_backplane", None)
            if sb:
                sb.register_task(task_id, int(priority.value), 1)
        except Exception:
            pass

        # Route to PythonWorker (L1) for now as it handles generic tasks
        await self._python_worker.submit_task(task_info, func, args, kwargs)
        logger.debug(f"Task scheduled - ID: {task_id}, Name: {name} (Backend: Python)")
        return task_id

    async def schedule_cpu_task(self, func: Callable, *args, **kwargs) -> str:
        """Shortcut for CPU-bound tasks"""
        return await self.schedule_task(
            func,
            task_type=TaskType.CPU_BOUND,
            name=kwargs.pop('name', 'cpu_task'),
            priority=kwargs.pop('priority', TaskPriority.MEDIUM),
            args=args,
            kwargs=kwargs
        )

    async def schedule_gpu_task(self, func: Callable, *args, **kwargs) -> str:
        """Shortcut for GPU-bound generic tasks"""
        return await self.schedule_task(
            func,
            task_type=TaskType.GPU_BOUND,
            name=kwargs.pop('name', 'gpu_task'),
            priority=kwargs.pop('priority', TaskPriority.HIGH),
            args=args,
            kwargs=kwargs
        )

    async def start(self, worker_count: int = 3):
        """Start scheduler and engines"""
        async with self._start_lock:
            if self._running:
                logger.warning("Scheduler already running")
                return

            self._running = True

            # Start L1 Python Worker
            await self._python_worker.start(worker_count)
            try:
                await self._recover_tasks_from_backplane()
            except Exception:
                pass

            await self.ensure_image_workers_started()

            # Schedule cleanup task (Synced from Main Core)
            await self.schedule_periodic_task(
                func=self._python_worker.clean_completed_tasks,
                interval=300,  # Every 5 mins
                name="cleanup_completed_tasks",
                args=(3600,),  # 1 hour max age
            )

            # Start L2 CPP Engine if configured
            try:
                settings = get_settings()
                scheduler_settings = getattr(settings, "scheduler", None)
                model_settings = getattr(settings, "model", None)

                if scheduler_settings and model_settings:
                    use_cpp = bool(getattr(scheduler_settings, "use_cpp", False))
                    use_cpp_for_llm = bool(getattr(scheduler_settings, "use_cpp_for_llm", False))
                    use_cpp_for_image = bool(getattr(scheduler_settings, "use_cpp_for_image", False))

                    text_path = str(getattr(model_settings, "text_path", "") or "")
                    can_cpp_llm = use_cpp_for_llm and text_path.lower().endswith(".gguf")
                    should_start_cpp = bool(use_cpp) and bool(can_cpp_llm or use_cpp_for_image)

                    if should_start_cpp:
                        from mvp_core.services.cpp_scheduler_engine import cpp_scheduler_engine

                        if getattr(cpp_scheduler_engine, "enabled", False):
                            backend = getattr(scheduler_settings, "llm_backend", "python")
                            if backend not in ("cpp", "python"):
                                backend = "python"

                            gpu_config = {
                                "backend": backend,
                                "model_path": text_path if can_cpp_llm else "",
                                "model_type": "llama",
                                "gpu_device_id": 0,
                                "n_gpu_layers": int(getattr(model_settings, "n_gpu_layers", -1) or 0),
                                "max_context_size": int(getattr(model_settings, "n_ctx", 4096) or 4096),
                                "max_batch_size": 512,
                                "image_engine_type": str(
                                    getattr(scheduler_settings, "image_engine_type", "SD15_TURBO") or "SD15_TURBO"
                                ),
                            }

                            worker_count_cpp = int(getattr(scheduler_settings, "worker_count", 4) or 4)
                            preload_llm = bool(getattr(scheduler_settings, "preload_llm", False)) and bool(can_cpp_llm)

                            current_dir = os.path.dirname(os.path.abspath(__file__))
                            mvp_root = os.path.dirname(os.path.dirname(current_dir))
                            gpu_config["image_output_dir"] = os.path.join(mvp_root, "output", "images")

                            await asyncio.to_thread(
                                cpp_scheduler_engine.start,
                                worker_count=worker_count_cpp,
                                gpu_config=gpu_config,
                                preload_llm=preload_llm,
                            )
                            logger.info("C++ Scheduler Engine started via Router")
            except Exception as e:
                logger.warning(f"Failed to start C++ scheduler engine: {e}")

        logger.info("GlobalTaskScheduler (Router) started")

    async def stop(self, timeout: Optional[float] = None):
        """Stop scheduler"""
        async with self._start_lock:
            if not self._running:
                return

            self._running = False

            background_tasks = list(self._background_tasks)
            for task in background_tasks:
                try:
                    task.cancel()
                except Exception:
                    pass
            if background_tasks:
                try:
                    if timeout is not None:
                        await asyncio.wait_for(
                            asyncio.gather(*background_tasks, return_exceptions=True),
                            timeout=timeout,
                        )
                    else:
                        await asyncio.gather(*background_tasks, return_exceptions=True)
                except Exception:
                    pass

            # Stop L1
            await self._python_worker.stop()

            try:
                for task in list(self._image_workers):
                    try:
                        task.cancel()
                    except Exception:
                        pass
                self._image_workers.clear()
                self._image_workers_started = False
                self._image_queue = None
            except Exception:
                pass

            # Stop L2
            try:
                from mvp_core.services.cpp_scheduler_engine import cpp_scheduler_engine
                if getattr(cpp_scheduler_engine, "enabled", False):
                    await cpp_scheduler_engine.stop()
            except Exception:
                pass

        logger.info("GlobalTaskScheduler stopped")

    async def _route_to_cpp(self, task_type: str, *args, **kwargs) -> Optional[Any]:
        """Attempt to route to C++ engine. Returns None if not routed/failed."""
        try:
            settings = get_settings()
            scheduler_settings = getattr(settings, "scheduler", None)

            if not scheduler_settings:
                return None

            use_cpp = bool(getattr(scheduler_settings, "use_cpp", False))
            backend = str(getattr(scheduler_settings, "backend", "python") or "python").lower() # Check global backend preference

            # Support granular control as well
            if backend != "cpp" and not use_cpp:
                return None

            from mvp_core.services.cpp_scheduler_engine import cpp_scheduler_engine
            if not getattr(cpp_scheduler_engine, "enabled", False):
                return None

            # Route based on type
            if task_type == "image":
                 if getattr(scheduler_settings, "use_cpp_for_image", False):
                     # Extract prompt and kwargs
                     prompt = args[0] if args else kwargs.get("prompt", "")
                     return await cpp_scheduler_engine.submit_image_task(prompt, **kwargs)
            elif task_type == "llm":
                 if getattr(scheduler_settings, "use_cpp_for_llm", False):
                     prompt = args[0] if args else kwargs.get("prompt", "")
                     # LLM usually returns a generator, handling this might be complex if unification is needed
                     # For now, let's assume LLM routing is handled by specialized methods or callers use engine directly.
                     # But for this task, we focus on what's schedulable via 'schedule_task' which usually expects a result.
                     pass

        except Exception as e:
            logger.warning(f"Routing to C++ failed: {e}, falling back to Python")
            return None
        return None

    def get_task(self, task_id: str) -> Optional[TaskInfo]:
        # Currently only PythonWorker tasks are tracked here
        # TODO: Unified task tracking with CPP engine
        # For now, we delegate to PythonWorker
        if task_id in self._python_worker._tasks:
             return self._python_worker._tasks[task_id]['info']
        return None

    async def cancel_task(self, task_id: str) -> bool:
        return await self._python_worker.cancel_task(task_id)

    async def get_task_status(self, task_id: str) -> Optional[TaskInfo]:
        return await self._python_worker.get_task_info(task_id)

    async def get_task_info(self, task_id: str) -> Optional[TaskInfo]:
        return await self.get_task_status(task_id)

    async def get_task_future(self, task_id: str) -> Optional[asyncio.Future]:
        return await self._python_worker.get_task_future(task_id)

    async def get_all_tasks(self) -> Dict[str, TaskInfo]:
        async with self._python_worker._lock:
            return {k: v['info'] for k, v in self._python_worker._tasks.items()}

    async def get_active_tasks(self) -> Dict[str, TaskInfo]:
        result = {}
        async with self._python_worker._lock:
            for task_id, task_data in self._python_worker._tasks.items():
                if task_data['info'].status in [TaskStatus.PENDING, TaskStatus.RUNNING]:
                    result[task_id] = task_data['info']
        return result

    async def schedule_periodic_task(
        self,
        func: Callable,
        interval: float,
        name: str = "periodic_task",
        priority: Union[TaskPriority, int] = TaskPriority.LOW,
        args: tuple = (),
        kwargs: dict = None
    ) -> str:
        if kwargs is None:
            kwargs = {}
        async def periodic_wrapper():
            next_run = time.time() + interval
            while self._running:
                wait_time = max(0, next_run - time.time())
                try:
                    await asyncio.sleep(wait_time)
                except asyncio.CancelledError:
                    break
                if not self._running:
                    break
                next_run = time.time() + interval
                try:
                    await self.schedule_task(
                        func=func,
                        name=f"{name}_run",
                        priority=priority,
                        args=args,
                        kwargs=kwargs
                    )
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    if not self._running:
                        break
                    logger.error(f"Periodic task {name} submission failed: {str(e)}", exc_info=True)

        periodic_task = asyncio.create_task(periodic_wrapper())
        periodic_task_id = f"periodic_{uuid.uuid4().hex[:8]}"
        self._background_tasks.add(periodic_task)
        periodic_task.add_done_callback(self._background_tasks.discard)
        logger.info(f"Periodic task scheduled - Name: {name}, Interval: {interval}s")
        return periodic_task_id

    async def clean_completed_tasks(self, max_age: float = 3600) -> int:
        count = await self._python_worker.clean_completed_tasks(max_age)
        if count > 0:
            logger.info(f"Cleaned {count} completed tasks")
        return count

# Global instance
_global_scheduler: Optional[GlobalTaskScheduler] = None

def get_global_scheduler() -> GlobalTaskScheduler:
    global _global_scheduler
    if _global_scheduler is None:
        _global_scheduler = GlobalTaskScheduler()
    return _global_scheduler

async def initialize_scheduler():
    scheduler = get_global_scheduler()
    await scheduler.start()
    await scheduler.schedule_periodic_task(
        func=scheduler.clean_completed_tasks,
        interval=300,
        name="cleanup_completed_tasks",
        args=(3600,),
    )

async def shutdown_scheduler():
    global _global_scheduler
    if _global_scheduler is not None:
        await _global_scheduler.stop()
        _global_scheduler = None
