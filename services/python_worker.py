"""Legacy Python Worker (L2 执行层)。

从 task_scheduler.py 拆出：封装 asyncio + ThreadPoolExecutor 逻辑，
作为 CPPSchedulerEngine 不可用时的回退执行器。
"""
import asyncio
import concurrent.futures
import time
from typing import Callable, Dict, List, Optional

from mvp_core.utils.trace_context import TraceContext
from mvp_core.utils.logger import logger

try:
    from mvp_core.services.task_models import (
        TaskInfo,
        TaskStatus,
        TaskType,
        set_current_task_id,
        reset_current_task_id,
    )
except ImportError:
    from services.task_models import (  # type: ignore
        TaskInfo,
        TaskStatus,
        TaskType,
        set_current_task_id,
        reset_current_task_id,
    )

try:
    from mvp_core.services.state_backplane import get_state_backplane
except ImportError:
    def get_state_backplane():
        return None


class PythonWorker:
    """
    Legacy Python Worker Implementation (L2 Execution Layer)
    Encapsulates the original asyncio + ThreadPoolExecutor logic.
    Serves as fallback executor when CPPSchedulerEngine is unavailable.
    """
    def __init__(self):
        self._tasks: Dict[str, Dict] = {}
        self._task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._worker_count: int = 3
        self._workers: List[asyncio.Task] = []
        self._running: bool = False
        self._lock: asyncio.Lock = asyncio.Lock()
        self._start_lock: asyncio.Lock = asyncio.Lock()
        self._task_futures: Dict[str, asyncio.Future] = {}

        # Resource management
        self._gpu_lock = asyncio.Lock()
        self._cpu_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

        self.state_backplane = get_state_backplane()

    async def start(self, worker_count: int = 3):
        async with self._start_lock:
            if self._running:
                return

            self._worker_count = worker_count
            self._running = True

            for i in range(self._worker_count):
                worker = asyncio.create_task(self._worker_coroutine(f"py-worker-{i}"))
                self._workers.append(worker)

        logger.info(f"PythonWorker started with {worker_count} workers")

    async def stop(self):
        async with self._start_lock:
            if not self._running:
                return

            self._running = False
            for worker in self._workers:
                worker.cancel()

        try:
            await asyncio.gather(*self._workers, return_exceptions=True)
        except asyncio.CancelledError:
            pass

        self._workers.clear()
        async with self._lock:
            for fut in list(self._task_futures.values()):
                try:
                    if fut is not None and not fut.done():
                        fut.set_exception(asyncio.CancelledError())
                except Exception:
                    pass
            self._task_futures.clear()
            self._tasks.clear()
            self._task_queue = asyncio.PriorityQueue()

        try:
            self._cpu_executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            self._cpu_executor.shutdown(wait=False)
        logger.info("PythonWorker stopped")

    async def submit_task(self, task_info: TaskInfo, func: Callable, args: tuple, kwargs: dict) -> asyncio.Future:
        async with self._lock:
            self._tasks[task_info.task_id] = {
                'info': task_info,
                'func': func,
                'args': args,
                'kwargs': kwargs
            }
            loop = asyncio.get_running_loop()
            future = loop.create_future()
            self._task_futures[task_info.task_id] = future

        await self._task_queue.put((-task_info.priority.value, task_info))
        return future

    async def get_task_future(self, task_id: str) -> Optional[asyncio.Future]:
        async with self._lock:
            return self._task_futures.get(task_id)

    async def _execute_task(self, task_func, task_type, *args, **kwargs):
        import inspect

        if task_type == TaskType.GPU_BOUND:
            async with self._gpu_lock:
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(
                    None, lambda: task_func(*args, **kwargs)
                )
        elif task_type == TaskType.CPU_BOUND:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                self._cpu_executor, lambda: task_func(*args, **kwargs)
            )
        else:
            if asyncio.iscoroutinefunction(task_func) or inspect.iscoroutinefunction(task_func):
                return await task_func(*args, **kwargs)
            else:
                result = await asyncio.to_thread(task_func, *args, **kwargs)
                if inspect.isawaitable(result):
                    return await result
                return result

    async def _worker_coroutine(self, worker_name: str):
        logger.debug(f"Worker {worker_name} started")
        try:
            while self._running:
                try:
                    priority_val, task_info = await asyncio.wait_for(
                        self._task_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Worker {worker_name} error getting task: {str(e)}", exc_info=True)
                    continue

                try:
                    # Set execution context early for logging
                    token = set_current_task_id(task_info.task_id)
                    trace_token = TraceContext.set_trace_id(task_info.trace_id)

                    try:
                        if task_info.cancel_requested:
                            logger.debug(f"Task {task_info.task_id} cancelled, skipping")
                            async with self._lock:
                                task_info.status = TaskStatus.CANCELLED
                                if self.state_backplane:
                                    try:
                                        updated = self.state_backplane.update_task_status(task_info.task_id, 5)
                                        if not updated:
                                            self.state_backplane.register_task(task_info.task_id, task_info.priority.value, 5)
                                    except Exception:
                                        pass

                                fut = self._task_futures.get(task_info.task_id)
                                if fut and not fut.done():
                                    fut.set_exception(asyncio.CancelledError())
                            continue

                        logger.debug(f"Worker {worker_name} executing task {task_info.task_id}")

                        task_func = None
                        task_args = ()
                        task_kwargs = {}

                        async with self._lock:
                            if task_info.task_id in self._tasks:
                                task_info.status = TaskStatus.RUNNING
                                task_info.start_time = time.time()

                                if self.state_backplane:
                                    try:
                                        updated = self.state_backplane.update_task_status(task_info.task_id, 2)
                                        if not updated:
                                            self.state_backplane.register_task(task_info.task_id, task_info.priority.value, 2)
                                    except Exception:
                                        pass

                                task_data = self._tasks[task_info.task_id]
                                task_func = task_data.get('func')
                                task_args = task_data.get('args', ())
                                task_kwargs = task_data.get('kwargs', {})
                            else:
                                logger.warning(f"Task {task_info.task_id} data missing")
                                continue

                        if task_func is None:
                            logger.error(f"Task {task_info.task_id} function is None")
                            continue

                        try:
                            result = await self._execute_task(
                                task_func, task_info.task_type, *task_args, **task_kwargs
                            )

                            async with self._lock:
                                task_info.status = TaskStatus.COMPLETED
                                task_info.result = result
                                task_info.end_time = time.time()

                                if self.state_backplane:
                                    try:
                                        updated = self.state_backplane.update_task_status(task_info.task_id, 3)
                                        if not updated:
                                            self.state_backplane.register_task(task_info.task_id, task_info.priority.value, 3)
                                    except Exception:
                                        pass

                                fut = self._task_futures.get(task_info.task_id)
                                if fut and not fut.done():
                                    fut.set_result(result)

                                if task_info.task_id in self._tasks:
                                    self._tasks[task_info.task_id]['func'] = None
                                    self._tasks[task_info.task_id]['args'] = None
                                    self._tasks[task_info.task_id]['kwargs'] = None

                            logger.debug(f"Task {task_info.task_id} completed")

                        except Exception as e:
                            async with self._lock:
                                task_info.status = TaskStatus.FAILED
                                task_info.error = str(e)
                                task_info.end_time = time.time()

                                if self.state_backplane:
                                    try:
                                        updated = self.state_backplane.update_task_status(task_info.task_id, 4)
                                        if not updated:
                                            self.state_backplane.register_task(task_info.task_id, task_info.priority.value, 4)
                                    except Exception:
                                        pass

                                fut = self._task_futures.get(task_info.task_id)
                                if fut and not fut.done():
                                    fut.set_exception(e)

                            logger.error(f"Task {task_info.task_id} failed: {str(e)}", exc_info=True)

                    finally:
                        reset_current_task_id(token)
                        TraceContext.reset_trace_id(trace_token)

                except Exception as e:
                    logger.error(f"Worker {worker_name} uncaught exception: {str(e)}", exc_info=True)
                finally:
                    self._task_queue.task_done()

        finally:
            logger.debug(f"Worker {worker_name} stopped")

    async def get_task_info(self, task_id: str) -> Optional[TaskInfo]:
        async with self._lock:
            if task_id in self._tasks:
                return self._tasks[task_id]['info']
            return None

    async def cancel_task(self, task_id: str) -> bool:
        async with self._lock:
            if task_id not in self._tasks:
                return False
            task_info = self._tasks[task_id]['info']
            if task_info.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                return False

            task_info.cancel_requested = True

            if task_info.status == TaskStatus.RUNNING:
                return True

            task_info.status = TaskStatus.CANCELLED
            fut = self._task_futures.get(task_id)
            if fut and not fut.done():
                fut.set_exception(asyncio.CancelledError())
            return True

    async def clean_completed_tasks(self, max_age: float) -> int:
        cleaned_count = 0
        current_time = time.time()
        async with self._lock:
            expired_task_ids = []
            for task_id, task_data in self._tasks.items():
                task_info = task_data['info']
                if (task_info.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED] and
                    task_info.end_time and
                    current_time - task_info.end_time > max_age):
                    expired_task_ids.append(task_id)
            for task_id in expired_task_ids:
                del self._tasks[task_id]
                cleaned_count += 1
        return cleaned_count
