"""图像管线 Mixin：图像队列、背压、准入快照、worker 循环与图像生成路由。

从 task_scheduler.py 的 GlobalTaskScheduler 拆出。
状态（self._image_* 等）仍由 GlobalTaskScheduler.__init__ 初始化，
本 Mixin 只提供方法，对象语义与拆分前完全一致。
"""
import asyncio
import os
import time
import uuid
from typing import Optional

from mvp_core.utils.trace_context import TraceContext
from mvp_core.utils.logger import logger

try:
    from mvp_core.config import get_settings
except Exception:
    from config import get_settings

try:
    from mvp_core.services.admission_controller import (
        get_admission_controller,
        AdmissionDecision,
        AdmissionRejectedError,
        SystemState,
    )
except ImportError:
    try:
        from services.admission_controller import (  # type: ignore
            get_admission_controller,
            AdmissionDecision,
            AdmissionRejectedError,
            SystemState,
        )
    except ImportError:
        get_admission_controller = None  # type: ignore
        AdmissionDecision = None  # type: ignore
        AdmissionRejectedError = RuntimeError  # type: ignore
        SystemState = None  # type: ignore

# Try to import torch, but don't fail if missing (unless used)
try:
    import torch
except ImportError:
    torch = None


class ImagePipelineMixin:
    """图像队列背压 + 图像生成 A/B 路由（Forge → C++ → SD → Synthetic）"""

    def _read_image_backpressure_settings(self) -> tuple[int, int]:
        queue_capacity = 8
        max_concurrent = 1
        try:
            settings = get_settings()
            scheduler_settings = getattr(settings, "scheduler", None)
            if scheduler_settings is not None:
                queue_capacity = int(getattr(scheduler_settings, "image_queue_capacity", queue_capacity) or queue_capacity)
                max_concurrent = int(
                    getattr(scheduler_settings, "image_max_concurrent_tasks", max_concurrent) or max_concurrent
                )
        except Exception:
            pass

        if queue_capacity <= 0:
            queue_capacity = 1
        if max_concurrent <= 0:
            max_concurrent = 1

        return queue_capacity, max_concurrent

    async def ensure_image_workers_started(self) -> None:
        if self._image_workers_started:
            return

        async with self._image_workers_lock:
            if self._image_workers_started:
                return

            queue_capacity, max_concurrent = self._read_image_backpressure_settings()
            total_slots = queue_capacity + max_concurrent
            self._image_total_slots = total_slots
            self._image_queue = asyncio.Queue(maxsize=total_slots)

            for i in range(max_concurrent):
                task = asyncio.create_task(self._image_worker_loop(i))
                self._image_workers.add(task)
                task.add_done_callback(self._image_workers.discard)

            self._image_workers_started = True

    def get_image_queue_size(self) -> int:
        try:
            if self._image_queue is None:
                return 0
            return int(self._image_queue.qsize())
        except Exception:
            return 0

    def _read_current_system_state(self) -> 'SystemState':
        """Build a point-in-time SystemState snapshot for the admission controller."""
        gpu_free_mb = 0
        gpu_total_mb = 0
        try:
            import torch as _torch_ss
            if _torch_ss.cuda.is_available():
                free, total = _torch_ss.cuda.mem_get_info()
                gpu_free_mb = int(free / 1024 / 1024)
                gpu_total_mb = int(total / 1024 / 1024)
        except Exception:
            pass

        resident_on_gpu = False
        try:
            from mvp_core.services.cpp_scheduler_engine import cpp_scheduler_engine
            if getattr(cpp_scheduler_engine, "enabled", False):
                resident_on_gpu = (
                    cpp_scheduler_engine._python_llm is not None
                    and not cpp_scheduler_engine._python_force_cpu
                )
        except Exception:
            pass
        queue_capacity, max_concurrent = self._read_image_backpressure_settings()
        # 准入容量必须同时统计正在执行和等待中的任务。只看 asyncio.Queue.qsize()
        # 会在 worker 取走任务后释放槽位，从而让活跃任务数突破背压上限。
        active_jobs = len(self._image_jobs)
        total_slots = int(self._image_total_slots or (queue_capacity + max_concurrent))
        return SystemState(
            gpu_free_mb=gpu_free_mb,
            gpu_total_mb=gpu_total_mb,
            resident_on_gpu=resident_on_gpu,
            queue_depth=active_jobs,
            queue_capacity=total_slots,
            burst_running=active_jobs > 0,
        )

    def get_image_metrics(self) -> dict:
        queue_size = self.get_image_queue_size()
        running = 0
        try:
            running = len(self._image_workers)
        except Exception:
            running = 0

        total_slots = int(self._image_total_slots or 0)
        return {
            "queue_size": queue_size,
            "running_workers": running,
            "total_slots": total_slots,
        }

    async def try_enqueue_image_generation(self, prompt: str, request_id: Optional[str] = None, trace_id: Optional[str] = None, **kwargs):
        await self.ensure_image_workers_started()

        if self._image_queue is None:
            raise RuntimeError("Image queue not initialized")

        # --- Admission controller decision (before creating the job) ---
        if get_admission_controller is not None:
            try:
                ac = get_admission_controller()
                system_state = self._read_current_system_state()
                admission_result = ac.decide(system_state)
                if admission_result.decision == AdmissionDecision.REJECT:
                    raise AdmissionRejectedError(admission_result)
            except (AdmissionRejectedError, RuntimeError):
                raise
            except Exception as e:
                logger.warning(f"Admission controller error, falling back to greedy: {e}")
                admission_result = None
        else:
            admission_result = None

        loop = asyncio.get_running_loop()
        started_fut = loop.create_future()
        result_fut = loop.create_future()

        job_id = f"imgjob_{uuid.uuid4().hex[:10]}"
        cancel_event = asyncio.Event()
        routed_kwargs = dict(kwargs or {})
        if request_id and "request_id" not in routed_kwargs:
            routed_kwargs["request_id"] = request_id
        if trace_id and "trace_id" not in routed_kwargs:
            routed_kwargs["trace_id"] = trace_id

        job = {
            "job_id": job_id,
            "request_id": request_id,
            "trace_id": trace_id,
            "prompt": prompt,
            "kwargs": routed_kwargs,
            "created_at": time.time(),
            "started_at": None,
            "finished_at": None,
            "cancel_event": cancel_event,
            "started_fut": started_fut,
            "result_fut": result_fut,
            "running_task": None,
            "admission_result": admission_result,
        }

        async with self._image_job_lock:
            self._image_jobs[job_id] = job
            if request_id:
                self._image_jobs_by_request.setdefault(request_id, set()).add(job_id)

        try:
            self._image_queue.put_nowait(
                {"job_id": job_id}
            )
        except asyncio.QueueFull as e:
            async with self._image_job_lock:
                try:
                    self._image_jobs.pop(job_id, None)
                    if request_id and request_id in self._image_jobs_by_request:
                        self._image_jobs_by_request[request_id].discard(job_id)
                        if not self._image_jobs_by_request[request_id]:
                            self._image_jobs_by_request.pop(request_id, None)
                except Exception:
                    pass
            raise RuntimeError("Image queue is full") from e

        return job_id, started_fut, result_fut

    async def cancel_image_job(self, job_id: str) -> bool:
        async with self._image_job_lock:
            job = self._image_jobs.get(job_id)
            if not job:
                return False
            try:
                job.get("cancel_event").set()
            except Exception:
                pass

            running_task = job.get("running_task")
            if running_task is not None:
                try:
                    running_task.cancel()
                except Exception:
                    pass
            return True

    async def cancel_image_request(self, request_id: str) -> None:
        job_ids = []
        async with self._image_job_lock:
            s = self._image_jobs_by_request.get(request_id)
            if s:
                job_ids = list(s)
        for jid in job_ids:
            try:
                await self.cancel_image_job(jid)
            except Exception:
                pass

    async def _image_worker_loop(self, worker_index: int) -> None:
        while True:
            if self._image_queue is None:
                await asyncio.sleep(0.05)
                continue

            q_item = await self._image_queue.get()
            try:
                job_id = str((q_item or {}).get("job_id") or "")
                if not job_id:
                    continue

                async with self._image_job_lock:
                    job = self._image_jobs.get(job_id)

                if not job:
                    continue

                # --- Admission controller: offload if decided ---
                _admission_result = job.get("admission_result")
                if _admission_result is not None and getattr(_admission_result, "requires_offload", False):
                    try:
                        from mvp_core.services.cpp_scheduler_engine import cpp_scheduler_engine
                        if getattr(cpp_scheduler_engine, "enabled", False):
                            logger.info("[Admission] Executing offload before burst execution")
                            await cpp_scheduler_engine.offload_llm_to_cpu()
                    except Exception as e:
                        logger.warning(f"[Admission] Pre-execution offload failed: {e}")

                trace_id = job.get("trace_id") or job.get("request_id")
                trace_token = None
                if trace_id:
                    try:
                        trace_token = TraceContext.set_trace_id(str(trace_id))
                    except Exception:
                        trace_token = None

                try:
                    cancel_event = job.get("cancel_event")
                    started_fut = job.get("started_fut")
                    result_fut = job.get("result_fut")

                    if cancel_event is not None and cancel_event.is_set():
                        if started_fut is not None and not started_fut.done():
                            started_fut.set_exception(asyncio.CancelledError())
                        if result_fut is not None and not result_fut.done():
                            result_fut.set_exception(asyncio.CancelledError())
                        continue

                    job["started_at"] = time.time()
                    if started_fut is not None and not started_fut.done():
                        started_fut.set_result(True)

                    # Pass skip_auto_offload if admission controller handled offload
                    _sched_kwargs = dict(job.get("kwargs") or {})
                    if _admission_result is not None and getattr(_admission_result, "requires_offload", False):
                        _sched_kwargs["skip_auto_offload"] = True
                    exec_task = asyncio.create_task(self.schedule_image_generation(str(job.get("prompt") or ""), **_sched_kwargs))
                    job["running_task"] = exec_task

                    cancel_wait_task = None
                    if cancel_event is not None:
                        cancel_wait_task = asyncio.create_task(cancel_event.wait())

                    try:
                        done, pending = await asyncio.wait(
                            {exec_task} | ({cancel_wait_task} if cancel_wait_task is not None else set()),
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                    except asyncio.CancelledError:
                        # Worker 被取消，清理任务并退出
                        for t in [exec_task, cancel_wait_task]:
                            if t is not None:
                                try:
                                    t.cancel()
                                except Exception:
                                    pass
                        break

                    if cancel_wait_task is not None and cancel_wait_task in done:
                        try:
                            exec_task.cancel()
                        except Exception:
                            pass
                        if result_fut is not None and not result_fut.done():
                            result_fut.set_exception(asyncio.CancelledError())
                        continue

                    for t in pending:
                        try:
                            t.cancel()
                        except Exception:
                            pass

                    result = await exec_task
                    if result_fut is not None and not result_fut.done():
                        result_fut.set_result(result)
                except asyncio.CancelledError as e:
                    try:
                        rf = job.get("result_fut")
                        if rf is not None and not rf.done():
                            rf.set_exception(e)
                    except Exception:
                        pass
                    raise
                except Exception as e:
                    try:
                        rf = job.get("result_fut")
                        if rf is not None and not rf.done():
                            rf.set_exception(e)
                    except Exception:
                        pass
                finally:
                    job["finished_at"] = time.time()
                    job["running_task"] = None
                    if trace_token is not None:
                        try:
                            TraceContext.reset_trace_id(trace_token)
                        except Exception:
                            pass
            finally:
                try:
                    self._image_queue.task_done()
                except Exception:
                    pass

                try:
                    async with self._image_job_lock:
                        req = job.get("request_id")
                        self._image_jobs.pop(job_id, None)
                        if req and req in self._image_jobs_by_request:
                            self._image_jobs_by_request[req].discard(job_id)
                            if not self._image_jobs_by_request[req]:
                                self._image_jobs_by_request.pop(req, None)
                except Exception:
                    pass

    async def schedule_image_generation(self, prompt: str, **kwargs) -> str:
        """
        New High-Level API for Image Generation supporting A/B Routing.
        Returns the path to the generated image.
        """
        # Pop internal flags before passing kwargs to any adapter
        _skip_auto_offload = kwargs.pop("skip_auto_offload", False)

        # 1. Try Forge UI API (most efficient)
        forge_adapter = None
        try:
            from mvp_core.data.adapters.forge_ui_adapter import ForgeUIAdapter
            forge_adapter = ForgeUIAdapter()
            if await forge_adapter.is_available():
                logger.info(f"Using Forge UI for: {prompt[:20]}...")
                result = await forge_adapter.generate_image(prompt, **kwargs)
                if result.get("status") == "success":
                    images = result.get("images", [])
                    if images:
                        out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output", "images")
                        os.makedirs(out_dir, exist_ok=True)
                        filename = f"forge_{int(time.time() * 1000)}.png"
                        path = os.path.join(out_dir, filename)
                        images[0].save(path)
                        return path
        except Exception as e:
            logger.debug(f"Forge UI not available: {e}")
        finally:
            try:
                if forge_adapter is not None:
                    await forge_adapter.close()
            except Exception:
                pass

        # 2. Try Routing to C++
        try:
            settings = get_settings()
            scheduler_settings = getattr(settings, "scheduler", None)
            backend = "python"
            if scheduler_settings:
                backend = str(getattr(scheduler_settings, "backend", "python") or "python").lower()
                if getattr(scheduler_settings, "use_cpp", False) and getattr(scheduler_settings, "use_cpp_for_image", False):
                    backend = "cpp"

            if backend == "cpp":
                from mvp_core.services.cpp_scheduler_engine import cpp_scheduler_engine
                if getattr(cpp_scheduler_engine, "enabled", False):
                    task_id = f"task_{uuid.uuid4().hex[:8]}_{self._next_task_id}"
                    self._next_task_id += 1
                    task_name = f"sd_cpp_{uuid.uuid4().hex[:4]}"
                    logger.debug(f"Task scheduled - ID: {task_id}, Name: {task_name} (Backend: C++)")
                    logger.debug(f"Worker cpp-img-worker executing task {task_id}")
                    try:
                        logger.info(f"Routing image generation to C++ Engine: {prompt[:20]}...")
                        result = await cpp_scheduler_engine.submit_image_task(
                            prompt, skip_auto_offload=_skip_auto_offload, **kwargs
                        )
                        logger.debug(f"Task {task_id} completed")
                        return result
                    except Exception:
                        logger.debug(f"Task {task_id} completed")
                        raise
        except Exception as e:
            logger.warning(f"C++ Image Generation failed/unavailable: {e}. Falling back to Python.")

        # 3. Try Python SDAdapter (real model)
        try:
            settings = get_settings()
            model_settings = getattr(settings, "model", None)
            sd_model_path = None
            if model_settings:
                sd_model_path = str(getattr(model_settings, "image_gen_path", "") or "")

            if sd_model_path and os.path.exists(sd_model_path):
                logger.info(f"Using SDAdapter for: {prompt[:20]}...")
                from mvp_core.data.adapters.sd_adapter import SDAdapter
                adapter = SDAdapter({
                    'sd_model_path': sd_model_path,
                    'device': 'cuda' if torch and torch.cuda.is_available() else 'cpu'
                })
                result = await adapter.generate_image(prompt, **kwargs)
                if isinstance(result, dict) and result.get("status") == "success":
                    images = result.get("images", [])
                    if images:
                        out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output", "images")
                        os.makedirs(out_dir, exist_ok=True)
                        filename = f"sd_{int(time.time() * 1000)}.png"
                        path = os.path.join(out_dir, filename)
                        images[0].save(path)
                        return path
        except Exception as e:
            logger.warning(f"SDAdapter failed: {e}")

        # 4. Fallback to Synthetic
        logger.info(f"Using Synthetic fallback for: {prompt[:20]}...")
        from mvp_core.data.adapters.synthetic_image_adapter import SyntheticImageAdapter
        adapter = SyntheticImageAdapter()
        return await adapter.generate_image(prompt, **kwargs)
