from fastapi import WebSocket, WebSocketDisconnect
from domain.services.chat_service import ChatService
import logging
import uuid
import asyncio
import os
import time
from typing import Any

from domain.interfaces.base_interfaces import ImageGenInterface
from shared.di import container
try:
    from mvp_core.utils.trace_context import TraceContext
except Exception:
    from utils.trace_context import TraceContext

logger = logging.getLogger("WebSocketHandler")

class WebSocketHandler:
    def __init__(self, chat_service: ChatService):
        self.chat_service = chat_service
        self._image_workers_lock = None
        self._image_workers = set()
        self._image_workers_started = False
        self._image_inflight = set()
        self._image_cancelled_ws_ids = set()
        self._active_chat_tasks = {}

    def _make_image_key(self, websocket: WebSocket, request_id: str, prompt: str) -> str:
        try:
            ws_id = id(websocket)
        except Exception:
            ws_id = 0
        return f"{ws_id}:{request_id}:{prompt}"

    def _ws_is_cancelled(self, websocket: WebSocket) -> bool:
        try:
            return id(websocket) in self._image_cancelled_ws_ids
        except Exception:
            return False

    async def _ensure_image_workers_started(self) -> None:
        if self._image_workers_started:
            return

        if self._image_workers_lock is None:
            self._image_workers_lock = asyncio.Lock()

        async with self._image_workers_lock:
            if self._image_workers_started:
                return

            scheduler = None
            try:
                from mvp_core.services.task_scheduler import GlobalTaskScheduler
            except Exception:
                try:
                    from services.task_scheduler import GlobalTaskScheduler
                except Exception:
                    GlobalTaskScheduler = None

            if GlobalTaskScheduler is not None:
                try:
                    scheduler = container.resolve(GlobalTaskScheduler)
                except Exception:
                    scheduler = None

            if scheduler is not None:
                await scheduler.ensure_image_workers_started()

            self._image_workers_started = True

    async def _try_enqueue_image_generation(self, websocket: WebSocket, request_id: str, prompt: str) -> bool:
        await self._ensure_image_workers_started()

        if self._ws_is_cancelled(websocket):
            return False

        key = self._make_image_key(websocket, request_id, prompt)
        if key in self._image_inflight:
            try:
                queued = 0
                try:
                    from mvp_core.services.task_scheduler import GlobalTaskScheduler
                except Exception:
                    from services.task_scheduler import GlobalTaskScheduler
                try:
                    scheduler = container.resolve(GlobalTaskScheduler)
                    queued = int(scheduler.get_image_queue_size())
                except Exception:
                    queued = 0
                await self._safe_send_json(
                    websocket,
                    {
                        "type": "image_status",
                        "request_id": request_id,
                        "data": {"prompt": prompt, "status": "queued", "queued": queued},
                    },
                )
            except Exception:
                pass
            return True

        scheduler = None
        try:
            from mvp_core.services.task_scheduler import GlobalTaskScheduler
        except Exception:
            try:
                from services.task_scheduler import GlobalTaskScheduler
            except Exception:
                GlobalTaskScheduler = None

        if GlobalTaskScheduler is not None:
            try:
                scheduler = container.resolve(GlobalTaskScheduler)
            except Exception:
                scheduler = None

        if scheduler is None:
            asyncio.create_task(self._run_image_generation(websocket, request_id, prompt))
            self._image_inflight.add(key)
            return True

        try:
            job_id, started_fut, result_fut = await scheduler.try_enqueue_image_generation(
                prompt,
                request_id=str(request_id),
                trace_id=str(request_id),
            )
        except Exception:
            err = {"error_type": "runtime_error", "message": "Image queue is full", "retryable": True}
            await self._safe_send_json(
                websocket,
                {"type": "image_error", "request_id": request_id, "data": {"prompt": prompt, "source": "image", **err}},
            )
            await self._safe_send_json(
                websocket,
                {"type": "image_status", "request_id": request_id, "data": {"prompt": prompt, "status": "finished"}},
            )
            return False

        self._image_inflight.add(key)

        try:
            queued = int(scheduler.get_image_queue_size())
            metrics = {}
            try:
                metrics = scheduler.get_image_metrics()
            except Exception:
                metrics = {}
            await self._safe_send_json(
                websocket,
                {
                    "type": "image_status",
                    "request_id": request_id,
                    "data": {"prompt": prompt, "status": "queued", "queued": queued, "job_id": job_id, "metrics": metrics},
                },
            )
        except Exception:
            pass

        task = asyncio.create_task(
            self._await_scheduled_image(websocket, request_id, prompt, key, job_id, started_fut, result_fut)
        )
        self._image_workers.add(task)
        task.add_done_callback(self._image_workers.discard)

        return True

    async def _await_scheduled_image(
        self,
        websocket: WebSocket,
        request_id: str,
        prompt: str,
        key: str,
        job_id: str,
        started_fut: asyncio.Future,
        result_fut: asyncio.Future,
    ) -> None:
        try:
            try:
                await started_fut
            except Exception:
                pass

            if not self._ws_is_cancelled(websocket):
                await self._safe_send_json(
                    websocket,
                    {
                        "type": "image_status",
                        "request_id": request_id,
                        "data": {"prompt": prompt, "status": "started", "job_id": job_id},
                    },
                )

            path = await result_fut
            if not self._ws_is_cancelled(websocket):
                path = await self._save_first_image(path)
                await self._safe_send_json(
                    websocket,
                    {"type": "image_result", "request_id": request_id, "data": {"prompt": prompt, "path": path, "job_id": job_id}},
                )
        except Exception as e:
            if not self._ws_is_cancelled(websocket):
                err = self._classify_error(e)
                await self._safe_send_json(
                    websocket,
                    {
                        "type": "image_error",
                        "request_id": request_id,
                        "data": {"prompt": prompt, "source": "image", "job_id": job_id, **err},
                    },
                )
        finally:
            try:
                self._image_inflight.discard(key)
            except Exception:
                pass

            if not self._ws_is_cancelled(websocket):
                await self._safe_send_json(
                    websocket,
                    {
                        "type": "image_status",
                        "request_id": request_id,
                        "data": {"prompt": prompt, "status": "finished", "job_id": job_id},
                    },
                )

    async def _cancel_request(self, request_id: str) -> None:
        task = self._active_chat_tasks.get(request_id)
        if task is not None:
            try:
                task.cancel()
            except Exception:
                pass

        try:
            from mvp_core.services.task_scheduler import GlobalTaskScheduler
        except Exception:
            try:
                from services.task_scheduler import GlobalTaskScheduler
            except Exception:
                GlobalTaskScheduler = None

        if GlobalTaskScheduler is not None:
            try:
                scheduler = container.resolve(GlobalTaskScheduler)
                await scheduler.cancel_request(str(request_id))
            except Exception:
                pass

    def _classify_error(self, e: BaseException) -> dict:
        err_type = "runtime_error"
        retryable = False

        if isinstance(e, (asyncio.TimeoutError, TimeoutError)):
            err_type = "timeout"
            retryable = True
        elif isinstance(e, asyncio.CancelledError):
            err_type = "cancelled"
            retryable = False
        else:
            msg = str(e or "")
            msg_lower = msg.lower()
            if any(
                s in msg_lower
                for s in (
                    "not available",
                    "not ready",
                    "bindings not available",
                    "scheduler_py not available",
                    "c++ scheduler not available",
                )
            ):
                err_type = "backend_unavailable"
                retryable = True

        return {
            "error_type": err_type,
            "message": str(e),
            "retryable": retryable,
        }

    async def _safe_send_json(self, websocket: WebSocket, payload: dict) -> None:
        try:
            await websocket.send_json(payload)
        except Exception:
            pass

    async def handle_connection(self, websocket: WebSocket):
        await websocket.accept()
        await self._ensure_image_workers_started()
        try:
            while True:
                data = await websocket.receive_json()
                msg_type = data.get("type")
                request_id = data.get("request_id") or uuid.uuid4().hex

                if msg_type in ("cancel", "interrupt"):
                    await self._cancel_request(str(request_id))
                    await self._safe_send_json(websocket, {"type": "response_done", "request_id": request_id})
                    continue

                user_input = data.get("text") or data.get("content")
                if not user_input:
                    continue

                if request_id in self._active_chat_tasks:
                    await self._cancel_request(str(request_id))

                task = asyncio.create_task(self._handle_chat_request(websocket, str(request_id), str(user_input)))
                self._active_chat_tasks[str(request_id)] = task

                def _cleanup(_t: asyncio.Task, rid: str = str(request_id)):
                    try:
                        self._active_chat_tasks.pop(rid, None)
                    except Exception:
                        pass

                task.add_done_callback(_cleanup)
                    
        except WebSocketDisconnect:
            logger.info("Client disconnected")
            try:
                self._image_cancelled_ws_ids.add(id(websocket))
            except Exception:
                pass
            try:
                for rid in list(self._active_chat_tasks.keys()):
                    try:
                        await self._cancel_request(str(rid))
                    except Exception:
                        pass
            except Exception:
                pass
        except Exception as e:
            logger.error(f"Error in websocket loop: {e}")
            try:
                self._image_cancelled_ws_ids.add(id(websocket))
            except Exception:
                pass
            try:
                for rid in list(self._active_chat_tasks.keys()):
                    try:
                        await self._cancel_request(str(rid))
                    except Exception:
                        pass
            except Exception:
                pass
            await websocket.close()

    async def _handle_chat_request(self, websocket: WebSocket, request_id: str, user_input: str) -> None:
        trace_token = None
        try:
            try:
                trace_token = TraceContext.set_trace_id(str(request_id))
            except Exception:
                trace_token = None

            await self._safe_send_json(websocket, {"type": "acknowledged", "request_id": request_id})

            async for chunk in self.chat_service.process_message(user_input, request_id=request_id, trace_id=request_id):
                if self._ws_is_cancelled(websocket):
                    break

                if chunk.get("type") == "token":
                    await self._safe_send_json(
                        websocket,
                        {
                            "type": "response",
                            "request_id": request_id,
                            "content": chunk.get("data", ""),
                        },
                    )
                elif chunk.get("type") == "image_trigger":
                    payload = chunk.get("data", {}) or {}
                    await self._safe_send_json(
                        websocket,
                        {"type": "image_trigger", "request_id": request_id, "data": payload},
                    )
                    prompt = str(payload.get("prompt") or "").strip()
                    if prompt:
                        await self._try_enqueue_image_generation(websocket, request_id, prompt)
                else:
                    await self._safe_send_json(
                        websocket,
                        {
                            "type": "event",
                            "request_id": request_id,
                            "event": chunk.get("type"),
                            "data": chunk.get("data", {}),
                        },
                    )
        except asyncio.CancelledError:
            err = {"error_type": "cancelled", "message": "cancelled", "retryable": False}
            await self._safe_send_json(websocket, {"type": "error", "request_id": request_id, "data": {"source": "chat", **err}})
            raise
        except Exception as e:
            err = self._classify_error(e)
            await self._safe_send_json(
                websocket,
                {
                    "type": "error",
                    "request_id": request_id,
                    "data": {"source": "chat", **err},
                },
            )
        finally:
            try:
                await self._safe_send_json(websocket, {"type": "response_done", "request_id": request_id})
            finally:
                if trace_token is not None:
                    try:
                        TraceContext.reset_trace_id(trace_token)
                    except Exception:
                        pass

    def _get_output_dir(self) -> str:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        mvp_root = os.path.dirname(os.path.dirname(current_dir))
        out_dir = os.path.join(mvp_root, "output", "images")
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    async def _save_first_image(self, result: Any) -> str:
        if isinstance(result, str):
            return result

        if isinstance(result, dict):
            images = result.get("images")
            if images and isinstance(images, list):
                first = images[0]
                if hasattr(first, "save"):
                    out_dir = self._get_output_dir()
                    filename = f"mvp_{int(time.time() * 1000)}.png"
                    path = os.path.join(out_dir, filename)
                    await asyncio.to_thread(first.save, path)
                    return path

        raise RuntimeError("Unsupported image result format")

    async def _run_image_generation(self, websocket: WebSocket, request_id: str, prompt: str):
        scheduler = None
        try:
            from mvp_core.services.task_scheduler import GlobalTaskScheduler
        except Exception:
            try:
                from services.task_scheduler import GlobalTaskScheduler
            except Exception:
                GlobalTaskScheduler = None

        if GlobalTaskScheduler is not None:
            try:
                scheduler = container.resolve(GlobalTaskScheduler)
            except Exception:
                scheduler = None

        image_gen = None
        if scheduler is None:
            try:
                image_gen = container.resolve(ImageGenInterface)
            except Exception as e:
                err = self._classify_error(e)
                await self._safe_send_json(
                    websocket,
                    {
                        "type": "image_error",
                        "request_id": request_id,
                        "data": {"prompt": prompt, "source": "image", **err},
                    },
                )
                await self._safe_send_json(
                    websocket,
                    {
                        "type": "image_status",
                        "request_id": request_id,
                        "data": {"prompt": prompt, "status": "finished"},
                    },
                )
                return

        try:
            await self._safe_send_json(
                websocket,
                {
                    "type": "image_status",
                    "request_id": request_id,
                    "data": {"prompt": prompt, "status": "started"},
                },
            )
            if scheduler is not None:
                res = await scheduler.schedule_image_generation(prompt)
            else:
                res = image_gen.generate_image(prompt)
            if asyncio.iscoroutine(res):
                res = await res
            path = await self._save_first_image(res)
            await self._safe_send_json(
                websocket,
                {"type": "image_result", "request_id": request_id, "data": {"prompt": prompt, "path": path}},
            )
        except Exception as e:
            err = self._classify_error(e)
            await self._safe_send_json(
                websocket,
                {
                    "type": "image_error",
                    "request_id": request_id,
                    "data": {"prompt": prompt, "source": "image", **err},
                },
            )
        finally:
            await self._safe_send_json(
                websocket,
                {
                    "type": "image_status",
                    "request_id": request_id,
                    "data": {"prompt": prompt, "status": "finished"},
                },
            )
