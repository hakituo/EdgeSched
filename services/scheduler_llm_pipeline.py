"""LLM 管线 Mixin：取消事件、回退适配器与 LLM 生成 A/B 路由。

从 task_scheduler.py 的 GlobalTaskScheduler 拆出。
状态（self._llm_cancel_* / self._fallback_*）仍由 GlobalTaskScheduler.__init__ 初始化。
"""
import asyncio
from typing import Optional

from mvp_core.utils.trace_context import TraceContext
from mvp_core.utils.logger import logger

try:
    from mvp_core.config import get_settings
except Exception:
    from config import get_settings


class LLMPipelineMixin:
    """LLM 请求取消事件管理 + C++/Python LLM 路由与回退"""

    async def get_or_create_llm_cancel_event(self, request_id: str) -> asyncio.Event:
        async with self._llm_cancel_lock:
            ev = self._llm_cancel_events.get(request_id)
            if ev is None:
                ev = asyncio.Event()
                self._llm_cancel_events[request_id] = ev
            return ev

    async def cancel_llm_request(self, request_id: str) -> None:
        async with self._llm_cancel_lock:
            ev = self._llm_cancel_events.get(request_id)
            if ev is not None:
                ev.set()

    async def _get_fallback_llm_adapter(self):
        if self._fallback_llm_adapter is not None:
            return self._fallback_llm_adapter

        async with self._fallback_llm_lock:
            if self._fallback_llm_adapter is not None:
                return self._fallback_llm_adapter

            settings = get_settings()
            model_settings = getattr(settings, "model", None)
            text_path = str(getattr(model_settings, "text_path", "") or "")

            adapter = None
            if text_path:
                if text_path.lower().endswith(".gguf"):
                    try:
                        from mvp_core.data.adapters.gguf_llm_adapter import GGUFLLMAdapter

                        n_ctx = int(getattr(model_settings, "n_ctx", 2048) or 2048) if model_settings else 2048
                        n_gpu_layers = int(getattr(model_settings, "n_gpu_layers", -1) or -1) if model_settings else -1
                        adapter = GGUFLLMAdapter(text_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers)
                    except Exception as e:
                        logger.warning(f"Failed to init GGUFLLMAdapter, fallback next: {e}")

                if adapter is None:
                    try:
                        from mvp_core.data.adapters.local_llm_adapter import LocalLLMAdapter

                        adapter = LocalLLMAdapter()
                    except Exception as e:
                        logger.warning(f"Failed to init LocalLLMAdapter, fallback next: {e}")

            if adapter is None:
                from mvp_core.data.adapters.mock_llm_adapter import MockLLMAdapter

                adapter = MockLLMAdapter()

            self._fallback_llm_adapter = adapter
            return adapter

    async def submit_llm_task(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        request_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        cancel_event: Optional[asyncio.Event] = None,
        **kwargs,
    ):
        """
        Submit LLM inference task (Prioritize C++ Scheduler)
        """
        if system_prompt is None:
            system_prompt = kwargs.pop("system_prompt", None)

        if cancel_event is None and request_id:
            cancel_event = await self.get_or_create_llm_cancel_event(request_id)

        current_trace = trace_id or request_id
        trace_token = None
        if current_trace:
            try:
                trace_token = TraceContext.set_trace_id(str(current_trace))
            except Exception:
                trace_token = None

        if current_trace and "trace_id" not in kwargs:
            kwargs["trace_id"] = str(current_trace)
        if request_id and "request_id" not in kwargs:
            kwargs["request_id"] = str(request_id)

        cpp_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        from mvp_core.services.cpp_scheduler_engine import cpp_scheduler_engine

        try:
            if cancel_event is not None and cancel_event.is_set():
                raise asyncio.CancelledError()

            if getattr(cpp_scheduler_engine, "enabled", False):
                try:
                    async for token in cpp_scheduler_engine.submit_llm_task(cpp_prompt, **kwargs):
                        if cancel_event is not None and cancel_event.is_set():
                            raise asyncio.CancelledError()
                        yield token
                    return
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.warning(f"C++ Engine LLM failed, trying fallback: {e}")

            logger.warning("Using L1 fallback for LLM task...")
            adapter = await self._get_fallback_llm_adapter()
            if adapter:
                stream_fn = getattr(adapter, "stream_generate", None)
                if callable(stream_fn):
                    async for token in stream_fn(prompt, system_prompt=system_prompt, **kwargs):
                        if cancel_event is not None and cancel_event.is_set():
                            raise asyncio.CancelledError()
                        yield token
                    return

                text = await adapter.generate(prompt, system_prompt=system_prompt, **kwargs)
                if cancel_event is not None and cancel_event.is_set():
                    raise asyncio.CancelledError()
                if text:
                    yield str(text)
            else:
                yield "Error: No LLM backend available"
        finally:
            if request_id:
                async with self._llm_cancel_lock:
                    self._llm_cancel_events.pop(request_id, None)
            if trace_token is not None:
                try:
                    TraceContext.reset_trace_id(trace_token)
                except Exception:
                    pass

    async def schedule_llm_generation(self, prompt: str, **kwargs) -> str:
        """
        New High-Level API for LLM Generation supporting A/B Routing.
        Returns the generated text.
        """
        try:
            settings = get_settings()
            scheduler_settings = getattr(settings, "scheduler", None)
            backend = "python"
            if scheduler_settings:
                backend = str(getattr(scheduler_settings, "backend", "python") or "python").lower()
                if getattr(scheduler_settings, "use_cpp", False) and getattr(scheduler_settings, "use_cpp_for_llm", False):
                    backend = "cpp"

            if backend == "cpp":
                from mvp_core.services.cpp_scheduler_engine import cpp_scheduler_engine
                if getattr(cpp_scheduler_engine, "enabled", False):
                    import uuid
                    task_id = f"task_{uuid.uuid4().hex[:8]}_{self._next_task_id}"
                    self._next_task_id += 1
                    task_name = f"llm_cpp_{uuid.uuid4().hex[:4]}"
                    logger.debug(f"Task scheduled - ID: {task_id}, Name: {task_name} (Backend: C++)")
                    logger.debug(f"Worker cpp-llm-worker executing task {task_id}")

                    logger.info(f"Routing LLM generation to C++ Engine: {prompt[:20]}...")
                    full_text = ""
                    try:
                        cpp_kwargs = dict(kwargs)
                        cpp_kwargs.setdefault("stream", False)
                        async for token in cpp_scheduler_engine.submit_llm_task(prompt, **cpp_kwargs):
                            full_text += token
                        # Log completion AFTER the generator finishes
                        logger.debug(f"Task {task_id} completed")
                        return full_text
                    except Exception:
                        logger.debug(f"Task {task_id} completed")
                        raise
        except Exception as e:
            logger.warning(f"C++ LLM Generation failed/unavailable: {e}. Falling back to Python.")

        system_prompt = kwargs.pop("system_prompt", None)
        try:
            adapter = await self._get_fallback_llm_adapter()
            return await adapter.generate(prompt, system_prompt=system_prompt, **kwargs)
        except Exception as e:
            logger.warning(f"Python LLM fallback failed: {e}. Using MockLLMAdapter.")
            from mvp_core.data.adapters.mock_llm_adapter import MockLLMAdapter

            adapter = MockLLMAdapter()
            return await adapter.generate(prompt, system_prompt=system_prompt, **kwargs)
