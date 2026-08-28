import pytest
from unittest.mock import AsyncMock, MagicMock
from domain.services.chat_service import ChatService
from domain.interfaces.base_interfaces import LLMInterface, MemoryInterface
from shared.di import container
import asyncio

from presentation.websocket.handler import WebSocketHandler
try:
    from mvp_core.services.task_scheduler import GlobalTaskScheduler
except Exception:
    from services.task_scheduler import GlobalTaskScheduler

try:
    from mvp_core.utils.trace_context import TraceContext
except Exception:
    from utils.trace_context import TraceContext

def test_chat_service_process_message():
    async def run():
        # Setup Mocks
        mock_llm = MagicMock(spec=LLMInterface)
        mock_llm.stream_generate = AsyncMock(return_value=iter(["Hello", " ", "World"]))
        
        async def async_gen(text, **kwargs):
            yield "Hello"
            yield " "
            yield "World"
        mock_llm.stream_generate = async_gen

        mock_memory = MagicMock(spec=MemoryInterface)
        mock_memory.add_message = AsyncMock()
        mock_memory.get_history = AsyncMock(return_value=[])

        # Register Mocks
        container.register(LLMInterface, mock_llm)
        container.register(MemoryInterface, mock_memory)

        scheduler = GlobalTaskScheduler()
        scheduler._fallback_llm_adapter = mock_llm
        container.register(GlobalTaskScheduler, scheduler)

        try:
            from mvp_core.services.cpp_scheduler_engine import cpp_scheduler_engine
        except Exception:
            from services.cpp_scheduler_engine import cpp_scheduler_engine

        prev_enabled = getattr(cpp_scheduler_engine, "enabled", False)
        cpp_scheduler_engine.enabled = False

        # Init Service
        service = ChatService()

        try:
            responses = []
            async for chunk in service.process_message("Hi", request_id="r0", trace_id="r0"):
                responses.append(chunk)
        finally:
            cpp_scheduler_engine.enabled = prev_enabled

        # Verify
        tokens = [r["data"] for r in responses if r["type"] == "token"]
        assert "".join(tokens) == "Hello World"

        mock_memory.add_message.assert_any_call("user", "Hi")
        mock_memory.add_message.assert_any_call("ai", "Hello World")

    asyncio.run(run())


def test_websocket_image_backpressure_queue_full():
    class FakeWebSocket:
        def __init__(self):
            self.sent = []

        async def send_json(self, payload):
            self.sent.append(payload)

    async def run():
        handler = WebSocketHandler(chat_service=MagicMock())

        scheduler = GlobalTaskScheduler()
        scheduler._read_image_backpressure_settings = lambda: (1, 1)

        block = asyncio.Event()

        async def _blocked_schedule_image_generation(prompt: str, **kwargs):
            await block.wait()
            return ""

        scheduler.schedule_image_generation = _blocked_schedule_image_generation

        container.register(GlobalTaskScheduler, scheduler)

        ws = FakeWebSocket()
        await handler._ensure_image_workers_started()

        try:
            ok1 = await handler._try_enqueue_image_generation(ws, "r1", "p1")
            ok2 = await handler._try_enqueue_image_generation(ws, "r2", "p2")
            ok3 = await handler._try_enqueue_image_generation(ws, "r3", "p3")

            assert ok1 is True
            assert ok2 is True
            assert ok3 is False

            assert any(m.get("type") == "image_error" and m.get("request_id") == "r3" for m in ws.sent)
        finally:
            block.set()
            await asyncio.sleep(0)

            workers = list(handler._image_workers)
            for t in workers:
                try:
                    t.cancel()
                except Exception:
                    pass
            await asyncio.gather(*workers, return_exceptions=True)

            sched_workers = list(getattr(scheduler, "_image_workers", set()))
            for t in sched_workers:
                try:
                    t.cancel()
                except Exception:
                    pass
            await asyncio.gather(*sched_workers, return_exceptions=True)

    asyncio.run(run())


def test_scheduler_llm_cancel_stops_stream():
    class FakeAdapter:
        async def stream_generate(self, prompt: str, **kwargs):
            yield "A"
            await asyncio.sleep(0)
            yield "B"

    async def run():
        scheduler = GlobalTaskScheduler()
        scheduler._fallback_llm_adapter = FakeAdapter()

        try:
            from mvp_core.services.cpp_scheduler_engine import cpp_scheduler_engine
        except Exception:
            from services.cpp_scheduler_engine import cpp_scheduler_engine

        prev_enabled = getattr(cpp_scheduler_engine, "enabled", False)
        cpp_scheduler_engine.enabled = False
        try:
            got = []
            try:
                async for t in scheduler.submit_llm_task("hi", request_id="r1"):
                    got.append(t)
                    await scheduler.cancel_llm_request("r1")
            except asyncio.CancelledError:
                pass

            assert got == ["A"]
        finally:
            cpp_scheduler_engine.enabled = prev_enabled

    asyncio.run(run())


def test_scheduler_trace_id_visible_in_adapter():
    class FakeAdapter:
        async def stream_generate(self, prompt: str, **kwargs):
            assert TraceContext.get_trace_id() == "tid"
            yield "X"

    async def run():
        scheduler = GlobalTaskScheduler()
        scheduler._fallback_llm_adapter = FakeAdapter()

        try:
            from mvp_core.services.cpp_scheduler_engine import cpp_scheduler_engine
        except Exception:
            from services.cpp_scheduler_engine import cpp_scheduler_engine

        prev_enabled = getattr(cpp_scheduler_engine, "enabled", False)
        cpp_scheduler_engine.enabled = False
        try:
            out = []
            async for t in scheduler.submit_llm_task("hi", trace_id="tid"):
                out.append(t)
            assert out == ["X"]
        finally:
            cpp_scheduler_engine.enabled = prev_enabled

    asyncio.run(run())


def test_scheduler_image_cancel_sets_future_cancelled():
    async def run():
        scheduler = GlobalTaskScheduler()
        scheduler._read_image_backpressure_settings = lambda: (1, 1)

        block = asyncio.Event()

        async def _blocked_schedule_image_generation(prompt: str, **kwargs):
            await block.wait()
            return ""

        scheduler.schedule_image_generation = _blocked_schedule_image_generation

        job_id, started_fut, result_fut = await scheduler.try_enqueue_image_generation(
            "p1",
            request_id="rid",
            trace_id="rid",
        )

        await asyncio.wait_for(started_fut, timeout=1.0)
        await scheduler.cancel_request("rid")

        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(result_fut, timeout=1.0)

        workers = list(getattr(scheduler, "_image_workers", set()))
        for t in workers:
            try:
                t.cancel()
            except Exception:
                pass
        await asyncio.gather(*workers, return_exceptions=True)

    asyncio.run(run())
