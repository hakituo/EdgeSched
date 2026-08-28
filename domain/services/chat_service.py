from typing import AsyncGenerator, Dict, Any
from domain.interfaces.base_interfaces import LLMInterface, MemoryInterface
from shared.di import container
import re
import asyncio

try:
    from mvp_core.services.task_scheduler import GlobalTaskScheduler, get_global_scheduler
except Exception:
    from services.task_scheduler import GlobalTaskScheduler, get_global_scheduler

class ChatService:
    def __init__(self):
        self.llm = container.resolve(LLMInterface)
        self.memory = container.resolve(MemoryInterface)

        self.scheduler = None
        try:
            self.scheduler = container.resolve(GlobalTaskScheduler)
        except Exception:
            try:
                self.scheduler = get_global_scheduler()
            except Exception:
                self.scheduler = None
        
    async def process_message(
        self,
        user_input: str,
        request_id: str | None = None,
        trace_id: str | None = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        # 1. Save User Message
        await self.memory.add_message("user", user_input)

        # 2. Get History
        history = await self.memory.get_history(limit=10)
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])

        # 3. Construct Prompt
        system_prompt = ""
        
        # 4. Stream Response
        full_response = ""
        try:
            if self.scheduler is not None:
                async for chunk in self.scheduler.submit_llm_task(
                    history_text,
                    system_prompt=system_prompt,
                    request_id=request_id,
                    trace_id=trace_id,
                ):
                    if chunk:
                        full_response += chunk
                        yield {"type": "token", "data": chunk}
            else:
                async for chunk in self.llm.stream_generate(history_text, system_prompt=system_prompt):
                    full_response += chunk
                    yield {"type": "token", "data": chunk}
        except asyncio.CancelledError:
            raise
        except Exception:
            msg = "本地推理引擎当前不可用，请稍后重试"
            full_response += msg
            yield {"type": "token", "data": msg}

        for match in re.finditer(r"\[GEN_IMG:(?P<prompt>.*?)\]", full_response, flags=re.DOTALL):
            prompt = (match.group("prompt") or "").strip()
            if prompt:
                yield {"type": "image_trigger", "data": {"prompt": prompt}}

        # 5. Save AI Message
        await self.memory.add_message("ai", full_response)
