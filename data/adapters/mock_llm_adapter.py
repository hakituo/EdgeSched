import asyncio
from typing import AsyncGenerator
from domain.interfaces.base_interfaces import LLMInterface

class MockLLMAdapter(LLMInterface):
    def __init__(self):
        print("Initialized MockLLMAdapter (No local model found)")

    async def generate(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        # Simulate processing time
        await asyncio.sleep(0.5)
        return f"[MOCK RESPONSE] I received your prompt: '{prompt}'. Since I am running in MVP Mock mode (model not found), I cannot generate a real response."

    async def stream_generate(self, prompt: str, system_prompt: str = None, **kwargs) -> AsyncGenerator[str, None]:
        response = await self.generate(prompt, system_prompt, **kwargs)
        for word in response.split(" "):
            yield word + " "
            await asyncio.sleep(0.05)
