"""明确标识为 Mock 的 TTS 测试适配器。"""

from __future__ import annotations

import asyncio


class MockTTSAdapter:
    def __init__(self, synthesis_time: float = 0.2) -> None:
        self.synthesis_time = synthesis_time

    async def synthesize(self, text: str, **kwargs) -> bytes:
        await asyncio.sleep(self.synthesis_time)
        return b"mock_audio_data"
