from abc import ABC, abstractmethod
from typing import AsyncGenerator, AsyncIterator, Any, Dict, List

from ..models.tts import TTSChunkRequest, TTSChunkResult

class LLMInterface(ABC):
    @abstractmethod
    async def generate(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        pass

    @abstractmethod
    async def stream_generate(self, prompt: str, system_prompt: str = None, **kwargs) -> AsyncGenerator[str, None]:
        pass

class MemoryInterface(ABC):
    @abstractmethod
    async def add_message(self, role: str, content: str, **kwargs):
        pass

    @abstractmethod
    async def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    async def search(self, query: str, limit: int = 5) -> List[str]:
        pass

class TTSInterface(ABC):
    @abstractmethod
    async def synthesize(self, text: str, emotion: str = None, **kwargs) -> str:
        """Returns path to audio file"""
        pass

    async def synthesize_chunk(
        self,
        request: TTSChunkRequest,
    ) -> TTSChunkResult:
        """合成一个已完成断句的语音块；旧适配器可暂不实现。"""
        raise NotImplementedError

    async def stream_synthesize(
        self,
        request_id: str,
        token_stream: AsyncIterator[str],
        *,
        voice_profile_id: str,
        language: str,
        **kwargs,
    ) -> AsyncIterator[TTSChunkResult]:
        """将文本 token 流转换为句子级音频结果流。"""
        raise NotImplementedError
        yield  # pragma: no cover

    async def cancel(self, request_id: str, **kwargs) -> None:
        """取消该请求尚未派发的语音块。"""
        raise NotImplementedError

class STTInterface(ABC):
    @abstractmethod
    async def transcribe(self, audio_data: bytes, **kwargs) -> str:
        pass

class ImageGenInterface(ABC):
    @abstractmethod
    async def generate_image(self, prompt: str, **kwargs) -> str:
        """Returns path to generated image"""
        pass

class VisionInterface(ABC):
    @abstractmethod
    async def analyze_image(self, image_path: str, prompt: str, **kwargs) -> str:
        """Returns description of the image"""
        pass
