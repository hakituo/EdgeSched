"""领域数据模型。"""

from .tts import (
    TTSChunkRequest,
    TTSChunkResult,
    TTSChunkStatus,
    TTSOutputMode,
    VoiceCloneProfile,
)

__all__ = [
    "TTSChunkRequest",
    "TTSChunkResult",
    "TTSChunkStatus",
    "TTSOutputMode",
    "VoiceCloneProfile",
]
