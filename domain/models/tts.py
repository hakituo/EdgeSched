"""TTS 领域契约。

音频字节只在进程内传递；实验 JSON 使用 ``to_record``，不会嵌入二进制内容。
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any


class TTSOutputMode(str, Enum):
    """音频结果输出方式。"""

    BYTES = "bytes"
    FILE = "file"
    BOTH = "both"


class TTSChunkStatus(str, Enum):
    """单个语音块的终态。"""

    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    DISCARDED = "discarded"


@dataclass(frozen=True)
class VoiceCloneProfile:
    """Qwen3-TTS Base 的可复用音色画像。"""

    profile_id: str
    ref_audio_path: str
    ref_text: str
    language: str
    x_vector_only_mode: bool = False
    audio_sha256: str = ""

    def __post_init__(self) -> None:
        if not self.profile_id.strip():
            raise ValueError("profile_id 不能为空")
        if not self.ref_audio_path.strip():
            raise ValueError("ref_audio_path 不能为空")
        if not self.language.strip():
            raise ValueError("language 不能为空")
        if not self.x_vector_only_mode and not self.ref_text.strip():
            raise ValueError("ICL 音色画像必须提供参考音频的准确转写 ref_text")
        if self.audio_sha256 and len(self.audio_sha256) != 64:
            raise ValueError("audio_sha256 必须是 64 位 SHA-256")

    def validate_audio(self) -> "VoiceCloneProfile":
        """校验参考音频并返回带真实哈希的不可变画像。"""

        audio_path = Path(self.ref_audio_path).expanduser().resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(f"找不到参考音频: {audio_path}")
        digest = hashlib.sha256(audio_path.read_bytes()).hexdigest()
        if self.audio_sha256 and digest.lower() != self.audio_sha256.lower():
            raise ValueError(
                f"参考音频哈希不一致: expected={self.audio_sha256}, actual={digest}"
            )
        return replace(
            self,
            ref_audio_path=str(audio_path),
            audio_sha256=digest,
        )

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TTSChunkRequest:
    """进入 TTS 有界队列的确定性语音块。"""

    request_id: str
    request_epoch: int
    chunk_id: int
    raw_text: str
    synthesis_text: str
    language: str
    voice_profile_id: str
    enqueue_time_ns: int = field(default_factory=time.monotonic_ns)
    output_mode: TTSOutputMode = TTSOutputMode.BOTH
    seed: int = 42

    def __post_init__(self) -> None:
        if not self.request_id.strip():
            raise ValueError("request_id 不能为空")
        if self.request_epoch < 0:
            raise ValueError("request_epoch 不能为负数")
        if self.chunk_id < 0:
            raise ValueError("chunk_id 不能为负数")
        if not self.synthesis_text.strip():
            raise ValueError("synthesis_text 不能为空")
        if not self.voice_profile_id.strip():
            raise ValueError("voice_profile_id 不能为空")

    @property
    def request_key(self) -> tuple[str, int]:
        return self.request_id, self.request_epoch


@dataclass
class TTSChunkResult:
    """TTS Worker 返回的单块结果及完整性能字段。"""

    request_id: str
    request_epoch: int
    chunk_id: int
    raw_text: str
    synthesis_text: str
    audio_bytes: bytes = b""
    wav_path: str = ""
    audio_sha256: str = ""
    sample_rate: int = 0
    audio_duration_s: float = 0.0
    first_chunk_ready_s: float = 0.0
    queue_wait_s: float = 0.0
    scheduler_wait_s: float = 0.0
    synthesis_s: float = 0.0
    serialization_s: float = 0.0
    output_write_s: float = 0.0
    rtf: float = 0.0
    device: str = ""
    cpu_threads: int = 0
    cpu_affinity: tuple[int, ...] = ()
    backend_kind: str = ""
    worker_pid: int = 0
    worker_epoch: int = 0
    model_load_count: int = 0
    voice_prompt_build_count: int = 0
    peak_rss_mb: float = 0.0
    peak_vram_allocated_mb: float = 0.0
    status: TTSChunkStatus = TTSChunkStatus.COMPLETED
    error: str = ""

    @property
    def request_key(self) -> tuple[str, int]:
        return self.request_id, self.request_epoch

    def to_record(self) -> dict[str, Any]:
        """转换为可写入 JSONL 的记录，明确排除音频字节。"""

        record = asdict(self)
        record.pop("audio_bytes", None)
        record["status"] = self.status.value
        return record
