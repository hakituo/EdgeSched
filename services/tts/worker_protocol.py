"""TTS 主进程代理与 spawn 子进程之间的协议。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from domain.models.tts import VoiceCloneProfile


class WorkerState(str, Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    READY = "ready"
    BUSY = "busy"
    RELOADING = "reloading"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass(frozen=True)
class TTSWorkerConfig:
    model_path: str
    device: str
    voice_profiles: tuple[VoiceCloneProfile, ...]
    output_dir: str
    cpu_threads: int = 0
    cpu_affinity: tuple[int, ...] = ()
    dtype: str = "auto"
    attention_backend: str = "sdpa"
    backend_kind: str = "real_qwen3_tts"
    worker_epoch: int = 1
    start_timeout_s: float = 180.0
    stop_timeout_s: float = 15.0
    fake_synthesis_s: float = 0.01
    generation_kwargs: dict[str, Any] = field(
        default_factory=lambda: {
            "do_sample": False,
            "max_new_tokens": 1024,
            "repetition_penalty": 1.1,
        }
    )

    def __post_init__(self) -> None:
        if self.backend_kind not in {"real_qwen3_tts", "test_fake_tts"}:
            raise ValueError(f"不支持的 backend_kind: {self.backend_kind}")
        if not self.voice_profiles:
            raise ValueError("至少需要一个 VoiceCloneProfile")
        if self.device != "cpu" and not self.device.startswith("cuda"):
            raise ValueError(f"不支持的 TTS 设备: {self.device}")


@dataclass
class TTSWorkerHealth:
    state: WorkerState
    worker_pid: int = 0
    worker_epoch: int = 0
    device: str = ""
    backend_kind: str = ""
    cpu_threads: int = 0
    cpu_affinity: tuple[int, ...] = ()
    model_load_count: int = 0
    voice_prompt_build_count: int = 0
    model_load_s: float = 0.0
    voice_prompt_build_s: float = 0.0
    last_error: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = vars(self).copy()
        data["state"] = self.state.value
        data["cpu_affinity"] = list(self.cpu_affinity)
        return data


@dataclass
class DeviceTransitionMetrics:
    source_device: str
    target_device: str
    drain_s: float = 0.0
    process_stop_s: float = 0.0
    model_load_s: float = 0.0
    voice_prompt_build_s: float = 0.0
    ready_s: float = 0.0
    total_transition_s: float = 0.0
    worker_epoch: int = 0
