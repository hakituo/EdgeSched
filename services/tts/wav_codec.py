"""PCM16 WAV 编码工具，供 worker 与实验执行器共用。"""

from __future__ import annotations

import io
import wave
from typing import Any


def decode_pcm16_wav(payload: bytes) -> tuple[Any, int]:
    """把 PCM16 WAV 解回 ``(float32 样本, 采样率)``，供播放侧消费。"""

    import numpy as np
    import wave
    import io

    with wave.open(io.BytesIO(payload), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        raw = wav_file.readframes(wav_file.getnframes())
    if sample_width != 2:
        raise ValueError(f"只支持 PCM16 WAV，当前 sample_width={sample_width}")
    array = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        array = array.reshape(-1, channels).mean(axis=1)
    return array, sample_rate


def pcm16_wav_bytes(samples: Any, sample_rate: int) -> bytes:
    """将模型浮点输出转换为标准单声道 PCM16 WAV。"""

    import numpy as np

    array = np.asarray(samples, dtype=np.float32).reshape(-1)
    array = np.clip(array, -1.0, 1.0)
    pcm = (array * 32767.0).astype("<i2").tobytes()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)
    return buffer.getvalue()
