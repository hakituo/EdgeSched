"""共置实验输出 WAV 的内容身份与文件验收。"""

from pathlib import Path
from typing import Any

from experiments.tts.profiling_utils import sha256_file


def output_wav_artifact(path: Path, audio_duration_s: float) -> dict[str, Any]:
    """冻结路径、大小与内容哈希；缺失或空文件不生成有效哈希。"""

    exists = path.is_file()
    size_bytes = path.stat().st_size if exists else 0
    digest = sha256_file(path) if size_bytes > 0 else None
    return {
        "path": str(path),
        "size_bytes": size_bytes,
        "sha256": digest,
        "checks": {
            "audio_written": exists and size_bytes > 0 and audio_duration_s > 0,
            "audio_sha256_recorded": (
                isinstance(digest, str) and len(digest) == 64
            ),
        },
    }
