"""真实 CPU 辅助后端；重库仅在子进程绑定物理核后导入。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .auxiliary_bge import BgeCachedRerank, BgeRerank


class JsonRag:
    def __init__(self, corpus: list[dict], **_: object):
        # 冻结驻内存 JSON 检索，计时不包含磁盘读取和索引建立。
        self.rows = corpus
        self.terms = [set(row["text"]) for row in corpus]

    def run(self, query: dict) -> dict:
        terms = set(query["text"])
        scores = [len(terms & entry) / max(1, len(terms | entry))
                  for entry in self.terms]
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:32]
        return {"output_valid": len(order) == 32,
                "top_ids": [self.rows[i]["id"] for i in order],
                "backend": "json_character_jaccard_top32"}


class WhisperAsr:
    def __init__(self, model: Path, cores: int, input_root: Path, **_: object):
        from faster_whisper import WhisperModel

        self.model = WhisperModel(str(model), device="cpu", compute_type="int8",
                                  cpu_threads=cores, num_workers=1,
                                  local_files_only=True)
        self.input_root = input_root

    def run(self, query: dict) -> dict:
        segments, info = self.model.transcribe(
            str(self.input_root / query["audio"]), language="zh", beam_size=1,
            temperature=0, vad_filter=False, condition_on_previous_text=False)
        # 转写是惰性生成器，必须完全消费后才能停止计时。
        segments = list(segments)
        text = "".join(segment.text for segment in segments).strip()
        return {"output_valid": bool(text), "audio_duration_s": info.duration,
                "text": text, "segment_count": len(segments),
                "backend": "faster_whisper_cpu_int8", "beam_size": 1}


WORKLOADS = {"json_rag": JsonRag, "bge": BgeRerank, "bge_cached": BgeCachedRerank,
             "whisper": WhisperAsr}


def output_digest(result: dict) -> str:
    return hashlib.sha256(json.dumps(result, sort_keys=True,
                                     ensure_ascii=False).encode()).hexdigest()
