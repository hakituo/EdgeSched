"""辅助画像的冻结输入与文件身份；合成语料只用于性能测试。"""

from __future__ import annotations

import argparse
import hashlib
import json
import wave
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: Path) -> dict:
    return {"path": str(path.resolve()), "size_bytes": path.stat().st_size,
            "sha256": sha256(path)}


def freeze_inputs(destination: Path, source_wav: Path) -> None:
    """从已有中文 TTS WAV 截取固定长度，保留来源，不伪称真人语音集。"""
    with wave.open(str(source_wav)) as stream:
        params = stream.getparams()
        if params.sampwidth != 2 or params.nframes / params.framerate < 10:
            raise ValueError("来源必须是至少 10 秒的 PCM16 中文 WAV")
        pcm = stream.readframes(params.nframes)
    destination.mkdir(parents=True, exist_ok=False)
    topics = ("语音识别", "显存迁移", "向量检索", "线程亲和性", "图像生成", "调度延迟")
    corpus = [{"id": f"doc_{i:04d}", "text":
               f"{topics[i % len(topics)]}实验记录{i}：固定输入并记录资源占用、"
               f"任务耗时与结果校验。配置编号{i % 37}，重复轮次{i % 5}。"}
              for i in range(1000)]
    (destination / "corpus.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in corpus),
        encoding="utf-8")
    clips = []
    for seconds in (6, 8, 10):
        target = destination / f"tts_crop_{seconds}s.wav"
        with wave.open(str(target), "wb") as stream:
            stream.setparams(params)
            stream.writeframes(pcm[:seconds * params.framerate * params.nchannels * 2])
        clips.append(target.name)
    payload = {
        "schema": "edgesched.auxiliary_inputs.v1", "corpus": "corpus.jsonl",
        "queries": [{"text": f"如何测量{topic}的资源占用和耗时？",
                     "audio": clips[i % len(clips)]} for i, topic in enumerate(topics)],
        "input_kind": "synthetic_engineering_corpus_and_cropped_chinese_tts",
        "limitations": "用于固定负载性能画像；不证明检索质量或真实人声 ASR 泛化",
        "audio_source": file_identity(source_wav),
        "files": {p.name: sha256(p) for p in sorted(destination.iterdir())},
    }
    (destination / "inputs.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_inputs(path: Path) -> tuple[dict, list[dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    root = path.resolve().parent
    required = {data["corpus"], *(q["audio"] for q in data["queries"])}
    if not required.issubset(data["files"]):
        raise ValueError("输入文件缺少冻结哈希")
    for name, expected in data["files"].items():
        target = (root / name).resolve()
        if root not in target.parents or sha256(target) != expected:
            raise ValueError(f"输入路径或哈希不合法: {name}")
    corpus = [json.loads(line) for line in (root / data["corpus"]).read_text(
        encoding="utf-8").splitlines()]
    if len(corpus) != 1000 or len({row["id"] for row in corpus}) != 1000:
        raise ValueError("RAG 语料必须有 1000 个唯一文档")
    if len(data["queries"]) < 6 or any(not q["text"].strip() for q in data["queries"]):
        raise ValueError("至少冻结六个非空查询")
    for name in {q["audio"] for q in data["queries"]}:
        with wave.open(str(root / name)) as stream:
            duration = stream.getnframes() / stream.getframerate()
            if not 5 <= duration <= 10 or stream.getsampwidth() != 2:
                raise ValueError("ASR 输入必须是 5–10 秒 PCM16 WAV")
    return data, corpus


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-wav", required=True, type=Path)
    options = parser.parse_args()
    freeze_inputs(options.output, options.source_wav)
