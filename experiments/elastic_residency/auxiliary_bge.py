"""BGE 全量重编码与候选向量缓存画像；保持相同输入和 padding 以隔离缓存收益。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


class BgeRerank:
    def __init__(self, corpus: list[dict], model: Path, cores: int, **_: object):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        options = ort.SessionOptions()
        options.intra_op_num_threads = cores
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(str(model / "onnx/model.onnx"),
                                            options, providers=["CPUExecutionProvider"])
        self.tokenizer = Tokenizer.from_file(str(model / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=128)
        self.tokenizer.enable_padding(length=128, pad_id=0, pad_token="[PAD]")
        self.candidates = tuple({"id": row["id"], "text": row["text"]} for row in corpus[:32])
        if len(self.candidates) != 32 or len({r["id"] for r in self.candidates}) != 32:
            raise ValueError("BGE 必须有 32 个唯一候选")

    def _encode(self, texts: list[str]):
        import numpy as np

        encoded = self.tokenizer.encode_batch(texts)
        arrays = {"input_ids": np.asarray([e.ids for e in encoded], dtype=np.int64),
                  "attention_mask": np.asarray([e.attention_mask for e in encoded],
                                               dtype=np.int64),
                  "token_type_ids": np.asarray([e.type_ids for e in encoded], dtype=np.int64)}
        result = self.session.run(None, {i.name: arrays[i.name]
                                         for i in self.session.get_inputs()})[0]
        embeddings = result[:, 0] if result.ndim == 3 else result
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        if (embeddings.shape != (len(texts), 512) or not np.isfinite(embeddings).all()
                or not np.isfinite(norms).all() or not (norms > 0).all()):
            raise ValueError("BGE 向量形状、有限性或范数不合法")
        return embeddings / np.maximum(norms, 1e-12)

    def _rank(self, query_embedding, candidates, encoded_count: int) -> dict:
        import numpy as np

        order = np.argsort(-(candidates @ query_embedding), kind="stable")[:8]
        return {"output_valid": True, "embedding_shape": [encoded_count, 512],
                "top_ids": [self.candidates[i]["id"] for i in order],
                "backend": "onnxruntime_cpu_dense_embedding_rerank",
                "providers": self.session.get_providers(), "max_length": 128,
                "padded_sequence_length": 128, "pooling": "cls_l2",
                "candidates_reencoded": encoded_count == 33}

    def run(self, query: dict) -> dict:
        embeddings = self._encode([query["text"], *(row["text"] for row in self.candidates)])
        return self._rank(embeddings[0], embeddings[1:], 33)


class BgeCachedRerank(BgeRerank):
    def prepare(self) -> dict:
        """每个独立进程仅预计算一次，调用方单独计时；不缓存 query 编码结果。"""
        if getattr(self, "candidate_embeddings", None) is not None:
            raise RuntimeError("候选缓存已构建，拒绝重复计算")
        embeddings = self._encode([row["text"] for row in self.candidates])
        embeddings.setflags(write=False)
        self.candidate_embeddings = embeddings
        self.candidate_identity = hashlib.sha256(json.dumps(
            self.candidates, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        self.cache_sha256 = hashlib.sha256(embeddings.tobytes()).hexdigest()
        return {"kind": "candidate_precompute", "candidate_count": 32,
                "embedding_shape": list(embeddings.shape), "cache_bytes": embeddings.nbytes,
                "cache_sha256": self.cache_sha256,
                "candidate_identity_sha256": self.candidate_identity,
                "included_in_request_latency": False, "query_cache_enabled": False}

    def run(self, query: dict) -> dict:
        if getattr(self, "candidate_embeddings", None) is None:
            raise RuntimeError("必须在计时请求前显式预计算候选，禁止静默回退到重编码")
        embeddings = self._encode([query["text"]])
        result = self._rank(embeddings[0], self.candidate_embeddings, 1)
        return {**result, "backend": "onnxruntime_cpu_cached_candidate_rerank",
                "candidate_embedding_shape": [32, 512], "query_cache_enabled": False,
                "cache_sha256": self.cache_sha256,
                "candidate_identity_sha256": self.candidate_identity}
