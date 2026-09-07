"""假 ONNX 后端验证缓存边界和计时口径，不加载任何真实模型。"""

import sys
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.elastic_residency.auxiliary_bge import BgeCachedRerank, BgeRerank
from experiments.elastic_residency.auxiliary_report import build_plan, cached_evidence_valid, summarize


@pytest.fixture
def backend(monkeypatch):
    state = SimpleNamespace(calls=[], bad=False)
    class Tokenizer:
        @classmethod
        def from_file(cls, path):
            return cls()
        def enable_truncation(self, **kwargs):
            assert kwargs == {"max_length": 128}
        def enable_padding(self, **kwargs):
            assert kwargs["length"] == 128
        def encode_batch(self, texts):
            state.calls.append(list(texts))
            return [SimpleNamespace(ids=[sum(map(ord, text)) % 512] + [0] * 127,
                                    attention_mask=[1] * 128, type_ids=[0] * 128)
                    for text in texts]
    class Session:
        def get_inputs(self):
            return [SimpleNamespace(name="input_ids")]
        def get_providers(self):
            return ["CPUExecutionProvider"]
        def run(self, names, inputs):
            ids = inputs["input_ids"]
            assert ids.shape[1] == 128
            values = np.eye(512, dtype=np.float32)[ids[:, 0]]
            if state.bad:
                values[:] = np.nan
            return [values]
    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(
        SessionOptions=SimpleNamespace, ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL=0),
        InferenceSession=lambda *args, **kwargs: Session()))
    monkeypatch.setitem(sys.modules, "tokenizers", SimpleNamespace(Tokenizer=Tokenizer))
    state.corpus = [{"id": str(i), "text": f"候选文档{i}"} for i in range(32)]
    return state


def make(backend, cls=BgeCachedRerank):
    return cls(corpus=backend.corpus, model=Path("fake"), cores=2)


def test_cache_encodes_32_once_and_each_query_once_even_when_repeated(backend):
    task = make(backend)
    assert backend.calls == []
    prep = task.prepare()
    first = task.run({"text": "同一个查询"})
    second = task.run({"text": "同一个查询"})
    assert list(map(len, backend.calls)) == [32, 1, 1]
    assert first == second
    assert prep["embedding_shape"] == [32, 512]
    assert prep["cache_bytes"] == 32 * 512 * 4
    assert not task.candidate_embeddings.flags.writeable
    assert first["candidates_reencoded"] is False


def test_cached_and_reencoded_ranking_are_equivalent(backend):
    cached = make(backend)
    cached.prepare()
    full = make(backend, BgeRerank)
    query = {"text": "候选文档7"}
    assert cached.run(query)["top_ids"] == full.run(query)["top_ids"]
    assert list(map(len, backend.calls)) == [32, 1, 33]


def test_missing_or_repeated_prepare_is_rejected(backend):
    task = make(backend)
    with pytest.raises(RuntimeError, match="预计算"):
        task.run({"text": "查询"})
    task.prepare()
    with pytest.raises(RuntimeError, match="重复计算"):
        task.prepare()


def test_invalid_precomputed_embeddings_fail_closed(backend):
    task = make(backend)
    backend.bad = True
    with pytest.raises(ValueError, match="向量"):
        task.prepare()
    assert getattr(task, "candidate_embeddings", None) is None


def test_candidate_content_is_snapshotted_per_worker(backend):
    task = make(backend)
    backend.corpus[0]["text"] = "外部改动"
    task.prepare()
    assert backend.calls[0][0] == "候选文档0"


def cached_rows(backend, plan):
    task = make(backend)
    prep = {**task.prepare(), "elapsed_s": 9.0}
    event = {"elapsed_s": .01, "output": task.run({"text": "查询"})}
    return [{**cell, "measurement_valid": True, "preparation": prep,
             "events": [event] * 6} for cell in plan]


def test_precompute_cost_is_separate_from_request_sla(backend):
    plan = build_plan(["bge_cached"], [2], 5, 42)
    assert len(plan) == 15
    rows = cached_rows(backend, plan)
    summary = summarize(plan, rows, 6)
    assert summary["measurement_valid"]
    assert summary["minimum_core_budgets"][0]["minimum_cores"] == 1
    assert all(g["p95"] == .01 and g["sla"] == .15 and
               g["candidate_precompute_s"]["mean"] == 9 for g in summary["groups"])


@pytest.mark.parametrize("field,value", [("candidates_reencoded", True),
    ("query_cache_enabled", True), ("cache_sha256", "a" * 64),
    ("embedding_shape", [33, 512])])
def test_wrong_cache_contract_is_rejected(backend, field, value):
    plan = build_plan(["bge_cached"], [2], 5, 42)
    rows = cached_rows(backend, plan)
    rows[0]["events"][0]["output"][field] = value
    assert not cached_evidence_valid(rows[0])
    assert not summarize(plan, rows, 6)["measurement_valid"]


def test_cached_model_identity_uses_bge_model_path(tmp_path):
    from experiments.elastic_residency.auxiliary_core_budget import model_files
    (tmp_path / "onnx").mkdir()
    (tmp_path / "onnx/model.onnx").write_bytes(b"fake-model-identity-only")
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    identities = model_files(SimpleNamespace(bge_model=tmp_path, whisper_model=Path("unused"),
                                            tasks=["bge_cached"]))
    assert set(identities) == {"bge_cached"}
    assert len(identities["bge_cached"]) == 2


def test_modes_stay_separate_and_mixed_csv_keeps_precompute_cost(backend, tmp_path):
    from experiments.elastic_residency.analyze_auxiliary_core_budget import write_csv
    full_plan = build_plan(["bge"], [2], 5, 42)
    cached_plan = build_plan(["bge_cached"], [2], 5, 42)
    full_rows = [{**cell, "measurement_valid": True,
                  "events": [{"elapsed_s": .2}] * 6} for cell in full_plan]
    result = summarize(full_plan + cached_plan,
                       full_rows + cached_rows(backend, cached_plan), 6)
    statuses = {r["task"]: r["status"] for r in result["minimum_core_budgets"]}
    assert statuses == {"bge": "infeasible", "bge_cached": "feasible"}
    path = tmp_path / "mixed.csv"
    write_csv(result, path)
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 6
    assert rows[0]["candidate_precompute_s"] == ""
    assert json.loads(rows[-1]["candidate_precompute_s"])["mean"] == 9
