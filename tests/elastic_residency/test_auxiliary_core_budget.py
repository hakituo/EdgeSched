"""验证核预算口径、拒绝伪成功、输入冻结和后端惰性消费。"""

import wave
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.elastic_residency.auxiliary_inputs import freeze_inputs, load_inputs
from experiments.elastic_residency.auxiliary_report import build_plan, summarize
from experiments.elastic_residency.auxiliary_workloads import JsonRag, WhisperAsr


def records(plan, elapsed=.01):
    return [{**cell, "measurement_valid": True,
             "events": [{"elapsed_s": elapsed, "rtf": elapsed} for _ in range(6)]}
            for cell in plan]


def test_plan_reproducible_complete_and_shuffled():
    plan = build_plan(["json_rag", "bge", "whisper"], [0, 1, 2, 4], 5, 42)
    assert len(plan) == 160
    assert plan == build_plan(["json_rag", "bge", "whisper"], [0, 1, 2, 4], 5, 42)
    assert plan != build_plan(["json_rag", "bge", "whisper"], [0, 1, 2, 4], 5, 43)
    assert len({tuple(sorted(p.items())) for p in plan}) == 160


@pytest.mark.parametrize("tasks,reserves,repeats", [(["x"], [2], 5),
    (["bge"], [3], 5), (["bge"], [2, 2], 5), (["bge"], [0], 0)])
def test_invalid_plan_rejected(tasks, reserves, repeats):
    with pytest.raises(ValueError):
        build_plan(tasks, reserves, repeats, 42)


def test_sla_failure_is_valid_measurement_but_not_formal_rsys():
    plan = build_plan(["whisper"], [2], 5, 42)
    result = summarize(plan, records(plan, 1.1), 6)
    assert result["measurement_valid"]
    assert all(g["status"] == "infeasible" for g in result["groups"])
    assert result["minimum_core_budgets"][0]["minimum_cores"] is None
    assert result["formal_r_sys"]["passed"] is None


def test_missing_or_failed_cell_never_yields_minimum_core_recommendation():
    plan = build_plan(["bge"], [2], 5, 42)
    rows = records(plan)
    rows[0]["measurement_valid"] = False
    assert summarize(plan, rows, 6)["minimum_core_budgets"][0]["status"] == "incomplete"
    assert not summarize(plan, rows[:-1], 6)["measurement_valid"]


def test_duplicate_repeat_is_not_complete():
    plan = build_plan(["bge"], [2], 5, 42)
    rows = records(plan)
    rows[0]["repeat"] = 99
    assert not summarize(plan, rows, 6)["measurement_valid"]


def test_smoke_does_not_claim_p95():
    plan = build_plan(["bge"], [2], 1, 42)
    result = summarize(plan, records(plan), 6)
    assert result["measurement_valid"]
    assert all(g["p95"] is None for g in result["groups"])


def test_smallest_feasible_core_budget():
    plan = build_plan(["bge"], [2], 5, 42)
    rows = records(plan)
    for row in rows:
        if row["cores"] == 1:
            row["events"] = [{"elapsed_s": .2} for _ in range(6)]
    assert summarize(plan, rows, 6)["minimum_core_budgets"][0]["minimum_cores"] == 2


def test_frozen_inputs_detect_tampering(tmp_path: Path):
    source = tmp_path / "source.wav"
    with wave.open(str(source), "wb") as stream:
        stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        stream.writeframes(b"\x01\x00" * 160000)
    target = tmp_path / "frozen"
    freeze_inputs(target, source)
    data, corpus = load_inputs(target / "inputs.json")
    assert len(corpus) == 1000 and len(data["queries"]) == 6
    (target / "corpus.jsonl").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="哈希"):
        load_inputs(target / "inputs.json")


def test_json_rag_stable_top32():
    task = JsonRag([{"id": str(i), "text": "固定语料"} for i in range(1000)])
    result = task.run({"text": "语料"})
    assert result["top_ids"] == [str(i) for i in range(32)]


def test_whisper_consumes_lazy_segments():
    seen = []
    def segments():
        seen.append(True)
        yield SimpleNamespace(text="测试")
    task = WhisperAsr.__new__(WhisperAsr)
    task.input_root = Path(".")
    task.model = SimpleNamespace(transcribe=lambda *args, **kwargs:
                                  (segments(), SimpleNamespace(duration=6)))
    assert task.run({"audio": "fixture.wav"})["output_valid"]
    assert seen == [True]


@pytest.mark.parametrize("available_cores", [1, 4])
def test_child_failure_returns_partial_and_full_leases(tmp_path, monkeypatch, available_cores):
    from experiments.elastic_residency import auxiliary_child as child
    from experiments.elastic_residency.cpu_topology import CPUTopology

    monkeypatch.setattr(child, "allowed_topology", lambda: CPUTopology.from_sibling_groups(
        [(i,) for i in range(available_cores)]))
    monkeypatch.setattr(child.os, "sched_getaffinity", lambda pid: {0}, raising=False)
    monkeypatch.setattr(child, "load_inputs", lambda path: ({"queries": []}, []))
    @contextmanager
    def scope(*args):
        yield (0,)
    monkeypatch.setattr(child, "process_affinity_scope", scope)
    monkeypatch.setattr(child, "ResourceSampler", lambda **kwargs:
                        SimpleNamespace(start=lambda: None, stop=lambda: {}, samples=[]))
    def broken(**kwargs):
        raise RuntimeError("injected_model_load_failure")
    monkeypatch.setitem(child.WORKLOADS, "bge", broken)
    config = {"cell": {"task": "bge", "cores": 1, "system_cores": 1, "repeat": 1},
              "inputs": "unused", "models": {"bge": "unused"}}
    result = child.run_child(config, tmp_path / "failure")
    assert not result["measurement_valid"]
    assert result["lease_after"]["active_lease_count"] == 0
    assert result["lease_after"]["free_physical_core_count"] == available_cores
    assert result["checks"]["affinity_restored"]


def test_short_cpu_window_requires_fresh_valid_difference_sample():
    from experiments.elastic_residency.auxiliary_child import finish_cpu_window
    sampler = SimpleNamespace(samples=[
        SimpleNamespace(monotonic_ns=1, cpu_sample_valid=False),
        SimpleNamespace(monotonic_ns=3, cpu_sample_valid=True)])
    finish_cpu_window(sampler, 2)
    with pytest.raises(RuntimeError, match="测量后有效样本"):
        finish_cpu_window(sampler, 4, timeout_s=.001)
