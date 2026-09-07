"""双模型计划、复制保护与比较口径的无模型验证。"""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.elastic_residency.compare_9b_models import audit_unit, unit_command
from experiments.elastic_residency.compare_9b_support import (
    copy_candidate, make_plan, markdown_report, measurement, summarize,
)


def test_plan_covers_balanced_pairs_with_rotating_order():
    plan = make_plan(["FULL", "K0"], 3)
    assert len(plan) == 12
    assert [p["model"] for p in plan[:2]] == ["large", "small"]
    assert [p["model"] for p in plan[4:6]] == ["small", "large"]
    assert plan[4]["config"] == "K0"
    assert len({(p["model"], p["config"], p["round"]) for p in plan}) == 12


@pytest.mark.parametrize("configs,repeats", [([], 3), (["FULL", "FULL"], 3), (["K4"], 3), (["FULL"], 0)])
def test_invalid_plan_rejected(configs, repeats):
    with pytest.raises(ValueError):
        make_plan(configs, repeats)


def test_copy_is_idempotent_and_preserves_different_target(tmp_path):
    source, target = tmp_path / "source", tmp_path / "native" / "small"
    source.write_bytes(b"frozen model fixture")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    copy_candidate(source, target, digest)
    copy_candidate(source, target, digest)
    assert target.read_bytes() == source.read_bytes()
    target.write_bytes(b"other model")
    with pytest.raises(ValueError):
        copy_candidate(source, target, digest)
    assert target.read_bytes() == b"other model"
    assert not list(target.parent.glob(".compare-copy-*"))


def test_bad_source_never_creates_target(tmp_path):
    source, target = tmp_path / "source", tmp_path / "small"
    source.write_bytes(b"wrong")
    with pytest.raises(ValueError):
        copy_candidate(source, target, "0" * 64)
    assert not target.exists()


def test_measurement_uses_lifetime_peak_and_keeps_missing_kv():
    record = {"memory_baseline": {"gpu_card_used_mb": 100}, "load_s": 2,
              "inference": {"ttft_s": 1, "prefill_tokens_per_s": 50, "decode_tokens_per_s": 20},
              "memory_after_inference": {"rss_bytes": 2**20}, "kv_save_s": 0.1, "kv_restore_s": 0.2}
    timeline = [{"gpu_card_used_mb": gpu, "process_rss_bytes": rss * 2**20,
                 "system_available_bytes": 20 * 2**20, "system_swap_used_bytes": 0}
                for gpu, rss in [(100, 1), (500, 4), (200, 2)]]
    result = measurement(record, timeline)
    assert result["gpu_peak_minus_baseline_mib"] == 400
    assert result["rss_peak_mib"] == 4
    assert result["rss_after_inference_mib"] == 1
    assert result["kv_gpu_mib"] is None


def test_summary_does_not_hide_incomplete_or_mismatched_pairs():
    plan = make_plan(["FULL"], 1)
    rows = [{**unit, "passed": True, "prompt_tokens_sha256": "same",
             "metrics": {"decode_tps": 10 + i, "kv_cpu_mib": None}} for i, unit in enumerate(plan)]
    result = summarize(rows, plan, [])
    assert result["passed"]
    assert result["pairs"][0]["small_minus_large"]["decode_tps"] == 1
    assert result["by_model_config"]["large|FULL"]["metrics"]["kv_cpu_mib"] is None
    assert not summarize(rows[:1], plan, [])["passed"]
    rows[1]["prompt_tokens_sha256"] = "different"
    assert not summarize(rows, plan, [])["passed"]
    assert not summarize(rows, plan, [{"error": "failed"}])["passed"]


def test_markdown_report_keeps_group_label_in_one_table_cell():
    rows = [{**unit, "passed": True, "prompt_tokens_sha256": "same",
             "metrics": {"decode_tps": 10}} for unit in make_plan(["FULL", "K0"], 1)]
    report = markdown_report(summarize(rows, make_plan(["FULL", "K0"], 1), []))
    table = [line for line in report.splitlines() if line.startswith("|")]
    assert len(table) == 6
    assert all(line.count("|") == 9 for line in table)
    assert "| small / FULL | 1 |" in report


def test_command_forwards_model_core_budgets_and_snapshot():
    args = SimpleNamespace(context_tokens=512, generation_tokens=128, llm_cores=14,
                           system_cores=2, child_timeout=1800, source_git_commit="a" * 40,
                           source_git_dirty="true", source_git_status_entry_count=1)
    unit = {"model": "small", "config": "K0", "round": 2, "id": "unit2"}
    command = unit_command(args, Path("/result"), unit, {"small": {"path": "/native/small.gguf"}})
    assert command[command.index("--model") + 1] == "/native/small.gguf"
    assert command[command.index("--seed") + 1] == "43"
    assert command[command.index("--source-git-snapshot") + 1].endswith("source_git_snapshot.tar.gz")
    assert "--continue-on-failure" not in command
    assert "--mmproj" not in command


@pytest.mark.parametrize("tamper", [None, "model", "child", "snapshot", "checks", "output_hash", "output_preview"])
def test_audit_requires_consistent_child_identity_and_provenance(tmp_path, tamper):
    def write(name, value):
        (tmp_path / name).write_text(json.dumps(value), encoding="utf-8")
    record = {"config": "FULL", "context_tokens": 512, "generation_tokens": 128,
              "llm_physical_cores": 14, "system_physical_cores": 2,
              "acceptance_checks": {"valid": tamper != "checks"}, "performance_target_passed": True,
              "memory_baseline": {"gpu_card_used_mb": 100}, "load_s": 2,
              "inference": {"ttft_s": 1, "prefill_tokens_per_s": 50, "decode_tokens_per_s": 20,
                            "output_text_preview": "固定续写前缀",
                            "output_text_sha256": hashlib.sha256("固定续写前缀及其余全文".encode()).hexdigest()},
              "prompt_tokens_sha256": "prompt",
              "memory_after_inference": {"rss_bytes": 2**20}, "kv_save_s": 0.1, "kv_restore_s": 0.2}
    if tamper == "output_hash":
        record["inference"]["output_text_sha256"] = "invalid"
    if tamper == "output_preview":
        del record["inference"]["output_text_preview"]
    child = dict(record)
    child["timeline"] = [{"gpu_card_used_mb": 400, "process_rss_bytes": 2**20,
                          "system_available_bytes": 10 * 2**20, "system_swap_used_bytes": 0}]
    if tamper == "child":
        child["load_s"] = 999
    write("C512_FULL_r1_child.json", child)
    record["matrix_execution_ordinal"] = 1
    write("summary.json", {"passed": True, "actual_records": 1, "failure_count": 0,
                           "git_provenance_complete": True, "records": [record]})
    (tmp_path / "iterations.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    (tmp_path / "failures.jsonl").write_text("", encoding="utf-8")
    write("run_config.json", {"backends": [{"model_sha256": "wrong" if tamper == "model" else "expected",
                                           "is_real": True, "fallback_occurred": False}]})
    (tmp_path / "snapshot").write_bytes(b"changed" if tamper == "snapshot" else b"original")
    write("manifest.json", {"git": {"source_copy": {"snapshot": "snapshot",
                           "snapshot_sha256": hashlib.sha256(b"original").hexdigest()}}})
    args = SimpleNamespace(context_tokens=512, generation_tokens=128, llm_cores=14, system_cores=2)
    unit = {"model": "large", "config": "FULL", "round": 1}
    if tamper:
        with pytest.raises(ValueError):
            audit_unit(tmp_path, unit, {"sha256": "expected"}, args)
    else:
        result = audit_unit(tmp_path, unit, {"sha256": "expected"}, args)
        assert result["passed"]
        assert result["metrics"]["gpu_peak_minus_baseline_mib"] == 300
        assert result["output_text_preview"] == "固定续写前缀"
        assert result["output_text_sha256"] == record["inference"]["output_text_sha256"]
        assert "output_text" not in result
