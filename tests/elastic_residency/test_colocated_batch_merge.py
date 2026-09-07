"""多批次共置合并视图的硬校验门禁测试。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from experiments.elastic_residency.colocated_batch_merge import load_batches

COMMIT_A = "a" * 64
COMMIT_B = "b" * 64


def _record(context: int, llm: str, tts: str) -> dict[str, Any]:
    return {
        "context_tokens": context,
        "llm_config": llm,
        "tts_config": tts,
        "passed": True,
    }


def _write_batch(
    root: Path,
    name: str,
    *,
    commit: str,
    candidates: list[tuple[int, str, str]],
    runs: int = 5,
    scope: str = "llm_tts_colocated_formal_repeated_matrix",
    passed: bool = True,
    failure_count: int = 0,
    dirty: bool = False,
) -> Path:
    records = [
        _record(context, llm, tts)
        for (context, llm, tts) in candidates
        for _ in range(runs)
    ]
    colocated = {
        "passed": passed,
        "failure_count": failure_count,
        "actual_records": len(records),
        "expected_records": len(records),
        "evidence_scope": scope,
        "records": records,
    }
    manifest = {
        "run_id": name,
        "git_provenance_complete": True,
        "git": {
            "source_copy": {
                "commit": commit,
                "dirty": dirty,
                "status_entry_count": 0,
            }
        },
    }
    batch_dir = root / name
    batch_dir.mkdir(parents=True)
    (batch_dir / "summary.json").write_text(
        json.dumps(colocated), encoding="utf-8"
    )
    (batch_dir / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return batch_dir / "summary.json"


def _two_valid_batches(tmp_path: Path) -> list[str]:
    first = _write_batch(
        tmp_path,
        "batch_a",
        commit=COMMIT_A,
        candidates=[(512, "K24", "T0"), (512, "K24", "T3")],
    )
    second = _write_batch(
        tmp_path,
        "batch_b",
        commit=COMMIT_B,
        candidates=[(512, "K0", "T0"), (512, "K24", "T4")],
    )
    return [str(first), str(second)]


def test_merge_valid_batches_with_distinct_commits(tmp_path: Path) -> None:
    merged, manifest, batches = load_batches(_two_valid_batches(tmp_path))

    assert merged["actual_records"] == 20
    assert merged["expected_records"] == 20
    assert len(batches) == 2
    assert batches[0]["candidates"] == ["C512_K24_T0", "C512_K24_T3"]
    # 多提交不能冒充单一来源：commit 串接且批次清单完整保留。
    assert manifest["git"]["source_copy"]["commit"] == f"{COMMIT_A}|{COMMIT_B}"
    assert {b["source_git_commit"] for b in batches} == {COMMIT_A, COMMIT_B}


def test_merge_rejects_duplicate_candidates_across_batches(tmp_path: Path) -> None:
    first = _write_batch(
        tmp_path, "batch_a", commit=COMMIT_A, candidates=[(512, "K24", "T0")]
    )
    second = _write_batch(
        tmp_path, "batch_b", commit=COMMIT_B, candidates=[(512, "K24", "T0")]
    )
    with pytest.raises(ValueError, match="重复"):
        load_batches([str(first), str(second)])


def test_merge_rejects_non_formal_or_failed_batches(tmp_path: Path) -> None:
    smoke = _write_batch(
        tmp_path,
        "smoke",
        commit=COMMIT_A,
        candidates=[(512, "K0", "T4")],
        scope="llm_tts_colocated_smoke_or_pilot",
    )
    failed = _write_batch(
        tmp_path,
        "failed",
        commit=COMMIT_A,
        candidates=[(512, "K0", "T4")],
        passed=False,
        failure_count=1,
    )
    dirty = _write_batch(
        tmp_path, "dirty", commit=COMMIT_A, candidates=[(512, "K0", "T4")], dirty=True
    )
    with pytest.raises(ValueError, match="正式重复矩阵"):
        load_batches([str(smoke)])
    with pytest.raises(ValueError, match="失败记录"):
        load_batches([str(failed)])
    with pytest.raises(ValueError, match="provenance"):
        load_batches([str(dirty)])


def test_merge_requires_five_runs_per_candidate(tmp_path: Path) -> None:
    short = _write_batch(
        tmp_path,
        "short",
        commit=COMMIT_A,
        candidates=[(512, "K24", "T0")],
        runs=3,
    )
    with pytest.raises(ValueError, match="5 轮"):
        load_batches([str(short)])


def test_merge_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="至少需要"):
        load_batches([])
