"""把多个正式 LLM+TTS 共置批次合并为单次复算可消费的视图。

只做读取与跨批次硬校验，不修改原始 artifact：
S0 的证据分散在多个 run_id（每批次一个），而复算入口按单批次设计；
这里负责批次级门禁（正式矩阵、通过、provenance、候选不重复、轮数一致）
与记录合并，合并视图再交给 analyze_colocated 做逐记录复算。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

FORMAL_SCOPE = "llm_tts_colocated_formal_repeated_matrix"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_key(record: dict[str, Any]) -> tuple[int, str, str]:
    return (
        int(record["context_tokens"]),
        str(record["llm_config"]),
        str(record["tts_config"]),
    )


def load_batches(
    summary_paths: list[str],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """返回 (合并 colocated 视图, 合成 manifest, 批次清单)；任何硬校验失败即抛错。"""
    if not summary_paths:
        raise ValueError("至少需要一个 colocated summary")
    records: list[dict[str, Any]] = []
    batches: list[dict[str, Any]] = []
    seen: set[tuple[int, str, str]] = set()
    runs_per_candidate: int | None = None
    commits: list[str] = []
    for raw_path in summary_paths:
        path = Path(raw_path)
        colocated = _load(path)
        manifest = _load(path.with_name("manifest.json"))
        if colocated.get("evidence_scope") != FORMAL_SCOPE:
            raise ValueError(f"{path.name} 不是正式重复矩阵，禁止并入冻结口径")
        if not colocated.get("passed") or colocated.get("failure_count") != 0:
            raise ValueError(f"{path.name} 整体未通过或存在失败记录")
        source = manifest.get("git", {}).get("source_copy", {})
        if not manifest.get("git_provenance_complete") or source.get("dirty") is not False:
            raise ValueError(f"{path.name} 的 Git provenance 不完整或不干净")
        batch_records = colocated.get("records", [])
        if (
            len(batch_records) != colocated.get("actual_records")
            or len(batch_records) != colocated.get("expected_records")
        ):
            raise ValueError(f"{path.name} 的记录数与 summary 声明不一致")
        batch_keys = {_candidate_key(record) for record in batch_records}
        if seen & batch_keys:
            raise ValueError(f"候选 {sorted(seen & batch_keys)} 在多个批次中重复")
        batch_runs = len(batch_records) / len(batch_keys)
        if runs_per_candidate is None:
            runs_per_candidate = batch_runs
        elif batch_runs != runs_per_candidate:
            raise ValueError(
                f"{path.name} 的每候选轮数 {batch_runs} 与先前批次 {runs_per_candidate} 不一致"
            )
        seen |= batch_keys
        records.extend(batch_records)
        commit = str(source.get("commit", ""))
        if commit and commit not in commits:
            commits.append(commit)
        batches.append(
            {
                "run_id": manifest.get("run_id"),
                "summary_path": str(path.resolve()),
                "source_git_commit": commit,
                "candidates": sorted(f"C{c}_{k}_{t}" for c, k, t in batch_keys),
                "records": len(batch_records),
            }
        )
    if runs_per_candidate != 5:
        raise ValueError(f"冻结口径要求每候选 5 轮，实际为 {runs_per_candidate}")
    merged = {
        "schema_version": "edgesched.elastic_residency.v1",
        "evidence_scope": FORMAL_SCOPE,
        "passed": True,
        "failure_count": 0,
        "actual_records": len(records),
        "expected_records": len(records),
        "records": records,
    }
    manifest = {
        "run_id": "merged:" + ",".join(str(b["run_id"]) for b in batches),
        "git_provenance_complete": True,
        "git": {
            "source_copy": {
                # 多批次可能来自不同干净提交；单值时保持原语义，多值用 | 连接并保留清单。
                "commit": commits[0] if len(commits) == 1 else "|".join(commits),
                "dirty": False,
                "status_entry_count": 0,
            }
        },
    }
    return merged, manifest, batches


def load_colocated_input(
    single: str | None, multi: list[str] | None
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """CLI 分发：单批次直读，多批次走合并；二者必须二选一。"""
    if bool(single) == bool(multi):
        raise SystemExit("--colocated-summary 与 --colocated-summaries 必须二选一")
    if multi:
        merged, manifest, batches = load_batches(multi)
        return merged, manifest, {"merged_batches": batches}
    path = Path(single)
    return _load(path), _load(path.with_name("manifest.json")), {}
