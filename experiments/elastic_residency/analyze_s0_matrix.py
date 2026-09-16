"""S0 正式成对比较的离线分析入口（纯标准库，Windows 侧可复算）。

用法：

    python -m experiments.elastic_residency.analyze_s0_matrix \
        --run-dir experiment_results/elastic_residency/s0_pipeline/<run_id>

只读取产物：先硬校验协议版本与冻结 placement 集，再按 ``run_index`` 配对，
回答三个冻结问题（是否改善、是否只是协变量、哪些值得进动态切换），
并把设备体验指标统一标为 pending，不参与比较。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.elastic_residency.s0_matrix_report import (
    BASELINE_NON_MIXED,
    FORMAL_STARTUP_POLICY,
    REFERENCE_SAME_LLM,
    aggregate,
    answer_questions,
    confound_checks,
    extract_metrics,
    paired_deltas,
    startup_policy_comparison,
)
from experiments.elastic_residency.s0_pipeline.config import (
    S0_PLACEMENTS,
    S0_PROTOCOL_VERSION,
    S0_STARTUP_POLICIES,
)
from experiments.elastic_residency.s0_pipeline.measurement_scope import (
    SCOPE_VERSION,
    STATUS_PENDING,
)

SCHEMA_VERSION = "edgesched.s0_matrix_analysis.v1"


def load_records(run_dir: Path) -> list[dict[str, Any]]:
    iterations = run_dir / "iterations.jsonl"
    if not iterations.is_file():
        raise FileNotFoundError(f"缺少 iterations.jsonl: {iterations}")
    records = []
    for line in iterations.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    if not records:
        raise ValueError("iterations.jsonl 为空，无法做成对比较")
    return records


def verify_protocol(run_dir: Path) -> dict[str, Any]:
    """硬校验：正式比较必须来自冻结协议，不允许看完结果再改口径。"""

    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"缺少 manifest.json: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    version = manifest.get("protocol_version")
    scope_version = manifest.get("measurement_scope_version")
    if version != S0_PROTOCOL_VERSION:
        raise ValueError(
            f"协议版本不一致：产物 {version!r} vs 冻结 {S0_PROTOCOL_VERSION!r}"
        )
    if scope_version != SCOPE_VERSION:
        raise ValueError(
            f"测量范围版本不一致：产物 {scope_version!r} vs 冻结 {SCOPE_VERSION!r}"
        )
    placements = manifest.get("protocol", {}).get("repetitions", {}).get(
        "formal_placements", []
    )
    if sorted(placements) != sorted(S0_PLACEMENTS):
        raise ValueError(f"冻结 placement 集不一致：{placements}")
    return {
        "protocol_version": version,
        "measurement_scope_version": scope_version,
        "freeze_status": manifest.get("freeze_status", {}),
        "startup_policies_frozen": manifest.get("protocol", {})
        .get("startup_policy", {}),
    }


def analyze(run_dir: Path) -> dict[str, Any]:
    protocol = verify_protocol(run_dir)
    records = load_records(run_dir)
    runs = [extract_metrics(record) for record in records]
    aggregates = aggregate(runs)
    paired: dict[str, dict[str, Any]] = {}
    targets = sorted({item["placement"] for item in runs})
    for target in targets:
        if target == REFERENCE_SAME_LLM:
            continue
        for baseline in (REFERENCE_SAME_LLM, BASELINE_NON_MIXED):
            pair = paired_deltas(runs, baseline=baseline, target=target)
            if pair["paired_run_count"] > 0:
                paired[f"{baseline}->{target}"] = pair
    confounds = confound_checks(runs)
    answers = answer_questions(aggregates, paired, confounds)
    startup_comparison = startup_policy_comparison(runs)
    policies = sorted({item["startup_policy"] for item in runs})
    pending = [
        key
        for key, item in aggregates.items()
        if STATUS_PENDING in {str(name) for name in item["device_status_counts"]}
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "run_dir": str(run_dir),
        **protocol,
        "run_count": len(runs),
        "passed_count": sum(1 for item in runs if item["passed"]),
        "placements": targets,
        "startup_policies_present": policies,
        "formal_startup_policy": FORMAL_STARTUP_POLICY,
        "frozen_startup_policies": list(S0_STARTUP_POLICIES),
        "aggregates": aggregates,
        "paired": paired,
        "startup_policy_comparison": startup_comparison,
        "answers": answers,
        "device_metrics": {
            "status": STATUS_PENDING if pending else "trusted",
            "pending_placements": pending,
            "note": (
                "设备体验指标未通过时钟验证时保持 pending："
                "实际播放时刻与设备 underflow 不参与本次比较"
            ),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-json", default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    report = analyze(run_dir)
    output = (
        Path(args.output_json).expanduser().resolve()
        if args.output_json
        else run_dir / "s0_matrix_analysis.json"
    )
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "analysis": str(output),
                "run_count": report["run_count"],
                "passed_count": report["passed_count"],
                "startup_policies": report["startup_policies_present"],
                "device_metrics_status": report["device_metrics"]["status"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
