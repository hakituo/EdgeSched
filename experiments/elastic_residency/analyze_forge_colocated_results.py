#!/usr/bin/env python3
"""离线复算 LLM+Forge 正式共置并冻结 G_safe/H_TTS/H_D。"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.elastic_residency.forge_colocated_capacity import (  # noqa: E402
    forge_colocated_metrics,
)
from experiments.elastic_residency.llm_forge_plan import (  # noqa: E402
    LlmForgeCandidate,
    screened_candidates,
)


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _validate_record(record: dict[str, Any]) -> None:
    checks = record.get("acceptance_checks", {})
    if not record.get("passed") or not checks or not all(checks.values()):
        raise ValueError("LLM+Forge 记录未通过全部 acceptance checks")
    if record.get("backend_kind") != "real_llama_cpp_gguf_plus_forge_http_api":
        raise ValueError("LLM+Forge 记录不是真实后端")
    if record.get("fallback_occurred"):
        raise ValueError("LLM+Forge 记录发生 fallback")
    image = record.get("forge_image", {})
    if image.get("size_bytes", 0) < 1024 or len(image.get("sha256", "")) != 64:
        raise ValueError("LLM+Forge 记录缺少有效图像大小或 SHA-256")
    runtime = record.get("runtime_evidence", {}).get("checks", {})
    if not runtime or not all(runtime.values()):
        raise ValueError("LLM+Forge 记录的新鲜度或 swap 门禁未通过")
    snapshots = (
        record.get("memory_forge_resident", {}),
        record.get("memory_b_llm", {}),
        record.get("memory_post_forge", {}),
    )
    if any(
        item.get("gpu_telemetry_backend") != "direct_nvml"
        or item.get("gpu_cli_subprocess_used")
        for item in snapshots
    ):
        raise ValueError("LLM+Forge 容量快照不是纯 direct NVML")


def analyze_forge_colocated(
    *,
    screen: dict[str, Any],
    tts_freeze: dict[str, Any],
    colocated: dict[str, Any],
    manifest: dict[str, Any],
    expected_runs_per_candidate: int = 5,
) -> dict[str, Any]:
    records = colocated.get("records", [])
    if not colocated.get("passed") or colocated.get("failure_count") != 0:
        raise ValueError("LLM+Forge summary 未整体通过或存在 failure")
    if colocated.get("evidence_scope") != "llm_forge_colocated_formal_repeated_matrix":
        raise ValueError("输入不是正式 LLM+Forge 重复矩阵")
    if len(records) != colocated.get("expected_records") or len(records) != colocated.get(
        "actual_records"
    ):
        raise ValueError("LLM+Forge 记录数与 summary 不一致")
    source = manifest.get("git", {}).get("source_copy", {})
    if (
        not manifest.get("git_provenance_complete")
        or source.get("dirty") is not False
        or source.get("status_entry_count") != 0
    ):
        raise ValueError("LLM+Forge Git provenance 不完整或不干净")

    grouped: dict[LlmForgeCandidate, list[dict[str, Any]]] = defaultdict(list)
    residuals: list[float] = []
    for record in records:
        _validate_record(record)
        candidate = LlmForgeCandidate(
            int(record["context_tokens"]), str(record["llm_config"])
        )
        grouped[candidate].append(record)
        residuals.append(float(record["colocated_metrics"]["colocated_positive_residual_mb"]))
    expected = set(screened_candidates(screen))
    if set(grouped) != expected:
        raise ValueError("正式 LLM+Forge 候选与筛选 H_D 不一致")
    if any(len(items) != expected_runs_per_candidate for items in grouped.values()):
        raise ValueError("每个 H_D 候选必须具有规定的重复轮数")
    for candidate, items in grouped.items():
        indexes = {int(item["run_index"]) for item in items}
        if indexes != set(range(1, expected_runs_per_candidate + 1)):
            raise ValueError(f"{candidate.label} 的 run_index 不完整或重复")
    ordinals = [int(item["matrix_execution_ordinal"]) for item in records]
    if set(ordinals) != set(range(1, len(records) + 1)):
        raise ValueError("matrix_execution_ordinal 不完整或重复")

    components = dict(tts_freeze["g_safe"]["components"])
    components["colocated_positive_residual_mb"] = max(
        float(components.get("colocated_positive_residual_mb") or 0.0),
        max(residuals),
    )
    final_g_safe = max(float(value) for value in components.values() if value is not None)
    previous_g_safe = float(tts_freeze["g_safe"]["value_mb"])
    g_safe_increase = final_g_safe - previous_g_safe

    candidates: dict[str, Any] = {}
    h_d: dict[str, Any] = {}
    for candidate in sorted(grouped, key=lambda item: item.context_tokens):
        corrected = []
        for record in grouped[candidate]:
            raw = record["colocated_metrics"]
            corrected.append(
                forge_colocated_metrics(
                    global_idle_reference_mb=raw["global_idle_reference_mb"],
                    b_llm_mb=raw["b_llm_mb"],
                    p_colocated_mb=raw["p_colocated_mb"],
                    gpu_total_mb=raw["gpu_total_mb"],
                    safety_margin_mb=final_g_safe,
                    llm_isolated_increment_mb=raw["llm_isolated_increment_mb"],
                    forge_isolated_increment_mb=raw["forge_isolated_increment_mb"],
                )
            )
        name = f"C{candidate.context_tokens}_{candidate.llm_config}_Forge"
        entry = {
            "context_tokens": candidate.context_tokens,
            "llm_config": candidate.llm_config,
            "runs": len(corrected),
            "all_capacity_admissible": all(
                item["capacity_admissible"] for item in corrected
            ),
            "p_colocated_max_mb": max(item["p_colocated_mb"] for item in corrected),
            "physical_headroom_min_mb": min(
                item["physical_headroom_mb"] for item in corrected
            ),
            "safe_capacity_headroom_min_mb": min(
                item["safe_capacity_headroom_mb"] for item in corrected
            ),
            "colocated_positive_residual_max_mb": max(
                item["colocated_positive_residual_mb"] for item in corrected
            ),
        }
        if not entry["all_capacity_admissible"]:
            raise ValueError(f"最终 G_safe 下 H_D 候选不可行: {name}")
        candidates[name] = entry
        h_d[str(candidate.context_tokens)] = {
            "llm_config": candidate.llm_config,
            "status": "frozen_real_colocated",
            "validated_runs": len(corrected),
            "safe_capacity_headroom_min_mb": entry[
                "safe_capacity_headroom_min_mb"
            ],
        }

    h_tts = json.loads(json.dumps(tts_freeze["h_tts"]))
    for configs in h_tts.values():
        for entry in configs.values():
            adjusted = float(entry["safe_capacity_headroom_min_mb"]) - g_safe_increase
            if adjusted < 0:
                raise ValueError("Forge 正残差增大 G_safe 后，既有 H_TTS 不再安全")
            entry["safe_capacity_headroom_min_mb"] = adjusted
            entry["status"] = "frozen_real_colocated"

    return {
        "schema_version": "edgesched.elastic_residency.v1",
        "analysis_name": "c3_all_colocated_freeze",
        "status": "g_safe_h_tts_h_d_frozen",
        "inputs": {
            "colocated_run_id": manifest.get("run_id"),
            "source_git_commit": source.get("commit"),
        },
        "evidence_validation": {
            "passed": True,
            "validated_records": len(records),
            "validated_candidates": len(grouped),
            "runs_per_candidate": expected_runs_per_candidate,
            "real_backend_without_fallback": True,
            "direct_nvml_only": True,
            "runtime_freshness_and_swap_passed": True,
        },
        "g_safe": {
            "value_mb": final_g_safe,
            "components": components,
            "pending_components": [],
            "formula": (
                "max(idle high water, allocator slack, repeat drift, "
                "all target-combination positive residuals)"
            ),
            "status": "frozen_all_target_combinations",
        },
        "capacity_semantics": {
            "formula": "(P_colocated-B_idle_reference)+G_safe<=gpu_total",
            "physical_peak_guard": "P_colocated<=gpu_total",
        },
        "candidates": candidates,
        "h_tts": h_tts,
        "h_d": h_d,
        "caveats": [
            "结果仍是 C3 工作证据，未经人工核验不得晋升 paper main。",
            "本冻结只处理容量与共置正残差，不证明端到端性能或调度收益。",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-analysis", required=True)
    parser.add_argument("--tts-freeze-analysis", required=True)
    parser.add_argument("--colocated-summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summary_path = Path(args.colocated_summary)
    result = analyze_forge_colocated(
        screen=load_json(args.screen_analysis),
        tts_freeze=load_json(args.tts_freeze_analysis),
        colocated=load_json(summary_path),
        manifest=load_json(summary_path.with_name("manifest.json")),
    )
    result["inputs"].update(
        screen_analysis_path=str(Path(args.screen_analysis).resolve()),
        tts_freeze_analysis_path=str(Path(args.tts_freeze_analysis).resolve()),
        colocated_summary_path=str(summary_path.resolve()),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"已写入 {output}")


if __name__ == "__main__":
    main()
