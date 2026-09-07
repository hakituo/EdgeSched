#!/usr/bin/env python3
"""离线复算 LLM+TTS 共置容量、正残差与 H_TTS 冻结状态。"""

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

from experiments.elastic_residency.colocated_batch_merge import (  # noqa: E402
    load_colocated_input,
)
from experiments.elastic_residency.colocated_capacity import (  # noqa: E402
    colocated_metrics,
)
from experiments.elastic_residency.small_screen_view import (  # noqa: E402
    SMALL,
    load_screen,
)


REAL_BACKEND = "real_llama_cpp_gguf_plus_qwen3_tts"


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _expected_tts_candidates(screen: dict[str, Any]) -> set[tuple[int, str, str]]:
    expected: set[tuple[int, str, str]] = set()
    for context, entry in screen["per_context"].items():
        for tts_config, h_tts in entry["h_tts"].items():
            llm_config = (
                "FULL" if h_tts["full_feasible"] else f"K{h_tts['h_layers']}"
            )
            expected.add((int(context), llm_config, tts_config))
    return expected


def _validate_measured_candidates(
    screen: dict[str, Any],
    grouped: dict[tuple[int, str, str], list[dict[str, Any]]],
) -> None:
    """small screen 只给容量预算与已画像档位，实测候选不能凭空出现。

    离线筛选的 H_TTS 边界本身待修正，因此这里不要求候选等于边界，
    只要求档位在已画像集合内，边界与实测的差异另行列出口供人工核对。
    """
    allowed = {
        int(context): set(entry.get("profiled_llm_configs", ()))
        for context, entry in screen["per_context"].items()
    }
    for context, llm_config, _tts_config in grouped:
        if llm_config not in allowed.get(context, set()):
            raise ValueError(f"共置候选 {llm_config} 不在 C{context} 已画像档位内")


def _screen_boundary(screen: dict[str, Any], context: int, tts_config: str) -> str | None:
    entry = screen["per_context"].get(str(context), {})
    h_tts = entry.get("h_tts", {}).get(tts_config)
    if not h_tts:
        return None
    if h_tts.get("full_feasible"):
        return "FULL"
    layers = h_tts.get("h_layers")
    return None if layers is None else f"K{layers}"


def _validate_record(record: dict[str, Any]) -> None:
    checks = record.get("acceptance_checks", {})
    if not record.get("passed") or not checks or not all(checks.values()):
        raise ValueError("共置记录未通过全部 acceptance checks")
    if record.get("backend_kind") != REAL_BACKEND or record.get(
        "fallback_occurred"
    ):
        raise ValueError("共置记录不是无 fallback 的真实 LLM+TTS 后端")
    baseline = record.get("memory_baseline", {})
    if baseline.get("gpu_telemetry_backend") != "direct_nvml" or baseline.get(
        "gpu_cli_subprocess_used"
    ):
        raise ValueError("共置记录未使用纯 direct NVML")
    runtime_checks = record.get("runtime_evidence", {}).get("checks", {})
    if not runtime_checks or not all(runtime_checks.values()):
        raise ValueError("共置记录的 GPU 新鲜度或 swap 门禁未通过")


def analyze_colocated(
    *,
    screen: dict[str, Any],
    colocated: dict[str, Any],
    manifest: dict[str, Any],
    expected_runs_per_candidate: int = 5,
    screen_schema: str = "legacy",
    inputs_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """复算有效共置记录；不修改原始 artifact。

    small schema 把候选判据放宽到已画像档位；inputs_extra 并入输出 inputs。
    """

    records = colocated.get("records", [])
    if not colocated.get("passed") or colocated.get("failure_count") != 0:
        raise ValueError("共置 summary 未整体通过或存在失败记录")
    if colocated.get("evidence_scope") != "llm_tts_colocated_formal_repeated_matrix":
        raise ValueError("输入不是正式重复共置矩阵")
    if len(records) != colocated.get("actual_records") or len(records) != colocated.get(
        "expected_records"
    ):
        raise ValueError("共置记录数与 summary 声明不一致")
    source = manifest.get("git", {}).get("source_copy", {})
    if not manifest.get("git_provenance_complete") or source.get("dirty") is not False:
        raise ValueError("共置 manifest 的 Git provenance 不完整或不干净")

    residuals: list[float] = []
    grouped: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    legacy_false_count = 0
    for record in records:
        _validate_record(record)
        raw_metrics = record["colocated_metrics"]
        isolated = raw_metrics.get("tts_isolated_increment_mb")
        if isolated is None:
            raise ValueError("正式共置记录缺少同配置隔离增量")
        residuals.append(
            max(0.0, raw_metrics["p_colocated_mb"] - raw_metrics["b_llm_mb"] - isolated)
        )
        legacy_false_count += not raw_metrics.get("capacity_admissible", False)
        key = (
            int(record["context_tokens"]),
            str(record["llm_config"]),
            str(record["tts_config"]),
        )
        grouped[key].append(record)

    if screen_schema == SMALL:
        _validate_measured_candidates(screen, grouped)
        expected = set(grouped)
    else:
        expected = _expected_tts_candidates(screen)
        if set(grouped) != expected:
            raise ValueError(
                f"正式共置候选与筛选 H_TTS 不一致: {set(grouped)} != {expected}"
            )
    if any(len(items) != expected_runs_per_candidate for items in grouped.values()):
        raise ValueError("每个共置候选的重复轮数不符合冻结要求")

    components = dict(screen["g_safe"]["components"])
    components["colocated_positive_residual_mb"] = max(residuals)
    g_safe_value = max(value for value in components.values() if value is not None)
    g_safe = {
        "value_mb": g_safe_value,
        "components": components,
        "pending_components": [],
        "formula": "max(idle 高水位, allocator slack, repeat drift, colocated 正残差)",
        "status": "frozen_for_llm_tts",
    }

    candidates: dict[str, Any] = {}
    h_tts: dict[str, dict[str, Any]] = defaultdict(dict)
    # 同一 (context, tts) 出现多个 llm 档时，键追加 llm 档名，避免后写覆盖先写。
    llm_by_context_tts: dict[tuple[str, str], set[str]] = defaultdict(set)
    for context, llm_config, tts_config in grouped:
        llm_by_context_tts[(str(context), tts_config)].add(llm_config)
    for key in sorted(grouped):
        context, llm_config, tts_config = key
        corrected = []
        for record in grouped[key]:
            raw = record["colocated_metrics"]
            corrected.append(
                colocated_metrics(
                    global_idle_baseline_mb=record["memory_baseline"][
                        "gpu_card_used_mb"
                    ],
                    b_llm_mb=raw["b_llm_mb"],
                    p_colocated_mb=raw["p_colocated_mb"],
                    gpu_total_mb=raw["gpu_total_mb"],
                    safety_margin_mb=g_safe_value,
                    tts_isolated_increment_mb=raw["tts_isolated_increment_mb"],
                )
            )
        name = f"C{context}_{llm_config}_{tts_config}"
        candidates[name] = {
            "context_tokens": context,
            "llm_config": llm_config,
            "tts_config": tts_config,
            "runs": len(corrected),
            "all_capacity_admissible": all(
                item["capacity_admissible"] for item in corrected
            ),
            "p_colocated_max_mb": max(item["p_colocated_mb"] for item in corrected),
            "colocated_card_delta_max_mb": max(
                item["colocated_card_delta_mb"] for item in corrected
            ),
            "physical_headroom_min_mb": min(
                item["physical_headroom_mb"] for item in corrected
            ),
            "safe_capacity_headroom_min_mb": min(
                item["capacity_headroom_mb"] for item in corrected
            ),
            "colocated_positive_residual_max_mb": max(
                item["colocated_positive_residual_mb"] for item in corrected
            ),
            "screen_boundary": _screen_boundary(screen, context, tts_config),
            "measured_beyond_screen_boundary": _screen_boundary(
                screen, context, tts_config
            )
            != llm_config,
        }
        if not candidates[name]["all_capacity_admissible"]:
            raise ValueError(f"修正口径后候选仍不满足安全容量: {name}")
        many_llm = len(llm_by_context_tts[(str(context), tts_config)]) > 1
        tts_key = f"{tts_config}_at_{llm_config}" if many_llm else tts_config
        h_tts[str(context)][tts_key] = {
            "llm_config": llm_config,
            "status": "frozen_real_colocated",
            "validated_runs": len(corrected),
            "safe_capacity_headroom_min_mb": candidates[name][
                "safe_capacity_headroom_min_mb"
            ],
        }

    if screen_schema == SMALL:
        # small screen 的 Diffusion 只有主机内存诊断，没有可复核的 H_D 档位。
        h_d = {
            context: {
                "screened_h_layers": None,
                "screened_full_feasible": None,
                "status": "pending_real_llm_forge_colocation",
                "note": "small screen 只有 Diffusion 主机内存诊断，未给出 H_D 档位。",
            }
            for context in screen["per_context"]
        }
    else:
        h_d = {
            context: {
                "screened_h_layers": entry["h_d"].get("h_layers"),
                "screened_full_feasible": entry["h_d"].get("full_feasible"),
                "status": "pending_real_llm_forge_colocation",
            }
            for context, entry in screen["per_context"].items()
        }
    return {
        "schema_version": "edgesched.elastic_residency.v1",
        "analysis_name": "c3_llm_tts_colocated_offline_reanalysis",
        "screen_schema": screen_schema,
        "status": "h_tts_frozen_h_d_pending",
        "inputs": {
            "screen_analysis_name": screen.get("analysis_name"),
            "colocated_run_id": manifest.get("run_id"),
            "source_git_commit": source.get("commit"),
            **(inputs_extra or {}),
        },
        "evidence_validation": {
            "passed": True,
            "validated_records": len(records),
            "validated_candidates": len(grouped),
            "runs_per_candidate": expected_runs_per_candidate,
            "real_backend_without_fallback": True,
            "direct_nvml_only": True,
            "runtime_freshness_and_swap_passed": True,
            "legacy_double_counted_capacity_false_records": legacy_false_count,
            "raw_measurements_reused": True,
            "rerun_required": False,
        },
        "capacity_semantics": {
            "formula": "(P_colocated - B_idle) + G_safe <= gpu_total",
            "physical_peak_guard": "P_colocated <= gpu_total",
            "reason": "G_safe 已含 idle GPU baseline high water，绝对峰值不可再叠加完整 G_safe",
        },
        "g_safe": g_safe,
        "candidates": candidates,
        "h_tts": dict(h_tts),
        "h_d": h_d,
        "caveats": [
            "H_TTS 已由真实 LLM+TTS 共置矩阵验证；原始 artifact 不回写。",
            "H_D 仍只有隔离容量筛选，必须完成真实 LLM+Forge 共置后才能冻结。",
            "本结果仍是探索/收尾证据，未经人工核验不得晋升 paper main。",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-analysis", required=True)
    parser.add_argument("--colocated-summary", help="单批次 summary.json；与 --colocated-summaries 二选一。")
    parser.add_argument(
        "--colocated-summaries", nargs="+", help="多批次合并：各正式批次的 summary.json。"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--screen-schema",
        choices=("auto", "legacy", "small"),
        default="auto",
        help="auto 按结构判别；small 适配 small 9B 离线筛选产物。",
    )
    args = parser.parse_args()
    screen, schema = load_screen(load_json(args.screen_analysis), args.screen_schema)
    colocated, manifest, inputs_extra = load_colocated_input(
        args.colocated_summary, args.colocated_summaries
    )
    result = analyze_colocated(
        screen=screen,
        colocated=colocated,
        manifest=manifest,
        screen_schema=schema,
        inputs_extra=inputs_extra,
    )
    result["inputs"]["screen_analysis_path"] = str(Path(args.screen_analysis).resolve())
    # 多批次时逐批次路径已在 merged_batches 清单里，不再重复单值字段。
    if args.colocated_summary:
        result["inputs"]["colocated_summary_path"] = str(
            Path(args.colocated_summary).resolve()
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
