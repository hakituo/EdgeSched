#!/usr/bin/env python3
"""C 阶段收尾分析：G_safe 校准、H_TTS/H_D 计算与临界补测建议。

输入：
- C2 驻留曲线 summary.json（``by_context_config.<ctx>.<config>.gpu_delta_mb``）；
- B2 TTS 组件矩阵 summary.json（``by_config.<T>.max_card_delta_mb``、
  ``torch_peak_allocated_mb`` / ``torch_peak_reserved_mb``）；
- 可选 Forge b=1 复测 summary.json（forge_b1_reprofile.py 产物，
  ``global_idle_high_water_mb`` 与 ``peak_card_used_mb.max``），
  或用 ``--legacy-forge-peak-mb`` 提供旧口径绝对峰值（仅估算）。

输出 JSON 包含：
- ``g_safe``：按执行计划 3.3 公式的分量与取值（缺数据分量记为 pending）；
- ``per_context``：每个上下文的 M_L 曲线、H_TTS（按 B2 Pareto 候选逐个计算）
  与 H_D；同时给出 mean 基准推荐值与 max 基准保守值；
- ``critical_refinement``：每个临界档位的 k±2/k±4 补测建议。

本脚本只做容量筛选，最终可行性仍必须由目标组合真实同时加载运行证明
（执行计划 3.3）；colocated 残差分量在真实共置数据出现前保持 pending。

用法：
  python experiments/elastic_residency/analyze_results.py \\
      --c2-summary experiment_results/elastic_residency/c2/<run>/summary.json \\
      --b2-summary experiment_results/elastic_residency/b2/<run>/summary.json \\
      [--forge-summary experiment_results/elastic_residency/forge/<run>/summary.json] \\
      [--output experiment_results/elastic_residency/c3_analysis/<name>.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

SCHEMA_VERSION = "edgesched.elastic_residency.v1"


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def extract_llm_memory_curve(
    c2_summary: Mapping[str, Any],
) -> dict[str, dict[str, dict[str, float]]]:
    """提取 {context: {config: {"mean", "max", "min"}}} 的整卡增量曲线。"""

    by_context = c2_summary["by_context_config"]
    curve: dict[str, dict[str, dict[str, float]]] = {}
    for context, configs in by_context.items():
        entry: dict[str, dict[str, float]] = {}
        for config, stats in configs.items():
            delta = stats["gpu_delta_mb"]
            entry[config] = {
                "mean": float(delta["mean"]),
                "max": float(delta["max"]),
                "min": float(delta["min"]),
            }
        curve[str(context)] = entry
    return curve


def extract_tts_profiles(
    b2_summary: Mapping[str, Any],
) -> dict[str, dict[str, float]]:
    """提取 {config: {max_card_delta_mb, allocator_slack_mb, ...}}。"""

    profiles: dict[str, dict[str, float]] = {}
    for config, stats in b2_summary["by_config"].items():
        allocated = stats.get("torch_peak_allocated_mb")
        reserved = stats.get("torch_peak_reserved_mb")
        slack = reserved - allocated if allocated is not None and reserved is not None else None
        profiles[config] = {
            "max_card_delta_mb": float(stats["max_card_delta_mb"]),
            "torch_peak_allocated_mb": (
                float(allocated) if allocated is not None else None
            ),
            "torch_peak_reserved_mb": (
                float(reserved) if reserved is not None else None
            ),
            "allocator_slack_mb": float(slack) if slack is not None else None,
        }
    return profiles


def extract_forge_profile(
    forge_summary: Mapping[str, Any] | None,
    legacy_peak_mb: float | None,
    legacy_idle_mb: float | None,
) -> dict[str, Any]:
    """提取 M_D(b=1) 增量与空闲高水位。"""

    if forge_summary is not None:
        peak = forge_summary.get("peak_card_used_mb", {}).get("max")
        idle = forge_summary.get("global_idle_high_water_mb")
        if peak is None or idle is None:
            raise ValueError("Forge 复测 summary 缺少 peak/global_idle 字段")
        return {
            "source": "forge_reprofile",
            "run_id": forge_summary.get("run_id"),
            "peak_card_used_mb": float(peak),
            "global_idle_high_water_mb": float(idle),
            "m_d_incremental_estimate_mb": float(peak) - float(idle),
        }
    if legacy_peak_mb is not None:
        idle = float(legacy_idle_mb if legacy_idle_mb is not None else 0.0)
        return {
            "source": "legacy_estimate",
            "peak_card_used_mb": float(legacy_peak_mb),
            "global_idle_high_water_mb": idle,
            "m_d_incremental_estimate_mb": float(legacy_peak_mb) - idle,
        }
    return {
        "source": "missing",
        "peak_card_used_mb": None,
        "global_idle_high_water_mb": None,
        "m_d_incremental_estimate_mb": None,
    }


def compute_g_safe(
    llm_curve: Mapping[str, Mapping[str, Mapping[str, float]]],
    tts_profiles: Mapping[str, Mapping[str, float | None]],
    idle_high_water_mb: float | None,
) -> dict[str, Any]:
    """按执行计划 3.3 计算 G_safe：取各分量最大值，缺数据分量记 pending。"""

    slack_values = [
        profile["allocator_slack_mb"]
        for profile in tts_profiles.values()
        if profile.get("allocator_slack_mb") is not None
    ]
    allocator_slack = max(slack_values) if slack_values else None

    drift_values: list[float] = []
    for configs in llm_curve.values():
        for delta in configs.values():
            drift_values.append(delta["max"] - delta["min"])
    repeat_drift = max(drift_values) if drift_values else None

    components = {
        "idle_gpu_baseline_high_water_mb": (
            float(idle_high_water_mb) if idle_high_water_mb is not None else None
        ),
        "torch_allocator_slack_mb": (
            float(allocator_slack) if allocator_slack is not None else None
        ),
        "llm_repeat_run_drift_mb": (
            float(repeat_drift) if repeat_drift is not None else None
        ),
        # 单模型峰值相加与真实共置峰值之间的最大正残差：
        # 在真实共置验证（阶段 E 前置）完成前没有数据，保持 pending。
        "colocated_positive_residual_mb": None,
    }
    available = [value for value in components.values() if value is not None]
    return {
        "value_mb": max(available) if available else 0.0,
        "components": components,
        "pending_components": [
            name for name, value in components.items() if value is None
        ],
        "formula": "max(可计算分量)；colocated 残差在真实共置验证前不计入",
    }


def _config_layer_count(name: str) -> int | None:
    """K0 → 0，K24 → 24；FULL 返回 None（单独档位）。"""

    normalized = name.upper()
    if normalized == "K0":
        return 0
    if normalized.startswith("K") and normalized[1:].isdigit():
        return int(normalized[1:])
    return None


def compute_h(
    m_l: Mapping[str, Mapping[str, float]],
    companion_mb: float,
    g_safe_mb: float,
    gpu_total_mb: float,
    basis: str = "mean",
) -> dict[str, Any]:
    """计算 H = max k: M_L(k) + companion + G_safe <= V_GPU。

    返回 H（整数层数；K0 也不可行时为 None）、FULL 是否可行
    以及逐配置占用表。
    """

    budget = gpu_total_mb - companion_mb - g_safe_mb
    usage: dict[str, float] = {
        config: float(delta[basis]) for config, delta in m_l.items()
    }
    feasible_layers: list[int] = []
    for config, memory in usage.items():
        layers = _config_layer_count(config)
        if layers is None:
            continue
        if memory <= budget:
            feasible_layers.append(layers)
    full_feasible = usage.get("FULL") is not None and usage["FULL"] <= budget
    return {
        "h_layers": max(feasible_layers) if feasible_layers else None,
        "full_feasible": full_feasible,
        "budget_for_m_l_mb": budget,
        "m_l_usage_mb": usage,
    }


def suggest_refinements(
    anchors: list[int | None],
    model_layers: int,
    measured_layers: set[int],
) -> dict[str, Any]:
    """对每个临界档位生成 k±2/k±4 补测建议（裁剪去重、排除已测）。"""

    suggestions: dict[str, list[int]] = {}
    merged: set[int] = set()
    for anchor in anchors:
        if anchor is None:
            continue
        offsets = (-4, -2, 2, 4)
        values = sorted(
            {
                max(0, min(model_layers, anchor + offset))
                for offset in offsets
            }
            - measured_layers
        )
        suggestions[str(anchor)] = values
        merged.update(values)
    return {
        "by_anchor": suggestions,
        "merged_sorted": sorted(merged),
        "excluded_measured_layers": sorted(measured_layers),
    }


def analyze(
    *,
    c2_summary: Mapping[str, Any],
    b2_summary: Mapping[str, Any],
    forge_summary: Mapping[str, Any] | None,
    legacy_forge_peak_mb: float | None,
    legacy_forge_idle_mb: float | None,
    gpu_total_mb: float,
    model_layers: int,
    tts_configs: list[str],
) -> dict[str, Any]:
    llm_curve = extract_llm_memory_curve(c2_summary)
    tts_profiles = extract_tts_profiles(b2_summary)
    forge = extract_forge_profile(forge_summary, legacy_forge_peak_mb, legacy_forge_idle_mb)
    g_safe = compute_g_safe(llm_curve, tts_profiles, forge["global_idle_high_water_mb"])

    per_context: dict[str, Any] = {}
    anchors: set[int | None] = set()
    for context, m_l in llm_curve.items():
        entry: dict[str, Any] = {
            "m_l_mb": {config: delta["mean"] for config, delta in m_l.items()},
        }
        h_tts: dict[str, Any] = {}
        for tts_config in tts_configs:
            if tts_config not in tts_profiles:
                raise ValueError(f"B2 summary 缺少配置 {tts_config}")
            h = compute_h(
                m_l,
                tts_profiles[tts_config]["max_card_delta_mb"],
                g_safe["value_mb"],
                gpu_total_mb,
                basis="mean",
            )
            h_conservative = compute_h(
                m_l,
                tts_profiles[tts_config]["max_card_delta_mb"],
                g_safe["value_mb"],
                gpu_total_mb,
                basis="max",
            )
            h_tts[tts_config] = {
                "m_t_mb": tts_profiles[tts_config]["max_card_delta_mb"],
                "h_layers": h["h_layers"],
                "full_feasible": h["full_feasible"],
                "budget_for_m_l_mb": h["budget_for_m_l_mb"],
                "conservative_max_basis": {
                    "h_layers": h_conservative["h_layers"],
                    "full_feasible": h_conservative["full_feasible"],
                },
            }
            anchors.add(h["h_layers"])
        if forge["m_d_incremental_estimate_mb"] is not None:
            h_d = compute_h(
                m_l,
                forge["m_d_incremental_estimate_mb"],
                g_safe["value_mb"],
                gpu_total_mb,
                basis="mean",
            )
            h_d_conservative = compute_h(
                m_l,
                forge["m_d_incremental_estimate_mb"],
                g_safe["value_mb"],
                gpu_total_mb,
                basis="max",
            )
            entry["h_d"] = {
                "m_d_mb": forge["m_d_incremental_estimate_mb"],
                "h_layers": h_d["h_layers"],
                "full_feasible": h_d["full_feasible"],
                "budget_for_m_l_mb": h_d["budget_for_m_l_mb"],
                "conservative_max_basis": {
                    "h_layers": h_d_conservative["h_layers"],
                    "full_feasible": h_d_conservative["full_feasible"],
                },
            }
            anchors.add(h_d["h_layers"])
        else:
            entry["h_d"] = {"status": "pending_forge_reprofile"}
        entry["h_tts"] = h_tts
        per_context[context] = entry

    measured_layers = {
        layers
        for configs in llm_curve.values()
        for config in configs
        if (layers := _config_layer_count(config)) is not None
    }
    # anchors 可能含 None（无可行档位）；补测建议只围绕具体临界层。
    refinement = suggest_refinements(
        sorted(anchor for anchor in anchors if anchor is not None),
        model_layers,
        measured_layers,
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "analysis_name": "c_stage_gsafe_h_calculation",
        "inputs": {
            "c2_run_id": c2_summary.get("run_id"),
            "b2_run_id": b2_summary.get("run_id"),
            "forge": {
                "source": forge["source"],
                "run_id": forge.get("run_id"),
            },
        },
        "gpu_total_mb": gpu_total_mb,
        "model_layers": model_layers,
        "m_basis_primary": "mean",
        "g_safe": g_safe,
        "forge": forge,
        "per_context": per_context,
        "critical_refinement": refinement,
        "caveats": [
            "H 计算只做容量筛选；最终可行性必须由目标组合真实同时加载并运行证明。",
            "colocated 正残差分量在真实共置验证完成前保持 pending，G_safe 可能被低估。",
            "M_T 使用 B2 的 max_card_delta_mb（15 条中的最大整卡增量，保守口径）。",
            "Forge 未复测时 M_D 只能使用 legacy 估算，禁止用于冻结 H_D。",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c2-summary", required=True)
    parser.add_argument("--b2-summary", required=True)
    parser.add_argument("--forge-summary")
    parser.add_argument("--legacy-forge-peak-mb", type=float)
    parser.add_argument("--legacy-forge-idle-mb", type=float, default=314.0)
    parser.add_argument("--gpu-total-mb", type=float, default=8151.0)
    parser.add_argument("--model-layers", type=int, default=32)
    parser.add_argument("--tts-configs", default="T0,T3")
    parser.add_argument("--output")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    forge_summary = load_json(args.forge_summary) if args.forge_summary else None
    result = analyze(
        c2_summary=load_json(args.c2_summary),
        b2_summary=load_json(args.b2_summary),
        forge_summary=forge_summary,
        legacy_forge_peak_mb=args.legacy_forge_peak_mb,
        legacy_forge_idle_mb=args.legacy_forge_idle_mb,
        gpu_total_mb=args.gpu_total_mb,
        model_layers=args.model_layers,
        tts_configs=[item.strip() for item in args.tts_configs.split(",") if item.strip()],
    )
    result["inputs"]["c2_summary_path"] = str(Path(args.c2_summary).resolve())
    result["inputs"]["b2_summary_path"] = str(Path(args.b2_summary).resolve())
    if args.forge_summary:
        result["inputs"]["forge_summary_path"] = str(
            Path(args.forge_summary).resolve()
        )
    encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(encoded + "\n", encoding="utf-8")
        print(f"已写入 {output_path}")
    else:
        print(encoded)


if __name__ == "__main__":
    main()
