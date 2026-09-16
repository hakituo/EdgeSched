"""S0 成对比较的判定部分：启动策略对照与三个冻结问题。

指标抽取、聚合与配对差见 ``s0_matrix_metrics``；本模块只做相对容差判定与
结论组织。纯标准库，不加载模型、不依赖 GPU。

判定只用调度主指标；设备体验指标一律作为 pending 记录，不参与比较。
"""

from __future__ import annotations

from typing import Any

from experiments.elastic_residency.s0_matrix_metrics import (  # noqa: F401
    FORMAL_STARTUP_POLICY,
    aggregate,
    confound_checks,
    extract_metrics,
    key_of,
    paired_deltas_by_key,
)
from experiments.elastic_residency.s0_pipeline.config import FROZEN_PROTOCOL

BASELINE_NON_MIXED = FROZEN_PROTOCOL["comparison_rule"]["baseline_non_mixed"]
REFERENCE_SAME_LLM = FROZEN_PROTOCOL["comparison_rule"]["reference_same_llm"]

#: 比较用的主指标：越小越好。
LOWER_IS_BETTER = (
    "first_sentence_ready_s",
    "last_sentence_ready_s",
    "rtf_mean",
    "supply_fault_episodes",
    "supply_fault_samples",
)

PREBUFFER_POLICY = FROZEN_PROTOCOL["startup_policy"]["comparison"][1]


def paired_deltas(
    runs: list[dict[str, Any]],
    *,
    baseline: str,
    target: str,
    policy: str = FORMAL_STARTUP_POLICY,
) -> dict[str, Any]:
    """同一启动策略下、两个 placement 的配对差（正式比较用）。"""

    summary = paired_deltas_by_key(
        runs,
        baseline_key=f"{baseline}|{policy}",
        target_key=f"{target}|{policy}",
    )
    return {"baseline": baseline, "target": target, "policy": policy, **summary}


def startup_policy_comparison(
    runs: list[dict[str, Any]],
    *,
    prebuffer_policy: str = PREBUFFER_POLICY,
) -> dict[str, Any]:
    """同一 placement 下「全量预缓冲 vs 首句即播」的等待代价与供给缺口。"""

    placements = sorted({item["placement"] for item in runs})
    comparisons: dict[str, Any] = {}
    for placement in placements:
        policies = {
            item["startup_policy"] for item in runs if item["placement"] == placement
        }
        if not {FORMAL_STARTUP_POLICY, prebuffer_policy} <= policies:
            continue
        pair = paired_deltas_by_key(
            runs,
            baseline_key=f"{placement}|{FORMAL_STARTUP_POLICY}",
            target_key=f"{placement}|{prebuffer_policy}",
        )
        comparisons[placement] = {
            "wait_cost_proxy_s": (pair.get("proxy_first_playback_s") or {}).get(
                "mean_delta"
            ),
            "first_sentence_ready_delta_s": (
                pair.get("first_sentence_ready_s") or {}
            ).get("mean_delta"),
            "supply_fault_episode_delta": (
                pair.get("supply_fault_episodes") or {}
            ).get("mean_delta"),
            "starve_ratio_in_window_delta": (
                pair.get("starve_ratio_in_window") or {}
            ).get("mean_delta"),
            "prebuffered_audio_s": (
                pair.get("prebuffered_audio_s") or {}
            ).get("mean_delta"),
            "text_identical_all": pair.get("text_identical_all"),
            "paired_run_count": pair.get("paired_run_count"),
            "pair": pair,
        }
    return {
        "baseline_policy": FORMAL_STARTUP_POLICY,
        "target_policy": prebuffer_policy,
        "comparisons": comparisons,
        "note": (
            "预缓冲只改变开播时机：首句波形就绪应基本不变，"
            "窗口内供给缺口应下降，代价是首播等待变长；"
            "首播时间的设备语义仍为 proxy/pending"
        ),
    }


def _tolerance() -> float:
    return float(FROZEN_PROTOCOL["comparison_rule"]["relative_tolerance"])


def _is_improvement(delta: float | None, baseline_value: float | None) -> bool:
    """只把「变小」且超过相对容差的差算作改善。

    主指标都是越小越好，因此 ``delta >= 0`` 一律不算改善——否则「变慢」会被
    误判成收益。缺失值同样不算。
    """

    if delta is None or baseline_value in (None, 0):
        return False
    if delta >= 0:
        return False
    return abs(delta) >= _tolerance() * abs(float(baseline_value))


def answer_questions(
    aggregates: dict[str, dict[str, Any]],
    paired: dict[str, dict[str, Any]],
    confounds: dict[str, Any],
) -> dict[str, Any]:
    """回答三个冻结问题：是否改善、是否只是协变量、哪些值得进动态研究。"""

    formal_keys = [
        key
        for key, item in aggregates.items()
        if item["startup_policy"] == FORMAL_STARTUP_POLICY
    ]
    same_llm = f"{REFERENCE_SAME_LLM}|{FORMAL_STARTUP_POLICY}"
    baseline_entry = aggregates.get(same_llm) or {}
    candidates: list[dict[str, Any]] = []
    per_target: dict[str, Any] = {}
    for key in formal_keys:
        placement = aggregates[key]["placement"]
        if placement == REFERENCE_SAME_LLM:
            continue
        pair = paired.get(f"{REFERENCE_SAME_LLM}->{placement}") or {}
        # 每个指标各用自己的基线值算容差，不能三个指标共用一个分母
        gains = {
            name: _is_improvement(
                (pair.get(name) or {}).get("mean_delta"),
                (baseline_entry.get(name) or {}).get("mean"),
            )
            for name in (
                "first_sentence_ready_s",
                "rtf_mean",
                "supply_fault_episodes",
            )
        }
        context = {
            name: pair.get(name)
            for name in (
                "sentence_count",
                "starve_ratio_in_window",
                "output_tokens",
                "audio_duration_s",
            )
        }
        flags: list[str] = []
        if pair.get("text_identical_all") is False:
            flags.append("output_text_differs")
        if (pair.get("sentence_count") or {}).get("mean_delta") not in (None, 0):
            flags.append("sentence_count_differs")
        if (pair.get("output_tokens") or {}).get("mean_delta") not in (None, 0):
            flags.append("output_tokens_differ")
        per_target[placement] = {
            "paired_run_count": pair.get("paired_run_count", 0),
            "both_passed_run_count": pair.get("both_passed_run_count", 0),
            "gains": gains,
            "deltas": {name: pair.get(name) for name in LOWER_IS_BETTER},
            "context_deltas": context,
            "confound_flags": flags,
            "text_identical_all": pair.get("text_identical_all"),
        }
        if any(gains.values()):
            candidates.append(
                {
                    "placement": placement,
                    "gains": {name: value for name, value in gains.items() if value},
                    "confound_flags": flags,
                    "context_deltas": context,
                    "text_identical_all": pair.get("text_identical_all"),
                    "paired_run_count": pair.get("paired_run_count", 0),
                    "both_passed_run_count": pair.get("both_passed_run_count", 0),
                }
            )
    return {
        "q1_hybrid_vs_feasible_baseline": {
            "non_mixed_baseline": BASELINE_NON_MIXED,
            "reference_same_llm": REFERENCE_SAME_LLM,
            "per_target": per_target,
        },
        "q2_confound_controls": confounds,
        "q3_candidates_for_dynamic_switching": {
            "candidates": candidates,
            "note": (
                "候选必须同时展示调度主指标改善与协变量可控；"
                "设备体验指标仍 pending，不能用来支撑进入动态切换"
            ),
        },
    }
