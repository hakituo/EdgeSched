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

#: 归一指标 -> 对应的原始指标；越小越好，用来检验「改善只是输出更短」这一替代解释。
#: 注意不含 audio_s_per_token：每 token 音频量是内容密度描述符，不是质量指标。
NORMALIZED_PAIRS: tuple[tuple[str, str], ...] = (
    ("last_ready_s_per_token", "last_sentence_ready_s"),
    ("last_ready_s_per_sentence", "last_sentence_ready_s"),
    ("supply_faults_per_10s_audio", "supply_fault_episodes"),
    ("starve_ratio_in_window", "starve_ratio_in_window"),
)

#: 参考为 0、基线为正时比值无定义，但这恰恰是「基线更差」的最强证据。
_GAP_UNDEFINED_NOTE = "参考为 0、基线为正：比值无定义，按『基线更差』记"


def _mean_of(agg: dict[str, Any], metric: str) -> float:
    return float(agg[metric]["mean"])


def sentence_aligned_comparison(
    runs: list[dict[str, Any]],
    *,
    reference: str = REFERENCE_SAME_LLM,
    baseline: str = BASELINE_NON_MIXED,
    policy: str = FORMAL_STARTUP_POLICY,
) -> dict[str, Any]:
    """句数对齐：只比两配置都产出的前 N 句。

    「把两个配置的话调成一样多再比一次」在当前数据上唯一可实现的形式——
    LLM 输出随放置确定性地不同（80 vs 65 token），多跑几轮不会让它们一致，
    固定文本重放又只能作隔离诊断。于是退一步：对齐到相同**句数**，比较
    第 k 句（k < 两配置句数的最小值）的波形就绪时刻，消除总句数差异。
    """

    def series(placement: str) -> dict[int, list[float]]:
        return {
            int(item["run_index"]): list(item.get("sentence_ready_s", []) or [])
            for item in runs
            if item["placement"] == placement and item["startup_policy"] == policy
        }

    ref, base = series(reference), series(baseline)
    shared = sorted(set(ref) & set(base))
    empty: dict[str, Any] = {
        "reference": reference,
        "baseline": baseline,
        "aligned_depth": 0,
        "paired_run_count": len(shared),
        "per_sentence": [],
        "verdict": "两个配置没有可配对的运行，无法做句数对齐",
    }
    if not shared:
        return empty
    depth = min(
        min(len(ref[index]) for index in shared),
        min(len(base[index]) for index in shared),
    )
    if depth <= 0:
        return empty
    per_sentence: list[dict[str, Any]] = []
    for index in range(depth):
        deltas = [
            base[run][index] - ref[run][index]
            for run in shared
            if len(base[run]) > index and len(ref[run]) > index
        ]
        ref_values = [ref[run][index] for run in shared if len(ref[run]) > index]
        base_values = [base[run][index] for run in shared if len(base[run]) > index]
        per_sentence.append(
            {
                "sentence_index": index,
                "reference_mean_s": round(sum(ref_values) / len(ref_values), 6)
                if ref_values
                else None,
                "baseline_mean_s": round(sum(base_values) / len(base_values), 6)
                if base_values
                else None,
                "mean_delta_s": round(sum(deltas) / len(deltas), 6) if deltas else None,
                "baseline_worse_runs": sum(1 for item in deltas if item > 0),
                "paired_run_count": len(deltas),
            }
        )
    consistent = all(
        item["baseline_worse_runs"] == item["paired_run_count"]
        for item in per_sentence
        if item["paired_run_count"]
    )
    if consistent:
        verdict = (
            f"对齐到前 {depth} 句后，基线在每一句的波形就绪上都更差"
            f"（{per_sentence[0]['paired_run_count'] if per_sentence else 0}/"
            f"{per_sentence[0]['paired_run_count'] if per_sentence else 0} 轮一致）："
            "「总句数不同」不足以解释差异。"
            "但各句文本内容仍不同，按冻结规则仍不构成干净归因。"
        )
    else:
        verdict = (
            f"对齐到前 {depth} 句后并非每句都是基线更差"
            "，句数对齐不能支持改善结论。"
        )
    return {
        "reference": reference,
        "baseline": baseline,
        "aligned_depth": depth,
        "paired_run_count": len(shared),
        "per_sentence": per_sentence,
        "all_sentences_baseline_worse": consistent,
        "verdict": verdict,
    }


def covariate_normalization(
    aggregates: dict[str, Any],
    *,
    reference: str = REFERENCE_SAME_LLM,
    baseline: str = BASELINE_NON_MIXED,
) -> dict[str, Any]:
    """按输出长度归一：检验「改善是否只是文本更短」。

    两配置输出不同（token/句数/文本 SHA 都不同）时，主指标的差距有两种解释：
    放置真的更快，或者只是说得少。把末句就绪按 token 与句数归一、供给缺口按
    每 10 秒音频归一后，如果基线相对参考的差距**没有缩小**，则「输出更短」
    不足以解释差距。首句就绪不归一：首句之前两配置输入完全相同，差异只可能来自
    生成与合成速度。
    """

    per_placement: dict[str, dict[str, float]] = {}
    for key, agg in aggregates.items():
        tokens = max(_mean_of(agg, "output_tokens"), 1e-9)
        sentences = max(_mean_of(agg, "sentence_count"), 1e-9)
        audio = _mean_of(agg, "audio_duration_s")
        per_placement[key] = {
            "last_ready_s_per_token": round(_mean_of(agg, "last_sentence_ready_s") / tokens, 6),
            "last_ready_s_per_sentence": round(
                _mean_of(agg, "last_sentence_ready_s") / sentences, 6
            ),
            "audio_s_per_token": round(audio / tokens, 6),
            "supply_faults_per_10s_audio": (
                round(_mean_of(agg, "supply_fault_episodes") / audio * 10.0, 4)
                if audio
                else 0.0
            ),
            "starve_ratio_in_window": _mean_of(agg, "starve_ratio_in_window"),
            "first_sentence_ready_s": _mean_of(agg, "first_sentence_ready_s"),
        }
    ref_key = next((key for key in aggregates if key.startswith(f"{reference}|")), None)
    base_key = next((key for key in aggregates if key.startswith(f"{baseline}|")), None)
    comparisons: dict[str, Any] = {}
    if ref_key and base_key:
        for norm_metric, raw_metric in NORMALIZED_PAIRS:
            ref_raw = _mean_of(aggregates[ref_key], raw_metric)
            base_raw = _mean_of(aggregates[base_key], raw_metric)
            ref_norm = per_placement[ref_key][norm_metric]
            base_norm = per_placement[base_key][norm_metric]
            raw_gap = (base_raw / ref_raw) if ref_raw else None
            norm_gap = (base_norm / ref_norm) if ref_norm else None
            # 参考为 0 时比值无定义：基线为正即更差，两者都为 0 即持平
            if ref_norm == 0.0 and base_norm > 0.0:
                status = "baseline_worse"
                note = _GAP_UNDEFINED_NOTE
            elif ref_norm == 0.0 and base_norm == 0.0:
                status = "tie"
                note = "参考与基线均为 0，无差异可判"
            elif norm_gap is not None and norm_gap > 1.0:
                status = "baseline_worse"
                note = ""
            elif norm_gap is not None and norm_gap == 1.0:
                status = "tie"
                note = ""
            else:
                status = "baseline_better"
                note = "归一化后基线反而更好，方向反转"
            comparisons[norm_metric] = {
                "raw_metric": raw_metric,
                "reference": ref_raw,
                "baseline": base_raw,
                "raw_gap_baseline_over_reference": (
                    round(raw_gap, 4) if raw_gap is not None else None
                ),
                "normalized_reference": ref_norm,
                "normalized_baseline": base_norm,
                "normalized_gap": round(norm_gap, 4) if norm_gap is not None else None,
                "status": status,
                # 越小越好：基线更差即方向保持
                "direction_preserved": status == "baseline_worse",
                "gap_shrinks_after_normalization": bool(
                    norm_gap is not None
                    and raw_gap is not None
                    and norm_gap < raw_gap * 0.9
                ),
                "note": note,
            }
    preserved = [
        name for name, item in comparisons.items() if item["direction_preserved"]
    ]
    shrunk = [
        name
        for name, item in comparisons.items()
        if item["gap_shrinks_after_normalization"]
    ]
    reversed_names = [
        name for name, item in comparisons.items() if item["status"] == "baseline_better"
    ]
    if comparisons and reversed_names:
        verdict = f"归一化后方向反转：{reversed_names}；不能支持改善结论。"
    elif shrunk:
        verdict = (
            f"归一化后差距缩小：{shrunk}；"
            "「输出更短」可能部分解释差异，必须补输出一致的重跑才能归因。"
        )
    elif comparisons and len(preserved) == len(comparisons):
        verdict = (
            "归一化后基线仍全部更差、且差距未缩小："
            "「输出更短」不足以解释主指标差异。"
            "但文本内容本身仍不同，按冻结规则不构成干净归因。"
        )
    else:
        verdict = "归一化后无差异或样本不足，不能支持改善结论。"
    return {
        "reference": reference,
        "baseline": baseline,
        "per_placement": per_placement,
        "comparisons": comparisons,
        "direction_preserved_metrics": preserved,
        "gap_shrinking_metrics": shrunk,
        "verdict": verdict,
        "note": (
            "首句就绪不参与归一：首句之前两配置输入相同，差异只可能来自生成与合成速度"
        ),
    }


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
