"""S0 成对比较的指标抽取、聚合与配对差（纯标准库）。

从 ``s0_matrix_report`` 拆出以保持单文件规模上限；这里只做逐条记录的
数值化、聚合与成对求差，不做结论判定，也不加载模型。

判定只用调度主指标；设备体验指标一律按 pending 记录，不参与比较。
"""

from __future__ import annotations

from statistics import mean, median
from typing import Any, Iterable

from experiments.elastic_residency.s0_pipeline.config import FROZEN_PROTOCOL
from experiments.elastic_residency.s0_pipeline.measurement_scope import (
    evaluate_measurement_scope,
    sentence_ready_seconds,
)

FORMAL_STARTUP_POLICY = FROZEN_PROTOCOL["startup_policy"]["formal"]

_NUMERIC_KEYS = (
    "first_sentence_ready_s",
    "last_sentence_ready_s",
    "queue_wait_mean_s",
    "queue_wait_max_s",
    "synthesis_mean_s",
    "synthesis_max_s",
    "rtf_mean",
    "rtf_max",
    "sentence_count",
    "audio_duration_s",
    "supply_fault_episodes",
    "supply_fault_samples",
    "starve_samples_in_window",
    "starve_ratio_in_window",
    "output_tokens",
    "raw_text_chars",
    "swap_growth_bytes",
    "device_underflow_count",
    "proxy_first_playback_s",
    "prebuffered_sentences",
    "prebuffered_audio_s",
)


def extract_metrics(record: dict[str, Any]) -> dict[str, Any]:
    """从单条迭代记录抽取指标；优先用已落盘的测量范围，缺则现场复算。"""

    scope = record.get("measurement_scope") or evaluate_measurement_scope(record)
    scheduling = scope["scheduling"]
    supply = scheduling["application_supply_gap"]
    sentence = scheduling["sentence"]
    resources = scheduling["resources_and_correctness"]
    device = scope["device_experience"]
    fingerprint = record.get("output_fingerprint", {}) or {}
    startup = record.get("run", {}).get("startup", {}) or {}
    return {
        "placement": record.get("placement"),
        "startup_policy": record.get("startup_policy", FORMAL_STARTUP_POLICY),
        "run_index": record.get("run_index"),
        "order_index": record.get("order_index"),
        "passed": bool(record.get("passed")),
        "first_sentence_ready_s": scheduling["first_sentence_waveform_ready_s"],
        "last_sentence_ready_s": scheduling["last_sentence_waveform_ready_s"],
        "queue_wait_mean_s": sentence["queue_wait_s"]["mean"],
        "queue_wait_max_s": sentence["queue_wait_s"]["max"],
        "synthesis_mean_s": sentence["synthesis_s"]["mean"],
        "synthesis_max_s": sentence["synthesis_s"]["max"],
        "rtf_mean": sentence["rtf"]["mean"],
        "rtf_max": sentence["rtf"]["max"],
        "sentence_count": sentence["sentence_count"],
        "audio_duration_s": sentence["audio_duration_s"],
        "supply_fault_episodes": supply["supply_fault_episode_count"],
        "supply_fault_samples": supply["supply_fault_samples"],
        "starve_samples_in_window": supply["starve_samples_in_window"],
        "starve_ratio_in_window": supply["starve_ratio_in_window"],
        "output_tokens": fingerprint.get("output_tokens"),
        "raw_text_chars": fingerprint.get("raw_text_chars"),
        "raw_text_sha256": fingerprint.get("raw_text_sha256"),
        "output_contract_failed": resources["output_contract_failed"],
        "swap_growth_bytes": resources["swap_growth_bytes"],
        "device_status": device["status"],
        "device_underflow_count": device["device_underflow_count"],
        "proxy_first_playback_s": device["proxy_first_playback_s"],
        "actual_first_playback_s": device["actual_first_playback_s"],
        "prebuffered_sentences": startup.get("prebuffered_sentences", 0),
        "prebuffered_audio_s": startup.get("prebuffered_audio_s", 0.0),
        "sentence_ready_s": _sentence_ready_series(record, scheduling),
    }


def _sentence_ready_series(
    record: dict[str, Any], scheduling: dict[str, Any]
) -> list[float]:
    """逐句就绪序列；旧产物没存该字段时从事件现场复算。"""

    series = list(scheduling.get("sentence_ready_s", []) or [])
    if series:
        return series
    return sentence_ready_seconds(record)


def key_of(metrics: dict[str, Any]) -> str:
    return f"{metrics['placement']}|{metrics['startup_policy']}"


def aggregate(runs: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """按 (placement, startup_policy) 聚合；SHA 单独汇总为去重集合。"""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in runs:
        grouped.setdefault(key_of(item), []).append(item)
    result: dict[str, dict[str, Any]] = {}
    for key, items in grouped.items():
        entry: dict[str, Any] = {
            "placement": items[0]["placement"],
            "startup_policy": items[0]["startup_policy"],
            "run_count": len(items),
            "passed_count": sum(1 for item in items if item["passed"]),
        }
        for name in _NUMERIC_KEYS:
            values = [
                float(item[name]) for item in items if item.get(name) is not None
            ]
            entry[name] = (
                {
                    "count": len(values),
                    "mean": round(mean(values), 6),
                    "median": round(median(values), 6),
                    "min": round(min(values), 6),
                    "max": round(max(values), 6),
                }
                if values
                else None
            )
        entry["distinct_text_sha256"] = sorted(
            {
                str(item["raw_text_sha256"])
                for item in items
                if item.get("raw_text_sha256")
            }
        )
        entry["device_status_counts"] = _counts(
            item["device_status"] for item in items
        )
        # 失败运行计入分母并列出下标，不静默删除
        entry["failed_run_indices"] = sorted(
            int(item["run_index"]) for item in items if not item["passed"]
        )
        result[key] = entry
    return result


def paired_deltas_by_key(
    runs: list[dict[str, Any]],
    *,
    baseline_key: str,
    target_key: str,
) -> dict[str, Any]:
    """按 ``run_index`` 配对求差；键为 ``placement|policy``。"""

    def index(key: str) -> dict[int, dict[str, Any]]:
        return {int(item["run_index"]): item for item in runs if key_of(item) == key}

    left, right = index(baseline_key), index(target_key)
    shared = sorted(set(left) & set(right))
    per_run: list[dict[str, Any]] = []
    for run_index in shared:
        row: dict[str, Any] = {"run_index": run_index}
        for name in _NUMERIC_KEYS:
            a, b = left[run_index].get(name), right[run_index].get(name)
            row[name] = (
                None if a is None or b is None else round(float(b) - float(a), 6)
            )
        row["baseline_text_sha256"] = left[run_index].get("raw_text_sha256")
        row["target_text_sha256"] = right[run_index].get("raw_text_sha256")
        row["text_identical"] = (
            row["baseline_text_sha256"] == row["target_text_sha256"]
        )
        # 失败轮次仍进配对与均值（计入分母），同时显式标记以便复核
        row["baseline_passed"] = bool(left[run_index].get("passed"))
        row["target_passed"] = bool(right[run_index].get("passed"))
        per_run.append(row)
    both_passed = [
        row for row in per_run if row["baseline_passed"] and row["target_passed"]
    ]
    summary: dict[str, Any] = {
        "baseline_key": baseline_key,
        "target_key": target_key,
        "paired_run_count": len(shared),
        "both_passed_run_count": len(both_passed),
        "per_run": per_run,
    }
    for name in _NUMERIC_KEYS:
        values = [float(row[name]) for row in per_run if row[name] is not None]
        clean = [float(row[name]) for row in both_passed if row[name] is not None]
        summary[name] = (
            {
                "count": len(values),
                "mean_delta": round(mean(values), 6),
                "median_delta": round(median(values), 6),
                "improved_runs": sum(1 for value in values if value < 0),
                "worsened_runs": sum(1 for value in values if value > 0),
                # 次级口径：只用两侧都通过的配对，透明展示失败轮次的影响
                "both_passed_count": len(clean),
                "both_passed_mean_delta": (
                    round(mean(clean), 6) if clean else None
                ),
            }
            if values
            else None
        )
    summary["text_identical_all"] = (
        all(row["text_identical"] for row in per_run) if per_run else None
    )
    return summary


def confound_checks(
    runs: list[dict[str, Any]], *, policy: str = FORMAL_STARTUP_POLICY
) -> dict[str, Any]:
    """检查收益是否只是「文本更短、分句不同或预缓冲更多」。"""

    formal = [item for item in runs if item["startup_policy"] == policy]
    by_placement: dict[str, dict[str, Any]] = {}
    for item in formal:
        entry = by_placement.setdefault(
            item["placement"],
            {"tokens": set(), "sentences": set(), "sha": set(), "audio": set()},
        )
        if item.get("output_tokens") is not None:
            entry["tokens"].add(int(item["output_tokens"]))
        entry["sentences"].add(int(item["sentence_count"]))
        if item.get("raw_text_sha256"):
            entry["sha"].add(str(item["raw_text_sha256"]))
        entry["audio"].add(round(float(item["audio_duration_s"]), 3))
    distinct_policies = sorted({item["startup_policy"] for item in runs})
    return {
        "policy_under_test": policy,
        "startup_policies_present": distinct_policies,
        "prebuffer_confound_present": len(distinct_policies) > 1,
        "per_placement": {
            placement: {
                "distinct_output_tokens": sorted(values["tokens"]),
                "distinct_sentence_counts": sorted(values["sentences"]),
                "distinct_text_sha256": sorted(values["sha"]),
                "distinct_audio_duration_s": sorted(values["audio"]),
            }
            for placement, values in by_placement.items()
        },
    }


def _counts(values: Iterable[Any]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        key = str(value)
        result[key] = result.get(key, 0) + 1
    return result
