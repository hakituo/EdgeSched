"""端点 smoke 的聚合与 CSV 汇总（纯计算，可离线单测）。

聚合只按 `(驻留档位, 驻留状态)` 单元分组。禁止把 `TEXT_ONLY` 与
`VISION_WARM_*` 混入同一组统计：两者 prompt 语义不同，TTFT 与 prefill
不可直接比较。
"""

from __future__ import annotations

import csv
import statistics
from pathlib import Path
from typing import Any

from experiments.tts.profiling_utils import percentile


CSV_FIELDS = (
    "config",
    "residency_state",
    "run_index",
    "passed",
    "n_gpu_layers",
    "llm_load_s",
    "ttft_s",
    "prefill_tokens_per_s",
    "decode_tokens_per_s",
    "prompt_token_count",
    "visual_token_total",
    "projector_load_s",
    "projector_free_s",
    "projector_gpu_card_delta_mb",
    "projector_rss_delta_bytes",
    "gpu_after_llm_mb",
    "gpu_after_inference_mb",
    "rss_after_inference_mb",
    "swap_growth_bytes",
    "output_text_sha256",
)


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """对同一单元的多轮记录做统计摘要。"""

    metrics = {
        "llm_load_s": [record["llm_load_s"] for record in records],
        "ttft_s": [record["inference"]["ttft_s"] for record in records],
        "prefill_tokens_per_s": [
            record["inference"]["prefill_tokens_per_s"] for record in records
        ],
        "decode_tokens_per_s": [
            record["inference"]["decode_tokens_per_s"] for record in records
        ],
        "gpu_delta_after_llm_mb": [
            (record["memory_after_llm_load"]["gpu_card_used_mb"] or 0.0)
            - (record["memory_baseline"]["gpu_card_used_mb"] or 0.0)
            for record in records
        ],
        "rss_after_inference_mb": [
            record["memory_after_inference_rss_bytes"] / 1048576.0
            for record in records
        ],
    }
    return {
        "samples": len(records),
        "passed_count": sum(bool(record["passed"]) for record in records),
        "all_passed": all(record["passed"] for record in records),
        **{
            name: {
                "mean": statistics.fmean(values),
                "p95": percentile(values, 0.95),
                "min": min(values),
                "max": max(values),
            }
            for name, values in metrics.items()
        },
    }


def summarize_by_cell(records: list[dict[str, Any]]) -> dict[str, Any]:
    """按 ``配置|状态`` 分组聚合，保持文本与视觉口径分离。"""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        key = f"{record['config']}|{record['residency_state']}"
        grouped.setdefault(key, []).append(record)
    return {key: summarize_records(items) for key, items in grouped.items()}


def _csv_row(record: dict[str, Any]) -> dict[str, Any]:
    vision = record.get("vision") or {}
    session = vision.get("session") or {}
    warm = vision.get("projector_warm_delta") or {}
    execution = vision.get("chunk_execution") or {}
    return {
        "config": record["config"],
        "residency_state": record["residency_state"],
        "run_index": record["run_index"],
        "passed": record["passed"],
        "n_gpu_layers": record["n_gpu_layers"],
        "llm_load_s": record["llm_load_s"],
        "ttft_s": record["inference"]["ttft_s"],
        "prefill_tokens_per_s": record["inference"]["prefill_tokens_per_s"],
        "decode_tokens_per_s": record["inference"]["decode_tokens_per_s"],
        "prompt_token_count": record["prompt"]["prompt_token_count"],
        "visual_token_total": execution.get("visual_token_total"),
        "projector_load_s": session.get("projector_load_s"),
        "projector_free_s": (vision.get("release") or {}).get("projector_free_s"),
        "projector_gpu_card_delta_mb": warm.get("gpu_card_delta_mb"),
        "projector_rss_delta_bytes": warm.get("rss_delta_bytes"),
        "gpu_after_llm_mb": record["memory_after_llm_load"]["gpu_card_used_mb"],
        "gpu_after_inference_mb": record["memory_after_inference"]["gpu_card_used_mb"],
        "rss_after_inference_mb": (
            record["memory_after_inference_rss_bytes"] / 1048576.0
        ),
        "swap_growth_bytes": record["runtime_evidence"]["swap"].get("growth_bytes"),
        "output_text_sha256": record["output"]["output_text_sha256"],
    }


def write_summary_csv(path: str | Path, records: list[dict[str, Any]]) -> None:
    """写单元级机器可读汇总。"""

    with Path(path).open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        for record in records:
            writer.writerow(_csv_row(record))
