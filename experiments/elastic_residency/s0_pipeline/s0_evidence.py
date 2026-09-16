"""S0 产物门禁与证据汇总：技术门禁、输出指纹与两层测量范围。

从 ``s0_parent`` 拆出以保持单文件规模上限；本模块只做纯判定与汇总，
不加载模型、不启动子进程。
"""

from __future__ import annotations

import hashlib
from typing import Any

from experiments.elastic_residency.s0_pipeline.measurement_scope import (
    evaluate_measurement_scope,
)


def gate_checks(record: dict[str, Any]) -> dict[str, bool]:
    """技术门禁；设备层 pending 不作为失败项，另由测量范围报告承载。"""

    run = record.get("run", {})
    semantics = run.get("semantics", {})
    swap = record.get("swap_report", {})
    return {
        "child_passed": record.get("passed") is True,
        "has_real_llm_identity": bool(record.get("llm_identity")),
        "has_events": run.get("event_count", 0) > 0,
        "has_execution_intervals": bool(run.get("tts_execution_intervals")),
        "swap_within_tolerance": bool(swap.get("within_tolerance")),
        "no_proxy_only_evidence": semantics.get("execution_event_count", 0) > 0,
        "output_contract_ok": run.get("text_contract", {}).get("failed") is not True,
        "prompt_contract_recorded": bool(
            record.get("prompt_contract", {}).get("system_prompt_sha256")
        ),
    }


def output_fingerprint(record: dict[str, Any]) -> dict[str, Any]:
    """输出差异协变量：文本 SHA-256、字符数、句数、输出 token 数与音频总长。"""

    run = record.get("run", {})
    sentences = list(run.get("sentences", []) or [])
    raw = "".join(str(item.get("raw_text", "")) for item in sentences)
    tokens = run.get("token_stats", {}) or {}
    return {
        "raw_text_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "raw_text_chars": len(raw),
        "sentence_count": len(sentences),
        "output_tokens": tokens.get("actual_output_tokens"),
        "audio_duration_s": round(
            sum(float(item.get("audio_duration_s", 0.0)) for item in sentences), 6
        ),
    }


def attach_evidence(record: dict[str, Any]) -> dict[str, Any]:
    """写入输出指纹、两层测量范围，并把调度层可判据纳入 ``passed``。"""

    record["output_fingerprint"] = output_fingerprint(record)
    record["measurement_scope"] = evaluate_measurement_scope(record)
    checks = gate_checks(record)
    checks["scheduling_scope_complete"] = bool(
        record["measurement_scope"]["gate"]["scheduling_scope_complete"]
    )
    record["acceptance_checks"] = checks
    record["passed"] = all(checks.values())
    return record


def summarize_record(record: dict[str, Any]) -> dict[str, Any]:
    """写进 ``summary.json`` 的轻量行，不放事件全量。"""

    scope = record.get("measurement_scope", {})
    scheduling = scope.get("scheduling", {})
    supply = scheduling.get("application_supply_gap", {})
    device = scope.get("device_experience", {})
    return {
        "placement": record.get("placement"),
        "startup_policy": record.get("startup_policy"),
        "run_index": record.get("run_index"),
        "order_index": record.get("order_index"),
        "passed": record.get("passed"),
        "checks": record.get("acceptance_checks"),
        "error": record.get("error", ""),
        "sentence_count": record.get("run", {}).get("sentence_count", 0),
        "output_fingerprint": record.get("output_fingerprint"),
        "first_sentence_waveform_ready_s": scheduling.get(
            "first_sentence_waveform_ready_s"
        ),
        "last_sentence_waveform_ready_s": scheduling.get(
            "last_sentence_waveform_ready_s"
        ),
        "supply_fault_episode_count": supply.get("supply_fault_episode_count"),
        "starve_ratio_in_window": supply.get("starve_ratio_in_window"),
        "device_metrics_status": device.get("status"),
    }
