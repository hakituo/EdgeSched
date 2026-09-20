"""S0 正式比较的测量范围分层（2026-09-17）。

把一次运行的证据拆成两层，禁止互相冒充：

- **调度主指标**（``scheduling``）：在当前环境下就可信，构成 Gate B 的形式判据。
  来源是 host monotonic 与真实执行区间——首句/末句波形就绪、句级排队与合成
  时间、应用层供给缺口、资源门禁与输出契约。
- **设备体验指标**（``device_experience``）：必须先有**通过验证的设备时钟映射**
  才能给出可信数值。时钟未通过时一律标 ``pending_device_validation``，只保留原始
  观测并注明是 callback 代理，禁止把代理当成实际播放时刻。

分层不是降低要求，而是把「还没有可信证据」与「已经失败」分开：设备层 pending
只说明测量能力不足，不代表连续供给失败；反之调度层失败也不能被设备层掩盖。
"""

from __future__ import annotations

from typing import Any

SCOPE_VERSION = "edgesched.s0_measurement_scope.v1"

LAYER_SCHEDULING = "scheduling"
LAYER_DEVICE = "device_experience"

STATUS_TRUSTED = "trusted"
STATUS_PENDING = "pending_device_validation"
#: 观测到但不附时刻：例如 PortAudio 直接报的 underflow 状态位，计数可用、
#: 时刻仍待 DAC 时钟验证。
STATUS_OBSERVED = "observed_not_clock_anchored"

SCHEDULING_METRICS: tuple[str, ...] = (
    "first_sentence_waveform_ready_s",
    "last_sentence_waveform_ready_s",
    "sentence_queue_and_synthesis",
    "application_supply_gap",
    "resources_and_correctness",
)

DEVICE_METRICS: tuple[str, ...] = (
    "actual_first_playback_s",
    "device_underflow",
)

_STAGE_REQUEST_START = "request_start"
_STAGE_WAVEFORM_READY = "waveform_ready"
_STAGE_PLAYBACK_START = "playback_start"


def _events(record: dict[str, Any]) -> list[dict[str, Any]]:
    return list(record.get("run", {}).get("events", []) or [])


def _request_start_ns(events: list[dict[str, Any]]) -> int | None:
    for event in events:
        if event.get("stage") == _STAGE_REQUEST_START:
            return int(event["timestamp_ns"])
    return None


def _stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "max": None, "values": []}
    return {
        "count": len(values),
        "mean": round(sum(values) / len(values), 6),
        "max": round(max(values), 6),
        "values": [round(item, 6) for item in values],
    }


def waveform_ready_seconds(record: dict[str, Any]) -> tuple[float | None, float | None]:
    """首句/末句完整波形就绪延迟（相对 ``REQUEST_START``，单位 s）。"""

    events = _events(record)
    start = _request_start_ns(events)
    ready = [
        int(event["timestamp_ns"])
        for event in events
        if event.get("stage") == _STAGE_WAVEFORM_READY
    ]
    if start is None or not ready:
        return None, None
    return (min(ready) - start) / 1e9, (max(ready) - start) / 1e9


def sentence_ready_seconds(record: dict[str, Any]) -> list[float]:
    """逐句完整波形就绪延迟（相对 ``REQUEST_START``，按 chunk_id 排序，单位 s）。

    用于句数对齐：两配置句数不同时，只比较都产出的前 N 句，消除「总句数不同」
    这一协变量的影响。
    """

    events = _events(record)
    start = _request_start_ns(events)
    ready = [
        (int(event.get("chunk_id", 0)), int(event["timestamp_ns"]))
        for event in events
        if event.get("stage") == _STAGE_WAVEFORM_READY
    ]
    if start is None or not ready:
        return []
    ready.sort(key=lambda item: item[0])
    return [round((stamp - start) / 1e9, 6) for _, stamp in ready]


def first_playback_seconds(record: dict[str, Any]) -> float | None:
    """首个 ``PLAYBACK_START`` 距 ``REQUEST_START`` 的秒数；语义由调用方判定。"""

    events = _events(record)
    start = _request_start_ns(events)
    starts = [
        int(event["timestamp_ns"])
        for event in events
        if event.get("stage") == _STAGE_PLAYBACK_START
    ]
    if start is None or not starts:
        return None
    return round((min(starts) - start) / 1e9, 6)


def sentence_metrics(record: dict[str, Any]) -> dict[str, Any]:
    """句级排队与合成时间；只汇总真实执行区间记录的字段。"""

    sentences = list(record.get("run", {}).get("sentences", []) or [])
    queue_wait = [float(item["queue_wait_s"]) for item in sentences if "queue_wait_s" in item]
    synthesis = [float(item["synthesis_s"]) for item in sentences if "synthesis_s" in item]
    rtf = [float(item["rtf"]) for item in sentences if "rtf" in item]
    audio = [float(item["audio_duration_s"]) for item in sentences if "audio_duration_s" in item]
    return {
        "sentence_count": len(sentences),
        "queue_wait_s": _stats(queue_wait),
        "synthesis_s": _stats(synthesis),
        "rtf": _stats(rtf),
        "audio_duration_s": round(sum(audio), 6),
        "audio_sha256": [item.get("audio_sha256") for item in sentences],
    }


def supply_gap_metrics(record: dict[str, Any]) -> dict[str, Any]:
    """应用层供给缺口：与设备 underflow 严格分开的另一口径。"""

    starvation = record.get("run", {}).get("playback_starvation", {}) or {}
    return {
        "starve_episode_count": int(starvation.get("starve_episode_count", 0) or 0),
        "starve_callback_count": int(starvation.get("starve_callback_count", 0) or 0),
        "supply_fault_episode_count": int(
            starvation.get("supply_fault_episode_count", 0) or 0
        ),
        "supply_fault_samples": int(starvation.get("supply_fault_samples", 0) or 0),
        "starve_samples_in_window": int(
            starvation.get("starve_samples_in_window", 0) or 0
        ),
        "window_samples": int(starvation.get("window_samples", 0) or 0),
        "starve_ratio_in_window": starvation.get("starve_ratio_in_window"),
        "by_phase": starvation.get("by_phase", {}),
    }


def scheduling_layer(record: dict[str, Any]) -> dict[str, Any]:
    """调度主指标；当前环境下即可作为 Gate B 判据的部分。"""

    first_ready, last_ready = waveform_ready_seconds(record)
    swap = record.get("swap_report", {}) or {}
    contract = record.get("run", {}).get("text_contract", {}) or {}
    identity = record.get("llm_identity", {}) or {}
    events = _events(record)
    sentences = sentence_metrics(record)
    return {
        "layer": LAYER_SCHEDULING,
        "status": STATUS_TRUSTED,
        "first_sentence_waveform_ready_s": (
            None if first_ready is None else round(first_ready, 6)
        ),
        "last_sentence_waveform_ready_s": (
            None if last_ready is None else round(last_ready, 6)
        ),
        "sentence_ready_s": sentence_ready_seconds(record),
        "sentence": sentences,
        "application_supply_gap": supply_gap_metrics(record),
        "resources_and_correctness": {
            "swap_within_tolerance": bool(swap.get("within_tolerance")),
            "swap_growth_bytes": swap.get("growth_bytes"),
            "swap_tolerance_bytes": swap.get("tolerance_bytes"),
            "output_contract_failed": bool(contract.get("failed")),
            "output_contract_failures": contract.get("failures", []),
            "event_count": len(events),
            "has_real_llm_identity": bool(identity),
            "finished": bool(record.get("run", {}).get("finished")),
            "error": record.get("run", {}).get("error", ""),
        },
    }


def device_layer(record: dict[str, Any]) -> dict[str, Any]:
    """设备体验指标；时钟未验证时保持 pending，只保留代理观测。"""

    calibration = record.get("run", {}).get("playback_calibration") or {}
    if not calibration:
        calibration = {}
    verified = bool(calibration.get("verified"))
    stats = record.get("run", {}).get("playback_stats", {}) or {}
    proxy_first = first_playback_seconds(record)
    reasons = list(calibration.get("reasons", []) or [])
    # underflow 计数来自 PortAudio 的 callback 状态位，不依赖时钟映射；
    # 但「发生在什么时刻」要靠 DAC 时钟，未验证前只报观测、不给时刻。
    underflow = int(stats.get("device_underflow_count", 0) or 0)
    underflow_status = (
        STATUS_TRUSTED if verified else STATUS_OBSERVED if stats.get("backend") else STATUS_PENDING
    )
    return {
        "layer": LAYER_DEVICE,
        "status": STATUS_TRUSTED if verified else STATUS_PENDING,
        "device_clock_verified": verified,
        "device_clock_method": calibration.get("method"),
        "device_clock_residual_max_ns": calibration.get("residual_max_ns"),
        "device_clock_drift_ns_per_s": calibration.get("drift_ns_per_s"),
        "device_clock_reasons": reasons,
        "actual_first_playback_s": (
            proxy_first if verified else None
        ),
        "proxy_first_playback_s": None if verified else proxy_first,
        "device_underflow_count": underflow,
        "device_underflow_source": (
            "portaudio_status_flag" if stats.get("backend") else ""
        ),
        "device_underflow_status": underflow_status,
        "device_underflow_note": (
            "underflow 计数由 PortAudio callback 状态位直接给出，不依赖时钟映射；"
            "但其发生时刻仍需 DAC 时钟，未验证前不得给出时刻"
        ),
        "playback_backend": stats.get("backend", ""),
        "note": (
            "设备时钟验证通过，实际播放时刻可用"
            if verified
            else "设备时钟映射未通过验证：实际播放时刻保持 pending，"
            "proxy_first_playback_s 只是 callback 代理，不得当作实际播放时刻；"
            "device_underflow_count 为观测计数（状态位），同样不附时刻"
        ),
    }


def evaluate_measurement_scope(record: dict[str, Any]) -> dict[str, Any]:
    """生成两层报告与 Gate B 的可判据部分。

    ``device_metrics_block_go`` 恒为 ``False``：设备层 pending 是测量能力不足，
    不能反过来当成连续性失败，也不能被当作已经通过。
    """

    scheduling = scheduling_layer(record)
    device = device_layer(record)
    correctness = scheduling["resources_and_correctness"]
    sentence = scheduling["sentence"]
    checks = {
        "has_events": correctness["event_count"] > 0,
        "output_contract_ok": not correctness["output_contract_failed"],
        "finished_without_error": correctness["finished"]
        and not str(correctness["error"]).strip(),
        "first_sentence_ready_recorded": scheduling[
            "first_sentence_waveform_ready_s"
        ]
        is not None,
        "sentence_metrics_complete": sentence["sentence_count"] > 0
        and sentence["queue_wait_s"]["count"] == sentence["sentence_count"]
        and sentence["rtf"]["count"] == sentence["sentence_count"],
        "swap_within_tolerance": correctness["swap_within_tolerance"],
    }
    return {
        "scope_version": SCOPE_VERSION,
        "layer_definitions": {
            LAYER_SCHEDULING: list(SCHEDULING_METRICS),
            LAYER_DEVICE: list(DEVICE_METRICS),
        },
        LAYER_SCHEDULING: scheduling,
        LAYER_DEVICE: device,
        "gate": {
            "scheduling_scope_complete": all(checks.values()),
            "checks": checks,
            "device_metrics_status": device["status"],
            "device_metrics_block_go": False,
            "statement": (
                "Gate B 的形式判据只使用调度主指标（正确性、资源、首句/末句就绪、"
                "句级排队与合成、应用层供给缺口）；设备体验指标在时钟映射通过前"
                "保持 pending，既不作为 Go 的充分条件，也不作为 No-Go 的失败项。"
            ),
        },
    }
