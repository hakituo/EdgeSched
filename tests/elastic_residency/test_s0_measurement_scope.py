"""测量范围分层测试：调度主指标可信、设备体验指标未验证时保持 pending。"""

from experiments.elastic_residency.s0_pipeline.measurement_scope import (
    LAYER_DEVICE,
    LAYER_SCHEDULING,
    STATUS_PENDING,
    STATUS_TRUSTED,
    evaluate_measurement_scope,
)

_T0 = 1_000_000_000


def _record(*, verified: bool) -> dict:
    events = [
        {"stage": "request_start", "timestamp_ns": _T0},
        {"stage": "waveform_ready", "timestamp_ns": _T0 + 2_000_000_000, "chunk_id": 0},
        {"stage": "waveform_ready", "timestamp_ns": _T0 + 4_000_000_000, "chunk_id": 1},
        {"stage": "playback_start", "timestamp_ns": _T0 + 3_000_000_000, "chunk_id": 0},
    ]
    calibration = (
        {"verified": True, "method": "paired", "reasons": []}
        if verified
        else {"verified": False, "method": "paired", "reasons": ["跨度比 1.12"]}
    )
    return {
        "passed": True,
        "llm_identity": {"model_path": "m"},
        "swap_report": {
            "within_tolerance": True,
            "growth_bytes": 0,
            "tolerance_bytes": 4 * 1024 * 1024,
        },
        "output_fingerprint": {"raw_text_sha256": "a" * 64},
        "run": {
            "finished": True,
            "error": "",
            "event_count": len(events),
            "events": events,
            "sentences": [
                {
                    "chunk_id": 0,
                    "queue_wait_s": 0.1,
                    "synthesis_s": 0.5,
                    "rtf": 1.2,
                    "audio_duration_s": 0.4,
                    "raw_text": "甲。",
                    "audio_sha256": "0" * 64,
                },
                {
                    "chunk_id": 1,
                    "queue_wait_s": 0.2,
                    "synthesis_s": 0.6,
                    "rtf": 1.5,
                    "audio_duration_s": 0.5,
                    "raw_text": "乙。",
                    "audio_sha256": "1" * 64,
                },
            ],
            "text_contract": {"failed": False, "failures": []},
            "playback_calibration": calibration,
            "playback_stats": {"device_underflow_count": 0, "backend": "sounddevice"},
            "playback_starvation": {
                "starve_episode_count": 2,
                "starve_callback_count": 5,
                "supply_fault_episode_count": 1,
                "supply_fault_samples": 4800,
                "starve_samples_in_window": 4800,
                "window_samples": 24000,
                "starve_ratio_in_window": 0.2,
                "by_phase": {"inter_sentence": {"episodes": 1}},
            },
        },
    }


def test_scheduling_layer_is_trusted_and_uses_host_clock() -> None:
    scope = evaluate_measurement_scope(_record(verified=False))
    scheduling = scope[LAYER_SCHEDULING]
    assert scheduling["status"] == STATUS_TRUSTED
    assert scheduling["first_sentence_waveform_ready_s"] == 2.0
    assert scheduling["last_sentence_waveform_ready_s"] == 4.0
    assert scheduling["sentence"]["rtf"]["mean"] == 1.35
    assert scheduling["application_supply_gap"]["supply_fault_episode_count"] == 1


def test_unverified_device_clock_keeps_device_metrics_pending() -> None:
    scope = evaluate_measurement_scope(_record(verified=False))
    device = scope[LAYER_DEVICE]
    assert device["status"] == STATUS_PENDING
    # 代理值只作为 proxy 保留，绝不填进实际播放时刻
    assert device["actual_first_playback_s"] is None
    assert device["proxy_first_playback_s"] == 3.0
    assert device["device_underflow_status"] == STATUS_PENDING
    assert "跨度比" in "；".join(device["device_clock_reasons"])


def test_verified_device_clock_promotes_device_metrics() -> None:
    scope = evaluate_measurement_scope(_record(verified=True))
    device = scope[LAYER_DEVICE]
    assert device["status"] == STATUS_TRUSTED
    assert device["actual_first_playback_s"] == 3.0
    assert device["proxy_first_playback_s"] is None


def test_gate_uses_scheduling_scope_and_device_pending_does_not_block() -> None:
    scope = evaluate_measurement_scope(_record(verified=False))
    assert scope["gate"]["scheduling_scope_complete"] is True
    assert scope["gate"]["device_metrics_status"] == STATUS_PENDING
    assert scope["gate"]["device_metrics_block_go"] is False


def test_missing_waveform_event_fails_scheduling_scope() -> None:
    record = _record(verified=True)
    record["run"]["events"] = [
        item for item in record["run"]["events"] if item["stage"] != "waveform_ready"
    ]
    scope = evaluate_measurement_scope(record)
    assert scope["gate"]["scheduling_scope_complete"] is False
    assert scope["gate"]["checks"]["first_sentence_ready_recorded"] is False


def test_output_contract_failure_fails_scheduling_scope() -> None:
    record = _record(verified=True)
    record["run"]["text_contract"] = {"failed": True, "failures": ["thinking_body"]}
    scope = evaluate_measurement_scope(record)
    assert scope["gate"]["scheduling_scope_complete"] is False
    assert scope["gate"]["checks"]["output_contract_ok"] is False
