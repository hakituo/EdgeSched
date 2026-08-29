import os

import pytest
import torch

from experiments.elastic_residency.tts_component_offload import (
    _actual_devices,
    _module_inventory,
    _nvidia_memory_snapshot,
    _placement_matches,
    placement_for,
)


def test_frozen_t0_t3_placement_contract() -> None:
    assert placement_for("T0").speech_codec_device == "cuda:0"
    assert placement_for("t1").speech_codec_device == "cpu"
    assert placement_for("T2").speaker_device == "cpu"
    assert placement_for("T3").unload_speaker_after_prompt is True
    with pytest.raises(ValueError):
        placement_for("T4")


def test_module_inventory_reports_real_parameter_device_and_size() -> None:
    module = torch.nn.Linear(3, 2)
    inventory = _module_inventory(module)
    assert inventory["present"] is True
    assert inventory["devices"] == ["cpu"]
    assert inventory["parameter_count"] == 8
    assert inventory["parameter_bytes"] > 0
    assert _module_inventory(None)["present"] is False


def test_t3_requires_speaker_to_be_unloaded_after_prompt() -> None:
    inventory = {
        "talker": {"present": True, "devices": ["cuda:0"]},
        "speech_codec": {"present": True, "devices": ["cpu"]},
        "speaker_encoder": {"present": False, "devices": []},
        "text_tokenizer": {"present": True, "devices": ["cpu"]},
    }
    actual = _actual_devices(inventory)
    assert actual["speaker_encoder"] == "unloaded"
    assert _placement_matches(placement_for("T3"), actual)


def test_wsl_non_numeric_process_memory_is_not_reported_as_zero() -> None:
    class _FakeProbe:
        def sample(self, _pid: int):
            return {
                "available": True,
                "backend": "direct_nvml",
                "card_used_mb": 2676.0,
                "card_total_mb": 8151.0,
                "process_used_mb": None,
                "process_memory_unavailable_reason": (
                    "driver_reported_non_numeric_process_memory"
                ),
            }

    snapshot = _nvidia_memory_snapshot(os.getpid(), probe=_FakeProbe())

    assert snapshot["gpu_card_used_mb"] == 2676.0
    assert snapshot["gpu_process_total_mb"] is None
    assert snapshot["gpu_process_total_available"] is False
    assert (
        snapshot["gpu_process_total_unavailable_reason"]
        == "driver_reported_non_numeric_process_memory"
    )
    assert snapshot["gpu_telemetry_backend"] == "direct_nvml"
    assert snapshot["gpu_cli_subprocess_used"] is False
