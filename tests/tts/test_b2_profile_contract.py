"""B2 冻结口径的静态验证：文本、序列、汇总形状与放置状态。"""

import ast
import os
import pathlib

import pytest

from experiments.elastic_residency.analyze_results import extract_tts_profiles
from experiments.tts import b2_profile_contract as contract
from experiments.tts.tts_phase_memory import PhaseMemory


def _rows() -> list[dict[str, object]]:
    return [
        {
            "path": "cuda_graph_static",
            "text": name,
            "call_index": index,
            "rtf": 0.3 + index / 100,
            "waveform_ready_s": 1.0 + index,
            "audio_s": 4.7,
            "pcm_sha256": f"h{index}",
        }
        for index, name in enumerate(("chars_20", "chars_50", "chars_100"), start=1)
    ]


def _phases() -> dict[str, object]:
    return {
        "prepare": {
            "load_s": 12.0,
            "graph_capture_s": 3.0,
            "rss_mb_max": 3000.0,
            "torch_peak_allocated_mb": 1800.0,
            "torch_peak_reserved_mb": 2000.0,
        },
        "execute": {
            "rss_mb_max": 3100.0,
            "torch_peak_allocated_mb": 1900.0,
            "torch_peak_reserved_mb": 2100.0,
            "gpu_card_used_mb_max": 2600.0,
        },
        "placement_status": {"supported": [contract.PLACEMENT_CURRENT]},
    }


def test_module_imports_stay_light_for_isolated_env() -> None:
    # 隔离环境只有 numpy；contract 不得在模块级引入 torch/psutil
    tree = ast.parse(pathlib.Path(contract.__file__).read_text(encoding="utf-8"))
    modules = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])
    assert "torch" not in modules
    assert "psutil" not in modules


def test_frozen_texts_are_exact() -> None:
    contract.validate_texts()
    assert {name: len(text) for name, text in contract.TEXTS.items()} == dict(
        contract.EXPECTED_LENGTHS
    )
    assert list(contract.TEXTS) == ["chars_20", "chars_50", "chars_100"]


def test_length_sequence_repeats_each_band() -> None:
    sequence = contract.length_sequence(2)
    assert [name for name, _ in sequence] == [
        "chars_20", "chars_20", "chars_50", "chars_50", "chars_100", "chars_100",
    ]
    with pytest.raises(ValueError, match="rounds"):
        contract.length_sequence(0)


def test_correctness_sequence_interleaves_repeats() -> None:
    sequence = contract.correctness_sequence()
    names = [name for name, _ in sequence]
    assert len(sequence) == 7
    assert names.count(contract.CORRECTNESS_REPEAT) == 5
    assert names[3] == "chars_100"
    assert names[5] == "chars_50"


def test_b2_summary_is_analyzable_by_capacity_screen() -> None:
    summary = contract.build_b2_summary(
        _rows(),
        integration_path="cuda_graph_static",
        phases=_phases(),
        idle_card_used_mb=600.0,
        git_state={"available": True, "dirty": False},
    )
    entry = summary["by_config"][contract.PLACEMENT_CURRENT]
    assert entry["max_card_delta_mb"] == 2000.0
    assert entry["max_card_delta_mb_source"] == "direct_nvml_delta"
    assert entry["load_s"] == 12.0
    assert entry["graph_capture_s"] == 3.0
    assert entry["waveform_ready_mean_s"] == 3.0
    assert len(summary["by_path"]) == 3
    # 波形正确性核对前不得自称通过；来源干净才算 provenance 完整
    assert summary["passed"] is False
    assert summary["git_provenance_complete"] is True
    profiles = extract_tts_profiles(summary)
    assert profiles[contract.PLACEMENT_CURRENT]["max_card_delta_mb"] == 2000.0
    assert profiles[contract.PLACEMENT_CURRENT]["allocator_slack_mb"] == 200.0


def test_b2_summary_flags_dirty_or_missing_provenance() -> None:
    for git_state in ({"available": True, "dirty": True}, None, {"available": False}):
        summary = contract.build_b2_summary(
            _rows(),
            integration_path="cuda_graph_static",
            phases=_phases(),
            idle_card_used_mb=600.0,
            git_state=git_state,
        )
        assert summary["git_provenance_complete"] is False


def test_b2_summary_falls_back_to_torch_peak_without_nvml() -> None:
    phases = _phases()
    phases["execute"].pop("gpu_card_used_mb_max")
    summary = contract.build_b2_summary(
        _rows(),
        integration_path="cuda_graph_static",
        phases=phases,
        idle_card_used_mb=None,
    )
    entry = summary["by_config"][contract.PLACEMENT_CURRENT]
    assert entry["max_card_delta_mb"] == 2100.0
    assert entry["max_card_delta_mb_source"] == "torch_peak_reserved_fallback"


def test_b2_summary_requires_integration_rows() -> None:
    with pytest.raises(ValueError, match="接入路径"):
        contract.build_b2_summary(
            _rows(),
            integration_path="upstream_dynamic",
            phases=_phases(),
            idle_card_used_mb=600.0,
        )


def test_placement_status_excludes_unconfirmed_dimensions() -> None:
    status = contract.placement_status({"whole_model_device_parameter": True})
    assert status["supported"] == [contract.PLACEMENT_CURRENT]
    assert status["excluded"] == ["T1", "T2", "T3", "T4"]
    assert "不沿用旧值" in status["rule"]
    assert contract.placement_status({})["supported"] == []


def test_phase_memory_records_without_cuda() -> None:
    phase = PhaseMemory("execute", os.getpid())
    phase.sample()
    record = phase.stop()
    assert record["label"] == "execute"
    assert record["sample_count"] == 1
    assert record["torch_peak_semantics"].startswith("max_memory_allocated")
