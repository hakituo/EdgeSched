"""视觉证据语义、prompt 构造、门禁装配与汇总的静态测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.elastic_residency.vision.vision_endpoint_checks import (  # noqa: E402
    build_acceptance_checks,
)
from experiments.elastic_residency.vision.vision_endpoint_report import (  # noqa: E402
    summarize_by_cell,
    write_summary_csv,
)
from experiments.elastic_residency.vision.vision_evidence import (  # noqa: E402
    ALL_STATES,
    RESIDENCY_TEXT_ONLY,
    RESIDENCY_VISION_WARM_CPU,
    RESIDENCY_VISION_WARM_GPU,
    VISION_MEASUREMENT_SEMANTICS,
    build_prompt_evidence,
    image_file_evidence,
    memory_delta,
    output_text_checks,
    projector_execution_checks,
    qwen_chat_prompt,
)


REQUIRED_SEMANTIC_FIELDS = (
    "residency_state",
    "visual_token_total",
    "visual_position_total",
    "projector_encode_s",
    "visual_embedding_available",
    "projector_load_s",
    "projector_free_s",
    "projector_gpu_card_delta_mb",
    "projector_rss_delta_bytes",
    "prompt_eval_s",
    "image_file_sha256",
    "image_pixel_payload_sha256",
    "text_only_prompt_semantics",
)

CHAT_TEMPLATE = (
    "{%- for message in messages %}"
    "{{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n' }}{%- endif %}"
)


def _session_evidence() -> dict[str, object]:
    return {
        "residency_state": RESIDENCY_VISION_WARM_GPU,
        "supports_vision": True,
        "projector_load_s": 1.25,
        "media_marker": "<__media__>",
    }


def _chunk_evidence(**overrides: object) -> dict[str, object]:
    payload = {
        "image_chunk_count": 1,
        "text_chunk_count": 2,
        "visual_token_total": 196,
        "visual_position_total": 196,
        "projector_encode_total_s": 0.31,
        "image_decode_total_s": 0.05,
        "visual_embedding_available": True,
    }
    payload.update(overrides)
    return payload


def _bitmaps() -> list[dict[str, object]]:
    return [
        {
            "decoded_width_px": 448,
            "decoded_height_px": 448,
            "decoded_bytes": 448 * 448 * 3,
            "is_audio": False,
        }
    ]


class TestSemanticsAndStates:
    def test_all_states_are_distinct_and_documented(self) -> None:
        assert len(set(ALL_STATES)) == len(ALL_STATES)
        assert RESIDENCY_TEXT_ONLY in ALL_STATES
        assert RESIDENCY_VISION_WARM_CPU in ALL_STATES
        for field in REQUIRED_SEMANTIC_FIELDS:
            assert field in VISION_MEASUREMENT_SEMANTICS
            assert VISION_MEASUREMENT_SEMANTICS[field].strip()


class TestPromptConstruction:
    def test_qwen_markup_includes_marker_and_generation_prompt(self) -> None:
        text = qwen_chat_prompt(question="描述图片", media_marker="<__media__>")
        assert "<__media__>" in text
        assert text.endswith("<think>\n\n</think>\n\n")
        assert "<|im_start|>assistant" in text

    def test_build_prompt_evidence_records_hash_and_marker(self) -> None:
        evidence = build_prompt_evidence(
            mode="qwen_im", question="描述图片", media_marker="<__media__>"
        )
        assert evidence["prompt_construction_mode"] == "qwen_im"
        assert evidence["prompt_contains_media_marker"] is True
        assert len(evidence["prompt_text_sha256"]) == 64

    def test_text_only_prompt_has_no_marker(self) -> None:
        evidence = build_prompt_evidence(
            mode="qwen_im", question="描述图片", media_marker=None
        )
        assert evidence["prompt_contains_media_marker"] is False
        assert "<__media__>" not in evidence["prompt_text"]

    def test_model_template_mode_uses_provided_template(self) -> None:
        pytest.importorskip("jinja2")
        evidence = build_prompt_evidence(
            mode="model_template",
            question="描述图片",
            media_marker="<__media__>",
            chat_template=CHAT_TEMPLATE,
            eos_token="<|im_end|>",
            bos_token="",
        )
        assert evidence["prompt_text"].startswith("<|im_start|>user")
        assert evidence["prompt_contains_media_marker"] is True

    def test_invalid_modes_are_rejected(self) -> None:
        with pytest.raises(ValueError):
            build_prompt_evidence(
                mode="model_template", question="q", media_marker=None
            )
        with pytest.raises(ValueError):
            build_prompt_evidence(mode="auto", question="q", media_marker=None)


class TestMemoryAndImageEvidence:
    def test_memory_delta_handles_missing_gpu_values(self) -> None:
        delta = memory_delta(
            before={"gpu_card_used_mb": 100.0},
            after={"gpu_card_used_mb": 640.5},
            before_rss_bytes=1000,
            after_rss_bytes=4000,
        )
        assert delta["gpu_card_delta_mb"] == pytest.approx(540.5)
        assert delta["rss_delta_bytes"] == 3000
        missing = memory_delta(
            before={"gpu_card_used_mb": None},
            after={"gpu_card_used_mb": 1.0},
            before_rss_bytes=0,
            after_rss_bytes=0,
        )
        assert missing["gpu_card_delta_mb"] is None

    def test_image_file_evidence_records_hash(self, tmp_path: Path) -> None:
        path = tmp_path / "photo.png"
        path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"payload")
        evidence = image_file_evidence(path)
        assert evidence["source"] == "external_file"
        assert len(evidence["file_sha256"]) == 64
        assert evidence["pixel_payload_sha256"] is None
        with pytest.raises(FileNotFoundError):
            image_file_evidence(tmp_path / "absent.png")


class TestProjectorChecks:
    def test_real_execution_passes(self) -> None:
        checks = projector_execution_checks(
            session_evidence=_session_evidence(),
            chunk_evidence=_chunk_evidence(),
            bitmap_evidence=_bitmaps(),
            expected_visual_tokens=196,
        )
        assert all(checks.values())
        assert checks["projector_returned_visual_embedding"] is True
        assert checks["visual_token_count_matches_patch_expectation"] is True

    def test_zero_visual_tokens_fails(self) -> None:
        checks = projector_execution_checks(
            session_evidence=_session_evidence(),
            chunk_evidence=_chunk_evidence(
                visual_token_total=0, visual_position_total=0
            ),
            bitmap_evidence=_bitmaps(),
        )
        assert not checks["visual_tokens_positive"]
        assert not checks["visual_positions_advanced_llm_context"]

    def test_chunk_single_mode_omits_embedding_checks(self) -> None:
        checks = projector_execution_checks(
            session_evidence=_session_evidence(),
            chunk_evidence=_chunk_evidence(
                visual_embedding_available=None, projector_encode_total_s=None
            ),
            bitmap_evidence=_bitmaps(),
        )
        assert "projector_returned_visual_embedding" not in checks
        assert "projector_encode_time_isolated" not in checks

    def test_audio_bitmap_and_missing_image_chunk_fail(self) -> None:
        bitmaps = _bitmaps()
        bitmaps[0]["is_audio"] = True
        checks = projector_execution_checks(
            session_evidence=_session_evidence(),
            chunk_evidence=_chunk_evidence(image_chunk_count=0),
            bitmap_evidence=bitmaps,
        )
        assert not checks["all_bitmaps_decoded_as_image"]
        assert not checks["image_chunk_count_matches_input"]

    def test_unsupported_vision_context_fails(self) -> None:
        session = _session_evidence()
        session["supports_vision"] = False
        session["projector_load_s"] = 0.0
        checks = projector_execution_checks(
            session_evidence=session,
            chunk_evidence=_chunk_evidence(),
            bitmap_evidence=_bitmaps(),
        )
        assert not checks["projector_context_supports_vision"]
        assert not checks["projector_load_time_recorded"]

    def test_output_text_checks(self) -> None:
        assert all(
            output_text_checks(
                output_text="白底红圆", generated_token_count=128,
                expected_token_count=128,
            ).values()
        )
        failing = output_text_checks(
            output_text="   ", generated_token_count=127, expected_token_count=128
        )
        assert not failing["output_text_non_empty"]
        assert not failing["generation_token_count_exact"]


def _phase(*, vision: bool) -> dict[str, object]:
    return {
        "kv_native_state": {
            "kv_tokens_used": 344,
            "kv_sequence_state_bytes": 24_000_000,
            "full_state_serialized_bytes": 70_263_000,
        },
        "kv_roundtrip": {"succeeded": True},
        "inference": {
            "prefill_tokens_per_s": 500.0,
            "decode_tokens_per_s": 40.0,
            "generated_token_count": 128,
            "token_gap_ms": {"count": 127},
            "ttft_s": 0.9,
        },
        "output": {"output_text": "白底红圆", "output_text_sha256": "a" * 64},
        "prompt": {"prompt_token_count": 20},
        "vision": (
            {
                "session": _session_evidence(),
                "chunk_execution": _chunk_evidence(),
                "bitmaps": _bitmaps(),
                "release": {"projector_freed": True, "projector_free_s": 0.02},
                "projector_warm_delta": {
                    "gpu_card_delta_mb": 640.0,
                    "rss_delta_bytes": 12345,
                },
            }
            if vision
            else None
        ),
    }


def _runtime_evidence() -> dict[str, object]:
    return {
        "checks": {
            "gpu_periodic_samples_fresh": True,
            "gpu_capacity_snapshots_fresh": True,
            "swap_growth_within_tolerance": True,
        },
        "swap": {"growth_bytes": 0},
    }


def _resource_summary() -> dict[str, object]:
    return {
        "gpu_unavailable_timeline_sample_count": 0,
        "gpu_telemetry_backend": "direct_nvml",
        "gpu_cli_subprocess_used": False,
        "sampler_thread_affinity": [0, 1],
        "gpu_thread_affinity": [0, 1],
    }


def _build_checks(*, vision: bool, **overrides: object) -> dict[str, bool]:
    kwargs: dict[str, object] = {
        "uses_vision": vision,
        "phase": _phase(vision=vision),
        "buffer_report": {"kv_gpu_buffer_mb": 24.0, "compute_gpu_buffer_mb": 100.0},
        "runtime_evidence": _runtime_evidence(),
        "resource_summary": _resource_summary(),
        "capacity_snapshots": [
            {"gpu_telemetry_backend": "direct_nvml", "gpu_cli_subprocess_used": False}
        ],
        "allocator_after_release": {
            "active_lease_count": 0,
            "free_physical_core_count": 16,
        },
        "physical_core_count": 16,
        "actual_layers": 32,
        "expected_layers": 32,
        "generation_tokens": 128,
        "expected_state_tokens": 344,
        "gpu_ready": True,
        "actual_affinity": (2, 3),
        "llm_affinity": (2, 3),
        "system_affinity": (0, 1),
        "expected_visual_tokens": 196 if vision else None,
    }
    kwargs.update(overrides)
    return build_acceptance_checks(**kwargs)  # type: ignore[arg-type]


class TestAcceptanceChecks:
    def test_vision_unit_passes(self) -> None:
        checks = _build_checks(vision=True)
        assert all(checks.values())
        assert checks["projector_released_after_run"] is True

    def test_text_unit_omits_projector_checks(self) -> None:
        checks = _build_checks(vision=False)
        assert all(checks.values())
        assert "projector_returned_visual_embedding" not in checks

    def test_state_mismatch_is_detected(self) -> None:
        checks = _build_checks(vision=True, phase=_phase(vision=False))
        assert not checks["residency_state_matches_projector_state"]

    def test_cli_fallback_and_stale_gpu_fail(self) -> None:
        summary = _resource_summary()
        summary["gpu_telemetry_backend"] = "nvidia_smi_cli"
        summary["gpu_cli_subprocess_used"] = True
        evidence = _runtime_evidence()
        evidence["checks"]["gpu_periodic_samples_fresh"] = False
        checks = _build_checks(
            vision=True, resource_summary=summary, runtime_evidence=evidence
        )
        assert not checks["gpu_telemetry_is_direct_nvml"]
        assert not checks["gpu_periodic_samples_fresh"]

    def test_kv_and_core_return_failures(self) -> None:
        checks = _build_checks(
            vision=False,
            expected_state_tokens=999,
            allocator_after_release={
                "active_lease_count": 1,
                "free_physical_core_count": 14,
            },
        )
        assert not checks["kv_tokens_match_executed_positions"]
        assert not checks["all_leases_released"]
        assert not checks["all_physical_cores_returned"]


def _record(*, config: str, state: str, run_index: int = 1) -> dict[str, object]:
    vision = state.startswith("VISION")
    phase = _phase(vision=vision)
    return {
        "config": config,
        "residency_state": state,
        "run_index": run_index,
        "passed": True,
        "n_gpu_layers": 0 if config == "K0" else -1,
        "llm_load_s": 2.0,
        "inference": phase["inference"],
        "prompt": phase["prompt"],
        "output": phase["output"],
        "vision": phase["vision"],
        "memory_baseline": {"gpu_card_used_mb": 100.0},
        "memory_after_llm_load": {"gpu_card_used_mb": 3000.0},
        "memory_after_inference": {"gpu_card_used_mb": 3200.0},
        "memory_after_inference_rss_bytes": 1_200_000_000,
        "runtime_evidence": _runtime_evidence(),
    }


class TestReport:
    def test_cells_are_grouped_by_config_and_state(self) -> None:
        records = [
            _record(config="K0", state=RESIDENCY_TEXT_ONLY),
            _record(config="K0", state=RESIDENCY_VISION_WARM_GPU),
            _record(config="FULL", state=RESIDENCY_VISION_WARM_GPU),
        ]
        summary = summarize_by_cell(records)
        assert set(summary) == {
            f"K0|{RESIDENCY_TEXT_ONLY}",
            f"K0|{RESIDENCY_VISION_WARM_GPU}",
            f"FULL|{RESIDENCY_VISION_WARM_GPU}",
        }
        cell = summary[f"K0|{RESIDENCY_TEXT_ONLY}"]
        assert cell["samples"] == 1
        assert cell["all_passed"] is True
        assert cell["gpu_delta_after_llm_mb"]["mean"] == pytest.approx(2900.0)

    def test_csv_contains_projector_columns(self, tmp_path: Path) -> None:
        target = tmp_path / "summary.csv"
        write_summary_csv(
            target,
            [
                _record(config="FULL", state=RESIDENCY_TEXT_ONLY),
                _record(config="FULL", state=RESIDENCY_VISION_WARM_GPU),
            ],
        )
        lines = target.read_text(encoding="utf-8").splitlines()
        assert "visual_token_total" in lines[0]
        assert "projector_gpu_card_delta_mb" in lines[0]
        assert len(lines) == 3
        assert lines[1].split(",")[10] == ""  # 纯文本行没有视觉 token
        assert "196" in lines[2]

    def test_empty_records_produce_header_only(self, tmp_path: Path) -> None:
        target = tmp_path / "empty.csv"
        write_summary_csv(target, [])
        assert len(target.read_text(encoding="utf-8").splitlines()) == 1
        assert summarize_by_cell([]) == {}
