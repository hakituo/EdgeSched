"""KV 校准入口与 LLM+TTS 共置公式/口径的单元测试。"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.elastic_residency.llm_kv_cache_calibration import (  # noqa: E402
    FORMAL_CONFIGS,
    FORMAL_CONTEXTS,
    _evidence_scope,
    build_parser as build_kv_parser,
)
from experiments.elastic_residency.llm_tts_colocated import (  # noqa: E402
    build_parser as build_colocated_parser,
    colocated_metrics,
    parse_isolated_increments,
)


class TestKvCalibrationScope:
    def test_formal_matrix_is_16_configs_3_contexts(self) -> None:
        assert len(FORMAL_CONFIGS) == 16
        assert len(set(FORMAL_CONFIGS)) == 16
        assert "K0" in FORMAL_CONFIGS and "FULL" in FORMAL_CONFIGS
        assert FORMAL_CONTEXTS == (512, 2048, 4096)
        assert len(FORMAL_CONFIGS) * len(FORMAL_CONTEXTS) == 48

    def test_formal_scope_requires_full_matrix(self) -> None:
        assert (
            _evidence_scope(FORMAL_CONTEXTS, FORMAL_CONFIGS, runs=1)
            == "kv_calibration_formal_48"
        )

    def test_partial_matrix_is_smoke(self) -> None:
        partial = tuple(name for name in FORMAL_CONFIGS if name != "K8")
        assert _evidence_scope(FORMAL_CONTEXTS, partial, runs=1) == (
            "kv_calibration_smoke"
        )
        assert _evidence_scope((512,), FORMAL_CONFIGS, runs=1) == (
            "kv_calibration_smoke"
        )

    def test_default_state_matches_c2_c3_post_generation_shape(self) -> None:
        args = build_kv_parser().parse_args([])
        assert args.generation_tokens == 128
        assert args.context_tokens + args.generation_tokens == 640


class TestColocatedMatrixContract:
    def test_default_generation_matches_c2_c3(self) -> None:
        args = build_colocated_parser().parse_args([])
        assert args.generation_tokens == 128

    def test_formal_repeat_count_is_accepted_by_parser(self) -> None:
        args = build_colocated_parser().parse_args(["--runs", "5"])
        assert args.runs == 5


class TestParseIsolatedIncrements:
    def test_parses_config_value_pairs(self) -> None:
        parsed = parse_isolated_increments(["T0=1234.5", "t3=890"])
        assert parsed == {"T0": 1234.5, "T3": 890.0}

    def test_none_yields_empty(self) -> None:
        assert parse_isolated_increments(None) == {}

    def test_invalid_entry_raises(self) -> None:
        with pytest.raises(ValueError, match="CONFIG=value"):
            parse_isolated_increments(["T0"])


class TestColocatedMetrics:
    def test_increment_and_residual(self) -> None:
        metrics = colocated_metrics(
            b_llm_mb=4000.0,
            p_colocated_mb=5500.0,
            gpu_total_mb=16384.0,
            safety_margin_mb=1024.0,
            tts_isolated_increment_mb=1400.0,
        )
        assert metrics["tts_colocated_increment_mb"] == pytest.approx(1500.0)
        assert metrics["colocated_positive_residual_mb"] == pytest.approx(100.0)
        assert metrics["capacity_headroom_mb"] == pytest.approx(16384.0 - 5500.0)
        assert metrics["capacity_admissible"] is True

    def test_negative_residual_clamps_to_zero(self) -> None:
        metrics = colocated_metrics(
            b_llm_mb=4000.0,
            p_colocated_mb=5000.0,
            gpu_total_mb=16384.0,
            safety_margin_mb=1024.0,
            tts_isolated_increment_mb=1400.0,
        )
        assert metrics["colocated_positive_residual_mb"] == 0.0

    def test_missing_isolated_increment_keeps_residual_none(self) -> None:
        metrics = colocated_metrics(
            b_llm_mb=4000.0,
            p_colocated_mb=5000.0,
            gpu_total_mb=16384.0,
            safety_margin_mb=1024.0,
            tts_isolated_increment_mb=None,
        )
        assert metrics["colocated_positive_residual_mb"] is None
        assert metrics["tts_colocated_increment_mb"] == pytest.approx(1000.0)

    def test_capacity_verdict_uses_measured_peak_only(self) -> None:
        tight = colocated_metrics(
            b_llm_mb=4000.0,
            p_colocated_mb=15800.0,
            gpu_total_mb=16384.0,
            safety_margin_mb=1024.0,
            tts_isolated_increment_mb=100.0,
        )
        assert tight["capacity_admissible"] is False
        assert tight["capacity_headroom_mb"] == pytest.approx(584.0)

    def test_peak_below_baseline_raises(self) -> None:
        with pytest.raises(ValueError, match="口径不一致"):
            colocated_metrics(
                b_llm_mb=5000.0,
                p_colocated_mb=4999.0,
                gpu_total_mb=16384.0,
                safety_margin_mb=1024.0,
                tts_isolated_increment_mb=100.0,
            )

    def test_none_inputs_raise_instead_of_zero(self) -> None:
        with pytest.raises(ValueError, match="不得计算"):
            colocated_metrics(
                b_llm_mb=None,  # type: ignore[arg-type]
                p_colocated_mb=100.0,
                gpu_total_mb=16384.0,
                safety_margin_mb=1024.0,
                tts_isolated_increment_mb=100.0,
            )

    def test_capacity_formula_does_not_accept_kv_arguments(self) -> None:
        """防止 KV 重复计入容量：签名不接受任何 KV 字段。"""

        parameters = inspect.signature(colocated_metrics).parameters
        for name in parameters:
            assert "kv" not in name.lower(), (
                f"colocated_metrics 不得接受 KV 字段: {name}"
            )
        metrics = colocated_metrics(
            b_llm_mb=4000.0,
            p_colocated_mb=5000.0,
            gpu_total_mb=16384.0,
            safety_margin_mb=1024.0,
            tts_isolated_increment_mb=100.0,
        )
        assert metrics["kv_buffer_added_to_capacity"] is False
        assert "不额外叠加" in metrics["capacity_formula"]
