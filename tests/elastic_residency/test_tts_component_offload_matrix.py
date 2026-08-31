from experiments.elastic_residency.tts_component_offload_matrix import (
    EXPECTED_LENGTHS,
    TEXTS,
    _pareto_configs,
    _validate_texts,
    build_parser,
)


def test_b2_frozen_text_lengths_are_exact() -> None:
    _validate_texts()
    assert {name: len(text) for name, text in TEXTS.items()} == EXPECTED_LENGTHS


def test_pareto_filter_removes_strictly_dominated_config() -> None:
    summaries = {
        "T0": {
            "max_card_delta_mb": 2300.0,
            "ttfa_s": {"p95": 5.0},
            "rtf": {"p95": 1.8},
        },
        "T1": {
            "max_card_delta_mb": 1900.0,
            "ttfa_s": {"p95": 6.0},
            "rtf": {"p95": 2.1},
        },
        "T2": {
            "max_card_delta_mb": 1950.0,
            "ttfa_s": {"p95": 6.5},
            "rtf": {"p95": 2.2},
        },
    }
    assert _pareto_configs(summaries) == ["T0", "T1"]


def test_b2_runtime_evidence_defaults_are_strict() -> None:
    args = build_parser().parse_args([])
    assert args.max_gpu_sample_freshness_ms == 750.0
    assert args.swap_sample_interval == 0.05
    assert args.max_swap_growth_mb == 0.0
