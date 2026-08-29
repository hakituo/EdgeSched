from types import SimpleNamespace

import pytest

from experiments.elastic_residency.llm_residency_curve import (
    _build_exact_prompt_tokens,
    _evidence_scope,
    _failed_acceptance_checks,
    _jitter_acceptance_checks,
    _phase_resource_metrics,
    _record_key,
    _token_ids_sha256,
    _validate_resume_contract,
    build_matrix_plan,
    resolve_config,
)


class _FakeLlama:
    def tokenize(self, text: bytes, add_bos: bool = True) -> list[int]:
        prefix = [1] if add_bos else []
        return prefix + list(text)


def test_resolve_llm_residency_config() -> None:
    assert resolve_config("K0", 32) == 0
    assert resolve_config("full", 32) == -1
    assert resolve_config("K16", 32) == 16
    with pytest.raises(ValueError):
        resolve_config("K33", 32)


def test_build_prompt_has_exact_token_count_and_stable_hash() -> None:
    tokens = _build_exact_prompt_tokens(_FakeLlama(), 512)
    assert len(tokens) == 512
    assert _token_ids_sha256(tokens) == _token_ids_sha256(list(tokens))
    assert len(_token_ids_sha256(tokens)) == 64


def test_phase_resource_metrics_excludes_other_phases() -> None:
    samples = [
        SimpleNamespace(
            monotonic_ns=100,
            scheduler_jitter_ms=1.0,
            gpu_sample_age_ms=20.0,
        ),
        SimpleNamespace(
            monotonic_ns=200,
            scheduler_jitter_ms=3.0,
            gpu_sample_age_ms=40.0,
        ),
        SimpleNamespace(
            monotonic_ns=300,
            scheduler_jitter_ms=99.0,
            gpu_sample_age_ms=500.0,
        ),
    ]
    metrics = _phase_resource_metrics(samples, 100, 200)
    assert metrics["sample_count"] == 2
    assert metrics["jitter_max_ms"] == 3.0
    assert metrics["gpu_sample_age_max_ms"] == 40.0


def test_matrix_plan_rotates_configs_within_each_context() -> None:
    plan = build_matrix_plan(
        (512, 2048),
        ("K0", "K8", "FULL"),
        runs=2,
    )
    assert len(plan) == 12
    assert plan[:6] == [
        (512, "K0", 1, 1),
        (512, "K8", 1, 2),
        (512, "FULL", 1, 3),
        (512, "K8", 2, 4),
        (512, "FULL", 2, 5),
        (512, "K0", 2, 6),
    ]
    assert plan[6] == (2048, "K0", 1, 7)
    assert len({(context, config, run) for context, config, run, _ in plan}) == 12


def test_matrix_plan_rejects_duplicate_dimensions() -> None:
    with pytest.raises(ValueError):
        build_matrix_plan((512, 512), ("K0", "FULL"), runs=1)
    with pytest.raises(ValueError):
        build_matrix_plan((512,), ("K0", "K0"), runs=1)


def test_record_key_is_context_config_and_run() -> None:
    assert _record_key({"context_tokens": 4096, "config": "K24", "run_index": 5}) == (
        4096,
        "K24",
        5,
    )


def test_failed_acceptance_checks_reports_only_failed_names() -> None:
    assert _failed_acceptance_checks(
        {"acceptance_checks": {"gpu_fresh": False, "swap_unused": True}}
    ) == ("gpu_fresh",)


def test_resume_contract_rejects_config_or_manifest_drift() -> None:
    _validate_resume_contract(
        existing_config={"runs": 5},
        expected_config={"runs": 5},
        existing_manifest={"commit": "abc"},
        expected_manifest={"commit": "abc"},
    )
    with pytest.raises(ValueError, match="冻结运行配置"):
        _validate_resume_contract(
            existing_config={"runs": 1},
            expected_config={"runs": 5},
            existing_manifest={"commit": "abc"},
            expected_manifest={"commit": "abc"},
        )
    with pytest.raises(ValueError, match="manifest"):
        _validate_resume_contract(
            existing_config={"runs": 5},
            expected_config={"runs": 5},
            existing_manifest={"commit": "old"},
            expected_manifest={"commit": "new"},
        )


def test_only_frozen_full_matrix_gets_c2_scope() -> None:
    configs = ("K0", "K8", "K16", "K24", "K32", "FULL")
    assert (
        _evidence_scope((512, 2048, 4096), configs, 5, 128, "observe")
        == "c2_curve"
    )
    assert _evidence_scope((512,), configs, 5, 128, "observe") == "c1_smoke"
    assert (
        _evidence_scope((512, 2048, 4096), configs, 1, 128, "observe")
        == "c1_smoke"
    )
    assert (
        _evidence_scope((512, 2048, 4096), configs, 5, 64, "observe")
        == "c1_smoke"
    )
    assert (
        _evidence_scope((512, 2048, 4096), configs, 5, 128, "hard")
        == "c1_smoke"
    )


def test_only_frozen_critical_matrix_gets_c3_scope() -> None:
    configs = ("K2", "K4", "K6", "K10", "K12", "K14", "K18", "K20", "K28", "K30")
    assert (
        _evidence_scope((512, 2048, 4096), configs, 5, 128, "observe")
        == "c3_critical_refinement"
    )
    assert _evidence_scope((512,), configs, 5, 128, "observe") == "c1_smoke"


def test_c2_observes_jitter_target_without_invalidating_measurement() -> None:
    targets = {
        "inference_resource_jitter_p95_within_target": True,
        "inference_resource_jitter_max_within_target": False,
    }
    assert _jitter_acceptance_checks(targets, "observe") == {}
    assert _jitter_acceptance_checks(targets, "hard") == {
        "inference_resource_jitter_p95_within_limit": True,
        "inference_resource_jitter_max_within_limit": False,
    }
