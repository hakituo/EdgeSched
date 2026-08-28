from types import SimpleNamespace

import pytest

from experiments.elastic_residency.llm_residency_curve import (
    _build_exact_prompt_tokens,
    _phase_resource_metrics,
    _token_ids_sha256,
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
