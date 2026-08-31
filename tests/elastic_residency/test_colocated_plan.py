"""LLM+TTS 显式候选矩阵的纯逻辑测试。"""

from __future__ import annotations

import pytest

from experiments.elastic_residency.colocated_plan import (
    build_execution_plan,
    parse_candidate_specs,
)
from experiments.elastic_residency.llm_tts_colocated import build_parser


def test_explicit_candidate_plan_preserves_fail_fast_order() -> None:
    candidates = parse_candidate_specs(
        [
            "4096:full:t3",
            "512:K24:T0",
            "2048:K20:T0",
        ]
    )
    assert [candidate.label for candidate in candidates] == [
        "4096:FULL:T3",
        "512:K24:T0",
        "2048:K20:T0",
    ]
    plan = build_execution_plan(candidates, runs=2)
    assert plan == [
        (4096, "FULL", "T3", 1, 1),
        (4096, "FULL", "T3", 2, 2),
        (512, "K24", "T0", 1, 3),
        (512, "K24", "T0", 2, 4),
        (2048, "K20", "T0", 1, 5),
        (2048, "K20", "T0", 2, 6),
    ]


@pytest.mark.parametrize(
    "spec,pattern",
    [
        ("4096:K20", "CONTEXT:LLM:TTS"),
        ("bad:K20:T0", "不是整数"),
        ("0:K20:T0", "context > 0"),
    ],
)
def test_invalid_candidate_is_rejected(spec: str, pattern: str) -> None:
    with pytest.raises(ValueError, match=pattern):
        parse_candidate_specs([spec])


def test_duplicate_candidate_and_invalid_runs_are_rejected() -> None:
    with pytest.raises(ValueError, match="重复"):
        parse_candidate_specs(["512:K24:T0", "512:k24:t0"])
    with pytest.raises(ValueError, match="至少为 1"):
        build_execution_plan(parse_candidate_specs(["512:K24:T0"]), runs=0)


def test_parser_accepts_six_filtered_candidates() -> None:
    specs = [
        "4096:FULL:T3",
        "2048:FULL:T3",
        "512:FULL:T3",
        "4096:K20:T0",
        "2048:K20:T0",
        "512:K24:T0",
    ]
    args = build_parser().parse_args(["--candidates", *specs, "--runs", "5"])
    assert args.candidates == specs
    assert args.runs == 5
    assert len(build_execution_plan(parse_candidate_specs(specs), runs=5)) == 30
