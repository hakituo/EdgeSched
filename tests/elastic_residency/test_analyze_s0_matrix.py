"""S0 成对比较分析测试：配对差、协变量检查与三个问题的回答。"""

import json

import pytest

from experiments.elastic_residency.analyze_s0_matrix import (
    analyze,
    load_records,
    verify_protocol,
)
from experiments.elastic_residency.s0_matrix_report import (
    aggregate,
    answer_questions,
    confound_checks,
    covariate_normalization,
    extract_metrics,
    paired_deltas,
    sentence_aligned_comparison,
    startup_policy_comparison,
)
from experiments.elastic_residency.s0_pipeline.config import (
    FREEZE_STATUS,
    S0_PLACEMENTS,
    S0_PROTOCOL_VERSION,
)
from experiments.elastic_residency.s0_pipeline.measurement_scope import (
    SCOPE_VERSION,
    STATUS_PENDING,
)

_T0 = 1_000_000_000


def _raw(
    placement: str,
    run_index: int,
    *,
    ready: float,
    rtf: float,
    faults: int,
    tokens: int,
    sha: str,
    sentences: int = 3,
    policy: str = "first_sentence",
) -> dict:
    events = [{"stage": "request_start", "timestamp_ns": _T0}]
    for index in range(sentences):
        events.append(
            {
                "stage": "waveform_ready",
                "timestamp_ns": _T0 + int(ready * 1e9) + index * 1_000_000,
                "chunk_id": index,
            }
        )
    events.append(
        {
            "stage": "playback_start",
            "timestamp_ns": _T0 + int((ready + 0.5) * 1e9),
            "chunk_id": 0,
        }
    )
    return {
        "placement": placement,
        "startup_policy": policy,
        "run_index": run_index,
        "passed": True,
        "llm_identity": {"model_path": "m"},
        "swap_report": {
            "within_tolerance": True,
            "growth_bytes": 0,
            "tolerance_bytes": 4 * 1024 * 1024,
        },
        "output_fingerprint": {
            "raw_text_sha256": sha,
            "raw_text_chars": 12 * sentences,
            "sentence_count": sentences,
            "output_tokens": tokens,
            "audio_duration_s": float(sentences),
        },
        "run": {
            "finished": True,
            "error": "",
            "event_count": len(events),
            "events": events,
            "sentences": [
                {
                    "chunk_id": index,
                    "queue_wait_s": 0.1,
                    "synthesis_s": 0.5,
                    "rtf": rtf,
                    "audio_duration_s": 1.0,
                    "raw_text": f"句{index}。",
                    "audio_sha256": f"{index:064d}",
                }
                for index in range(sentences)
            ],
            "text_contract": {"failed": False, "failures": []},
            "playback_calibration": {"verified": False, "reasons": ["跨度比 1.12"]},
            "playback_stats": {"device_underflow_count": 0, "backend": "sounddevice"},
            "playback_starvation": {
                "starve_episode_count": faults,
                "starve_callback_count": faults,
                "supply_fault_episode_count": faults,
                "supply_fault_samples": faults * 1000,
                "starve_samples_in_window": faults * 1000,
                "window_samples": 100000,
                "starve_ratio_in_window": 0.01,
                "by_phase": {},
            },
        },
    }


def _write_run_dir(tmp_path, records) -> None:
    (tmp_path / "iterations.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in records) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "protocol_version": S0_PROTOCOL_VERSION,
        "measurement_scope_version": SCOPE_VERSION,
        "freeze_status": FREEZE_STATUS,
        "protocol": {"repetitions": {"formal_placements": list(S0_PLACEMENTS)}},
    }
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )


def _agg(first, last, tokens, sentences, audio, faults, ratio) -> dict:
    return {
        "first_sentence_ready_s": {"mean": first},
        "last_sentence_ready_s": {"mean": last},
        "output_tokens": {"mean": tokens},
        "sentence_count": {"mean": sentences},
        "audio_duration_s": {"mean": audio},
        "supply_fault_episodes": {"mean": faults},
        "starve_ratio_in_window": {"mean": ratio},
    }


def _aligned_run(placement: str, run_index: int, series: list[float]) -> dict:
    return {
        "placement": placement,
        "startup_policy": "first_sentence",
        "run_index": run_index,
        "passed": True,
        "sentence_ready_s": series,
    }


def test_sentence_aligned_comparison_ignores_extra_sentences() -> None:
    # 参考 9 句、基线 8 句：对齐到前 8 句，基线每句都更慢
    runs = [
        _aligned_run("K24:T0", 1, [4.0, 5.5, 7.0, 10.0, 13.5, 16.5, 19.5, 21.3, 25.0]),
        _aligned_run("K0:T0", 1, [10.0, 14.5, 18.5, 27.0, 36.5, 46.5, 52.5, 56.0]),
    ]
    result = sentence_aligned_comparison(runs)
    assert result["aligned_depth"] == 8
    assert result["paired_run_count"] == 1
    assert result["all_sentences_baseline_worse"] is True
    assert len(result["per_sentence"]) == 8
    assert "不足以解释" in result["verdict"]


def test_sentence_aligned_comparison_detects_inconsistency() -> None:
    runs = [
        _aligned_run("K24:T0", 1, [4.0, 5.5, 7.0]),
        _aligned_run("K0:T0", 1, [10.0, 4.0, 18.5]),
    ]
    result = sentence_aligned_comparison(runs)
    assert result["all_sentences_baseline_worse"] is False
    assert "不能支持改善结论" in result["verdict"]


def test_sentence_aligned_comparison_without_pair() -> None:
    result = sentence_aligned_comparison([_aligned_run("K24:T0", 1, [4.0])])
    assert result["aligned_depth"] == 0
    assert "无法做句数对齐" in result["verdict"]


def test_covariate_normalization_keeps_direction() -> None:
    # 基线说得更少却更慢：归一化后差距不缩小，长度差异不足以解释
    aggregates = {
        "K24:T0|first_sentence": _agg(4.384, 25.24, 80, 9, 31.6, 0, 0.0),
        "K0:T0|first_sentence": _agg(10.74, 56.05, 65, 8, 26.46, 6, 0.4414),
    }
    result = covariate_normalization(aggregates)
    assert result["comparisons"]["last_ready_s_per_token"]["direction_preserved"] is True
    assert result["gap_shrinking_metrics"] == []
    assert len(result["direction_preserved_metrics"]) == len(result["comparisons"])
    # 基线 token 更少，按 token 归一后每 token 更慢
    assert (
        result["comparisons"]["last_ready_s_per_token"]["normalized_baseline"]
        > result["comparisons"]["last_ready_s_per_token"]["normalized_reference"]
    )
    assert "不足以解释" in result["verdict"]


def test_covariate_normalization_detects_shrinking_gap() -> None:
    # 基线 token 数倍于参考：归一后差距几乎消失，说明「说得更多」才是主因
    aggregates = {
        "K24:T0|first_sentence": _agg(4.0, 10.0, 100, 5, 20.0, 0, 0.0),
        "K0:T0|first_sentence": _agg(5.0, 20.2, 200, 10, 40.0, 0, 0.0),
    }
    result = covariate_normalization(aggregates)
    assert result["gap_shrinking_metrics"]
    assert "缩小" in result["verdict"]


def test_covariate_normalization_treats_zero_reference_as_baseline_worse() -> None:
    # 参考 0 段缺口、基线 6 段：比值无定义，但这是「基线更差」的最强证据
    aggregates = {
        "K24:T0|first_sentence": _agg(4.0, 25.0, 80, 9, 31.6, 0, 0.0),
        "K0:T0|first_sentence": _agg(10.0, 56.0, 65, 8, 26.4, 6, 0.4414),
    }
    result = covariate_normalization(aggregates)
    fault = result["comparisons"]["supply_faults_per_10s_audio"]
    assert fault["normalized_gap"] is None
    assert fault["status"] == "baseline_worse"
    assert fault["direction_preserved"] is True
    assert "不足以解释" in result["verdict"]


def test_covariate_normalization_detects_reversal() -> None:
    aggregates = {
        "K24:T0|first_sentence": _agg(4.0, 10.0, 80, 9, 31.6, 4, 0.4),
        "K0:T0|first_sentence": _agg(10.0, 30.0, 65, 8, 26.4, 1, 0.1),
    }
    result = covariate_normalization(aggregates)
    assert "反转" in result["verdict"]


def test_covariate_normalization_requires_both_placements() -> None:
    result = covariate_normalization({"K24:T0|first_sentence": _agg(4, 10, 80, 9, 31, 0, 0.0)})
    assert result["comparisons"] == {}
    assert result["verdict"].startswith("归一化后无差异或样本不足")


def _records() -> list[dict]:
    records = []
    for run_index in (1, 2):
        records.append(
            _raw("K24:T0", run_index, ready=12.0, rtf=2.2, faults=8, tokens=80, sha="a" * 64)
        )
        records.append(
            _raw("K24:T3", run_index, ready=8.0, rtf=1.8, faults=5, tokens=80, sha="a" * 64)
        )
        records.append(
            _raw("K0:T0", run_index, ready=19.0, rtf=2.0, faults=4, tokens=65, sha="b" * 64)
        )
        records.append(
            _raw("K24:T4", run_index, ready=39.0, rtf=8.0, faults=0, tokens=80, sha="a" * 64)
        )
    return records


def test_extract_metrics_reports_device_pending() -> None:
    metrics = extract_metrics(_raw("K24:T0", 1, ready=10.0, rtf=2.0, faults=1, tokens=80, sha="a" * 64))
    assert metrics["first_sentence_ready_s"] == 10.0
    assert metrics["device_status"] == STATUS_PENDING
    assert metrics["actual_first_playback_s"] is None
    assert metrics["proxy_first_playback_s"] == 10.5


def test_paired_delta_uses_run_index_pairing() -> None:
    runs = [extract_metrics(item) for item in _records()]
    pair = paired_deltas(runs, baseline="K24:T0", target="K24:T3")
    assert pair["paired_run_count"] == 2
    assert pair["first_sentence_ready_s"]["mean_delta"] == -4.0
    assert pair["first_sentence_ready_s"]["improved_runs"] == 2
    assert pair["text_identical_all"] is True


def test_confound_check_flags_output_token_differences() -> None:
    runs = [extract_metrics(item) for item in _records()]
    confounds = confound_checks(runs)
    assert confounds["prebuffer_confound_present"] is False
    assert confounds["per_placement"]["K0:T0"]["distinct_output_tokens"] == [65]
    assert confounds["per_placement"]["K24:T0"]["distinct_output_tokens"] == [80]


def test_answers_pick_candidate_with_real_gain() -> None:
    runs = [extract_metrics(item) for item in _records()]
    aggregates = aggregate(runs)
    pair = paired_deltas(runs, baseline="K24:T0", target="K24:T3")
    answers = answer_questions(
        aggregates,
        {"K24:T0->K24:T3": pair},
        confound_checks(runs),
    )
    candidates = {
        item["placement"]
        for item in answers["q3_candidates_for_dynamic_switching"]["candidates"]
    }
    assert "K24:T3" in candidates


def test_analyze_end_to_end_and_protocol_guard(tmp_path) -> None:
    _write_run_dir(tmp_path, _records())
    assert len(load_records(tmp_path)) == 8
    report = verify_protocol(tmp_path)
    assert report["protocol_version"] == S0_PROTOCOL_VERSION
    analysis = analyze(tmp_path)
    assert analysis["run_count"] == 8
    assert analysis["aggregates"]["K24:T3|first_sentence"]["first_sentence_ready_s"][
        "mean"
    ] == 8.0
    assert analysis["device_metrics"]["status"] == STATUS_PENDING
    assert "K24:T0->K24:T3" in analysis["paired"]


def test_startup_policy_comparison_reports_wait_cost_and_gap() -> None:
    records = [
        _raw(
            "K24:T0",
            1,
            ready=12.0,
            rtf=2.0,
            faults=8,
            tokens=80,
            sha="a" * 64,
            policy="first_sentence",
        ),
        _raw(
            "K24:T0",
            1,
            ready=12.2,
            rtf=2.0,
            faults=0,
            tokens=80,
            sha="a" * 64,
            policy="full_prebuffer",
        ),
    ]
    runs = [extract_metrics(item) for item in records]
    comparison = startup_policy_comparison(runs)["comparisons"]["K24:T0"]
    assert comparison["supply_fault_episode_delta"] == -8.0
    assert comparison["first_sentence_ready_delta_s"] == 0.2
    assert comparison["wait_cost_proxy_s"] == 0.2
    assert comparison["text_identical_all"] is True


def _answers_with_delta(delta: float) -> dict:
    aggregates = {
        "K24:T0|first_sentence": {
            "placement": "K24:T0",
            "startup_policy": "first_sentence",
            "first_sentence_ready_s": {"mean": 12.0},
        },
        "K24:T3|first_sentence": {
            "placement": "K24:T3",
            "startup_policy": "first_sentence",
            "first_sentence_ready_s": {"mean": 12.0 + delta},
        },
    }
    pairs = {
        "K24:T0->K24:T3": {
            "paired_run_count": 5,
            "both_passed_run_count": 5,
            "text_identical_all": True,
            "first_sentence_ready_s": {"mean_delta": delta},
            "rtf_mean": {"mean_delta": 0.0},
            "supply_fault_episodes": {"mean_delta": 0.0},
        }
    }
    return answer_questions(aggregates, pairs, {"per_placement": {}})


def test_worsening_delta_is_never_a_gain() -> None:
    # 主指标越小越好：目标更慢（+2 s）不是收益，不能进候选
    answers = _answers_with_delta(2.0)
    assert answers["q1_hybrid_vs_feasible_baseline"]["per_target"]["K24:T3"]["gains"][
        "first_sentence_ready_s"
    ] is False
    assert answers["q3_candidates_for_dynamic_switching"]["candidates"] == []


def test_real_improvement_becomes_candidate() -> None:
    answers = _answers_with_delta(-2.0)
    assert answers["q1_hybrid_vs_feasible_baseline"]["per_target"]["K24:T3"]["gains"][
        "first_sentence_ready_s"
    ] is True
    candidates = answers["q3_candidates_for_dynamic_switching"]["candidates"]
    assert [item["placement"] for item in candidates] == ["K24:T3"]


def test_each_metric_uses_its_own_baseline_tolerance() -> None:
    # 缺口段数基线 8 → 容差 0.8；改善 1 段必须被判为改善，
    # 不能拿首句就绪（12 s）当分母而漏判
    aggregates = {
        "K24:T0|first_sentence": {
            "placement": "K24:T0",
            "startup_policy": "first_sentence",
            "first_sentence_ready_s": {"mean": 12.0},
            "rtf_mean": {"mean": 1.9},
            "supply_fault_episodes": {"mean": 8.0},
        },
        "K0:T0|first_sentence": {
            "placement": "K0:T0",
            "startup_policy": "first_sentence",
            "first_sentence_ready_s": {"mean": 18.0},
            "rtf_mean": {"mean": 2.1},
            "supply_fault_episodes": {"mean": 7.0},
        },
    }
    pairs = {
        "K24:T0->K0:T0": {
            "paired_run_count": 5,
            "both_passed_run_count": 5,
            "text_identical_all": False,
            "first_sentence_ready_s": {"mean_delta": 6.0},
            "rtf_mean": {"mean_delta": 0.2},
            "supply_fault_episodes": {"mean_delta": -1.0},
            "sentence_count": {"mean_delta": -1.0},
            "output_tokens": {"mean_delta": -15.0},
        }
    }
    answers = answer_questions(aggregates, pairs, {"per_placement": {}})
    target = answers["q1_hybrid_vs_feasible_baseline"]["per_target"]["K0:T0"]
    assert target["gains"] == {
        "first_sentence_ready_s": False,
        "rtf_mean": False,
        "supply_fault_episodes": True,
    }
    # 候选必须带上"分句数/文本不同"的混入标记，不能只报收益
    candidate = answers["q3_candidates_for_dynamic_switching"]["candidates"][0]
    assert candidate["placement"] == "K0:T0"
    assert "output_text_differs" in candidate["confound_flags"]
    assert "sentence_count_differs" in candidate["confound_flags"]
    assert candidate["context_deltas"]["sentence_count"]["mean_delta"] == -1.0


def test_protocol_guard_rejects_wrong_version(tmp_path) -> None:
    _write_run_dir(tmp_path, _records())
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    manifest["protocol_version"] = "s0_protocol_v0"
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="协议版本不一致"):
        analyze(tmp_path)
