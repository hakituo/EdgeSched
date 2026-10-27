"""V0 单元定义、核划分与汇总的纯逻辑测试（不加载模型）。"""

from __future__ import annotations

import pytest

from experiments.elastic_residency.colocation.v0_cells import (
    CELLS,
    Cell,
    decode_rate,
    select_cells,
    split_cores,
    summarize,
    tts_rtf_during,
)

# 本机拓扑：逻辑 CPU i 与 i+16 是同一物理核
GROUPS = [[core, core + 16] for core in range(16)]


def test_split_cores_disjoint_and_smt_complete():
    llm, tts = split_cores(GROUPS, 14, 2)
    assert llm == sorted(list(range(14)) + list(range(16, 30)))
    assert tts == [14, 15, 30, 31]
    assert not set(llm) & set(tts)


def test_split_cores_rejects_oversubscription():
    with pytest.raises(ValueError):
        split_cores(GROUPS, 15, 2)


def test_cells_unique_and_selectable():
    names = [cell.name for cell in CELLS]
    assert len(names) == len(set(names))
    assert [c.name for c in select_cells(["alone", "tts_thread"])] == ["alone", "tts_thread"]
    with pytest.raises(ValueError):
        select_cells(["nope"])


def test_cell_validation():
    with pytest.raises(ValueError):
        Cell("bad", True, "magic", None, None, "")
    with pytest.raises(ValueError):
        Cell("bad", False, "none", None, None, "")


def test_decode_rate_and_rtf():
    assert decode_rate([0, 250_000_000, 500_000_000]) == 4.0
    assert decode_rate([1]) is None
    records = [
        {"start_ns": 0, "end_ns": 1_000_000_000, "audio_s": 4.0},
        {"start_ns": 5_000_000_000, "end_ns": 6_000_000_000, "audio_s": 2.0},
    ]
    assert tts_rtf_during(records) == pytest.approx(2.0 / 6.0)
    assert tts_rtf_during(records, start_ns=4_000_000_000) == pytest.approx(0.5)
    assert tts_rtf_during([], 0, 1) is None


def test_summarize_slowdown_against_alone():
    def result(name, rates, rtf=None, passed=True):
        return {
            "cell": {"name": name},
            "passed": passed,
            "reps": [{"decode_tok_s": r, "ttft_s": 0.3, "tts_rtf": rtf} for r in rates],
        }

    rows = summarize([
        result("alone", [14.0, 13.0, 15.0]),
        result("tts_thread", [3.5, 3.4, 3.6], rtf=0.4),
        {"cell": {"name": "tts_alone"}, "passed": True, "reps": [], "tts_alone_rtf": 0.33},
    ])
    assert rows["alone"]["slowdown_vs_alone_x"] == 1.0
    assert rows["tts_thread"]["slowdown_vs_alone_x"] == 4.0
    assert rows["tts_thread"]["tts_rtf"] == 0.4
    assert rows["tts_alone"]["tts_rtf"] == 0.33
    assert rows["tts_alone"]["decode_tok_s"] is None


def test_summarize_pools_rounds():
    rows = summarize([
        {"cell": {"name": "alone"}, "passed": True, "reps": [{"decode_tok_s": 10.0}]},
        {"cell": {"name": "alone"}, "passed": True, "reps": [{"decode_tok_s": 8.0}, {"decode_tok_s": 9.0}]},
    ])
    assert rows["alone"]["reps"] == 3
    assert rows["alone"]["processes"] == 2
    assert rows["alone"]["decode_tok_s"] == 9.0
    assert (rows["alone"]["decode_tok_s_min"], rows["alone"]["decode_tok_s_max"]) == (8.0, 10.0)
