"""B2 波形正确性核对器的静态验证：文本判定、状态残留与 summary 回填。"""

import json
from pathlib import Path

import pytest

from experiments.tts import check_tts_output_correctness as checker

_EXPECTED = "别着急，先喝口水，再继续处理后面的任务。"


def test_punctuation_difference_is_not_an_error() -> None:
    result = checker.compare(_EXPECTED, "别着急先喝口水再继续处理后面的任务")
    assert result["verdict"] == "ok"
    assert result["char_ratio"] == 1.0


def test_empty_truncated_repeat_and_mismatch_are_distinguished() -> None:
    assert checker.compare(_EXPECTED, "   ")["verdict"] == "empty"
    assert checker.compare(_EXPECTED, "别着急先")["verdict"] == "truncated"
    assert checker.compare(_EXPECTED, "别着急先喝口水再继续处理后面的任务" * 5)["verdict"] == "repeat"
    assert checker.compare(_EXPECTED, "完全不同的内容在这里出现请再检查一下")["verdict"] == "mismatch"


def test_homophone_noise_is_partial_not_a_hard_failure() -> None:
    # 真实样本：small-int8 把「尾延迟」听成「伪延迟」，比例 0.87 属 partial
    result = checker.compare(
        "每个测试句子都会保存独立音频和结构化指标，后续可以从原始结果"
        "重新计算平均值、尾延迟和峰值显存。复核。",
        "每个测试句子都会保存独立音频和结构画指标后续可以从原始结果"
        "重新计算平均值、伪言值和风值显存复合",
    )
    assert result["verdict"] == "partial"
    assert "partial" not in checker.HARD_FAILURES


def test_traditional_script_is_not_silently_accepted() -> None:
    # 简繁差异仍是硬失败，必须靠 ASR 简体偏置解决，而不是放宽阈值
    result = checker.compare(_EXPECTED, "別著急,先喝口水,再繼續處理後面的任務")
    assert result["verdict"] in checker.HARD_FAILURES
    assert result["char_ratio"] < checker.PARTIAL_RATIO


def test_repeated_ngram_detection_needs_a_real_repetition() -> None:
    assert checker.repeated_ngrams("今天天气很好我们要出门散步") == []
    assert checker.repeated_ngrams("测试测试测试测试测试") == ["测试测试"]


def test_normalize_keeps_cjk_and_alnum_only() -> None:
    assert checker.normalize_text("Ab，1。 ") == "ab1"


def test_asr_contract_pins_simplified_bias() -> None:
    args = checker.build_parser().parse_args(["--run-dir", "."])
    assert args.initial_prompt == checker.DEFAULT_INITIAL_PROMPT
    assert "简体" in args.initial_prompt
    assert args.language == "zh"


class _StubAsr:
    """文件名以 a.wav 结尾视为转写正常，b.wav 视为被截断。"""

    def __init__(self, model_path: Path, **kwargs: object) -> None:
        self.input_root = kwargs["input_root"]

    def run(self, relative: str) -> dict:
        text = "别着急先喝口水再继续处理后面的任务" if relative.endswith("a.wav") else "别着急先"
        return {"text": text, "audio_duration_s": 4.7, "segment_count": 1}


def _write_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    audio = run_dir / "audio"
    audio.mkdir(parents=True)
    rows = []
    for index, (name, sha) in enumerate(
        (("a.wav", "same"), ("a.wav", "same"), ("b.wav", "other")), start=1
    ):
        target = audio / f"c{index:02d}_{name}"
        target.write_bytes(b"RIFF")
        rows.append(
            {
                "path": "cuda_graph_static",
                "text": checker.CORRECTNESS_REPEAT,
                "call_index": index,
                "expected_text": _EXPECTED,
                "pcm_sha256": sha,
                "audio_s": 4.7,
                "wav_path": str(target),
            }
        )
    (run_dir / "chunks.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8"
    )
    return run_dir


def test_state_residue_and_failures_are_reported(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(checker, "RoundtripAsr", _StubAsr)
    run_dir = _write_run(tmp_path)
    report = checker.check_run(run_dir, whisper_model=tmp_path / "absent-model", cores=2)
    assert report["call_count"] == 3
    assert report["verdict_counts"] == {"ok": 2, "truncated": 1}
    assert report["all_ok"] is False
    assert [item["verdict"] for item in report["failures"]] == ["truncated"]
    residue = report["state_residue"]["cuda_graph_static|" + checker.CORRECTNESS_REPEAT]
    assert residue["calls"] == 3
    # 波形变体数只作记录；文本层不唯一才算残留
    assert residue["waveform_bitwise_variants"] == 2
    assert residue["text_deterministic"] is False


def test_missing_audio_flag_is_rejected(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(checker, "RoundtripAsr", _StubAsr)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "chunks.jsonl").write_text(
        json.dumps({"path": "p", "text": "chars_20", "call_index": 1}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="wav_path"):
        checker.check_run(run_dir, whisper_model=tmp_path / "m", cores=1)


def test_update_summary_rewrites_pending_flag(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(
        json.dumps({"correctness_check": "pending_external_asr_pass"}), encoding="utf-8"
    )
    checker.update_summary(
        run_dir,
        {
            "all_ok": True,
            "verdict_counts": {"ok": 3},
            "hard_failure_count": 0,
            "partial_count": 0,
            "pass_rule": "stub",
            "checker": "stub",
        },
    )
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["correctness_check"] == "completed_all_ok"
    assert summary["correctness_failure_count"] == 0
    assert summary["passed"] is True
