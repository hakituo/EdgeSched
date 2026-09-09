"""输入适配层与输出契约测试：格式安全、数字正文保护、失败显式上报。"""

from experiments.elastic_residency.s0_pipeline.runner import PipelineRunner  # noqa: F401
from experiments.elastic_residency.s0_pipeline.text_contract import (
    StreamingTextNormalizer,
    assess_output_contract,
)


def _feed_all(text: str) -> tuple[str, StreamingTextNormalizer]:
    normalizer = StreamingTextNormalizer()
    out = normalizer.feed(text) + normalizer.finish()
    return out, normalizer


def test_line_start_list_marker_is_removed() -> None:
    out, normalizer = _feed_all("\n1. Topic\n- Item\n* Star\n# Heading\n")
    assert "Topic" in out and "Item" in out and "Star" in out and "Heading" in out
    assert "1. Topic" not in out and "- Item" not in out
    reasons = {change["reason"] for change in normalizer.changes}
    assert "list_marker_removed" in reasons


def test_numeric_body_is_not_removed() -> None:
    out, _ = _feed_all("延迟降低到 3.5 秒。\n版本 1.2 发布了。")
    assert "3.5 秒" in out
    assert "版本 1.2 发布了" in out


def test_emphasis_markers_are_removed_but_body_kept() -> None:
    out, normalizer = _feed_all("这是**重点**内容。")
    assert out == "这是重点内容。"
    assert any(
        change["reason"] == "markdown_emphasis_removed"
        for change in normalizer.changes
    )


def test_pure_marker_line_yields_no_sentence_body() -> None:
    out, normalizer = _feed_all("第一句。\n**\n第二句。")
    assert "第一句。" in out and "第二句。" in out
    assert normalizer.dropped_chars > 0


def test_newlines_are_preserved_for_sentence_splitting() -> None:
    out, _ = _feed_all("第一句。\n第二句。")
    assert "\n" in out


def test_char_by_char_feeding_matches_whole_text() -> None:
    text = "\n1. Topic\n这是**重点**，数字 3.5 保留。\n"
    whole, _ = _feed_all(text)
    charwise = StreamingTextNormalizer()
    pieces = [charwise.feed(ch) for ch in text]
    pieces.append(charwise.finish())
    assert "".join(pieces) == whole


def test_contract_passes_on_clean_answer() -> None:
    normalizer = StreamingTextNormalizer()
    body = "边缘计算把计算放在靠近数据的位置。\n这能降低延迟并节省带宽。\n"
    normalizer.feed(body)
    normalizer.finish()
    contract = assess_output_contract(
        normalizer_stats=normalizer.stats(),
        sentences=["边缘计算把计算放在靠近数据的位置。", "这能降低延迟并节省带宽。"],
    )
    assert contract["failed"] is False


def test_contract_fails_when_model_keeps_emitting_markup() -> None:
    normalizer = StreamingTextNormalizer()
    body = "**Topic:**\n* a\n* b\n* c\n" * 6
    normalizer.feed(body)
    normalizer.finish()
    contract = assess_output_contract(
        normalizer_stats=normalizer.stats(),
        sentences=["Topic:"] * 20,
    )
    assert contract["failed"] is True
    # 适配层已把标记删干净，残留句为 0；契约失败由丢弃比例触发
    assert contract["residual_marker_sentences"] == 0
    assert contract["dropped_char_ratio"] > 0.3
    assert contract["threshold"]["status"] == "pilot_initial_pending_freeze"


def test_contract_failure_keeps_evidence_not_silence() -> None:
    normalizer = StreamingTextNormalizer()
    normalizer.feed("**分析**\n* 步骤一\n")
    normalizer.finish()
    stats = normalizer.stats()
    assert stats["change_count"] > 0
    assert all(change["raw"] for change in stats["changes"])
