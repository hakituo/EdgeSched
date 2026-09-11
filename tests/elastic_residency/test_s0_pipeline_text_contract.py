"""输入适配层与输出契约测试：格式安全、数字正文保护、失败显式上报。"""

from experiments.elastic_residency.s0_pipeline.output_contract import (
    assess_output_contract,
    detect_thinking_body,
    evaluate_text_contract,
)
from experiments.elastic_residency.s0_pipeline.runner import PipelineRunner  # noqa: F401
from experiments.elastic_residency.s0_pipeline.text_contract import (
    StreamingTextNormalizer,
    map_sentences_to_raw,
)


def _feed_all(text: str) -> tuple[str, StreamingTextNormalizer]:
    normalizer = StreamingTextNormalizer()
    out = normalizer.feed(text) + normalizer.finish()
    return out, normalizer


def _feed_split(text: str, cut: int) -> tuple[str, StreamingTextNormalizer]:
    normalizer = StreamingTextNormalizer()
    out = normalizer.feed(text[:cut]) + normalizer.feed(text[cut:])
    out += normalizer.finish()
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


def test_any_token_boundary_yields_same_result() -> None:
    text = "\n1. Topic\n这是**重点**，数字 3.5 保留。\n- 项目 1.2 结束"
    whole, _ = _feed_all(text)
    for cut in range(len(text) + 1):
        split_out, _ = _feed_split(text, cut)
        assert split_out == whole, f"切分位置 {cut} 结果不一致"


def test_marker_split_across_tokens_is_still_dropped() -> None:
    normalizer = StreamingTextNormalizer()
    out = normalizer.feed("第一行。\n1")
    out += normalizer.feed(". ")
    out += normalizer.feed("\n第二行。")
    out += normalizer.finish()
    assert "1." not in out
    assert "第一行。" in out and "第二行。" in out


def test_pure_symbol_line_produces_no_body() -> None:
    for marker in ("**\n", "##\n", "---\n", "1.\n", "* \n"):
        out, normalizer = _feed_all(marker)
        assert out.strip() == "", f"{marker!r} 未清理"
        assert normalizer.dropped_chars > 0


def test_decimal_and_version_numbers_are_preserved() -> None:
    out, _ = _feed_all("延迟 3.5 秒，版本 1.2 发布，温度 -5 度，共 12 项。")
    assert out == "延迟 3.5 秒，版本 1.2 发布，温度 -5 度，共 12 项。"


def test_tail_sentence_without_terminator_is_preserved() -> None:
    normalizer = StreamingTextNormalizer()
    out = normalizer.feed("第一句。第二句") + normalizer.finish()
    assert out == "第一句。第二句"


def test_trailing_markup_at_line_start_is_dropped_on_finish() -> None:
    normalizer = StreamingTextNormalizer()
    out = normalizer.feed("第一句。\n**") + normalizer.finish()
    assert out == "第一句。\n"
    assert normalizer.dropped_chars == 2


def test_empty_output_is_reported_not_swallowed() -> None:
    normalizer = StreamingTextNormalizer()
    normalizer.feed("")
    assert normalizer.finish() == ""
    contract = evaluate_text_contract(normalizer=normalizer, sentences=[])
    assert contract["empty_output"] is True
    assert contract["failed"] is True
    assert [item["code"] for item in contract["failure_reasons"]] == ["empty_output"]


def test_thinking_body_is_detected_and_never_deleted() -> None:
    raw = "Thinking Process:\n1. **Analyze the Request:**\n边缘计算降低延迟。"
    out, normalizer = _feed_all(raw)
    contract = evaluate_text_contract(
        normalizer=normalizer,
        sentences=["边缘计算降低延迟。"],
    )
    assert "Thinking Process:" in out, "思考正文被静默删除"
    assert contract["failed"] is True
    codes = [item["code"] for item in contract["failure_reasons"]]
    assert "thinking_body_detected" in codes
    assert contract["raw_text"] == raw
    assert detect_thinking_body(raw) is not None


def test_sentence_mapping_is_ordered_and_traces_raw_offsets() -> None:
    text = "\n1. 第一句话在这里。\n**重点**第二句话在这里。\n"
    out, normalizer = _feed_all(text)
    sentences = ["第一句话在这里。", "第二句话在这里。"]
    mapping = map_sentences_to_raw(normalizer, sentences)
    assert [item["mapped"] for item in mapping] == [True, True]
    starts = [item["raw_start"] for item in mapping]
    assert starts == sorted(starts)
    for item in mapping:
        begin, end = item["raw_start"], item["raw_end"]
        assert text[begin:end].replace("**", "").strip() == item["text"].strip()
        assert out[item["processed_start"] : item["processed_end"]] == item["text"]


def test_unmappable_sentence_is_reported_not_guessed() -> None:
    _, normalizer = _feed_all("第一句话在这里。")
    mapping = map_sentences_to_raw(normalizer, ["原文中不存在的句子"])
    assert mapping[0]["mapped"] is False
    assert mapping[0]["reason"] == "not_found_in_processed_text"
