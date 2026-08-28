from services.tts.sentence_splitter import DeterministicSentenceSplitter


def _collect(splitter: DeterministicSentenceSplitter, tokens: list[str]):
    chunks = []
    for token in tokens:
        chunks.extend(splitter.feed(token))
    chunks.extend(splitter.finish())
    return chunks


def test_cross_token_internal_tags_are_filtered() -> None:
    chunks = _collect(
        DeterministicSentenceSplitter(min_chars=4, max_chars=40),
        ["你好<th", "ink>不应朗读</thi", "nk>世界[GEN_", "IMG:cat]。"],
    )
    assert [chunk.synthesis_text for chunk in chunks] == ["你好世界。"]
    assert "不应朗读" in chunks[0].raw_text
    assert chunks[0].start_char == 0
    assert chunks[0].end_char > 0


def test_decimal_abbreviation_and_ellipsis_do_not_split_early() -> None:
    chunks = _collect(
        DeterministicSentenceSplitter(min_chars=6, max_chars=80),
        ["版本是 3.", "14，Dr. Wang 说...", "可以。下一句"],
    )
    assert [chunk.synthesis_text for chunk in chunks] == [
        "版本是 3.14，",
        "Dr. Wang 说...可以。",
        "下一句",
    ]


def test_max_chars_uses_deterministic_weak_break() -> None:
    chunks = _collect(
        DeterministicSentenceSplitter(min_chars=5, max_chars=12),
        ["一二三四五六 七八九十十一十二十三"],
    )
    assert chunks[0].synthesis_text == "一二三四五六"
    assert "".join(chunk.synthesis_text for chunk in chunks).replace(" ", "") == (
        "一二三四五六七八九十十一十二十三"
    )
