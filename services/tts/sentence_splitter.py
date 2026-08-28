"""面向句子级流式 TTS 的确定性增量断句器。"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SentenceChunk:
    """断句结果，同时保留原文、合成文本和原文绝对偏移。"""

    raw_text: str
    synthesis_text: str
    start_char: int
    end_char: int


class DeterministicSentenceSplitter:
    """增量过滤内部标签，并按固定规则生成完整句子。

    ``<think>...</think>`` 的标签及内容会被删除；``[GEN_IMG:...]``
    只删除控制标签本身。两类标签均允许跨 token 到达。
    """

    _STRONG_BOUNDARIES = frozenset("。！？!?；;\n")
    _WEAK_BOUNDARIES = frozenset("，,：:")
    _ABBREVIATIONS = frozenset(
        {
            "mr.",
            "mrs.",
            "ms.",
            "dr.",
            "prof.",
            "sr.",
            "jr.",
            "vs.",
            "etc.",
            "e.g.",
            "i.e.",
        }
    )

    def __init__(self, *, min_chars: int = 12, max_chars: int = 80) -> None:
        if min_chars < 1:
            raise ValueError("min_chars 必须大于 0")
        if max_chars < min_chars:
            raise ValueError("max_chars 不能小于 min_chars")
        self.min_chars = min_chars
        self.max_chars = max_chars
        self._raw_text = ""
        self._raw_emit_start = 0
        self._visible_chars: list[str] = []
        self._visible_raw_ends: list[int] = []
        self._angle_buffer: list[tuple[str, int]] = []
        self._square_buffer: list[tuple[str, int]] = []
        self._think_depth = 0
        self._finished = False

    def feed(self, token: str) -> list[SentenceChunk]:
        """接收一个 token，并返回本次新形成的句子。"""

        if self._finished:
            raise RuntimeError("finish() 后不能继续 feed()")
        for char in token:
            self._raw_text += char
            self._consume_char(char, len(self._raw_text))
        return self._extract_available(final=False)

    def finish(self) -> list[SentenceChunk]:
        """结束输入并输出剩余可见文本。"""

        if self._finished:
            return []
        self._finished = True

        if self._angle_buffer and self._think_depth == 0:
            self._flush_pending(self._angle_buffer)
        self._angle_buffer.clear()

        if self._square_buffer:
            pending = "".join(char for char, _ in self._square_buffer)
            if not "[GEN_IMG:".startswith(pending.upper()):
                self._flush_pending(self._square_buffer)
        self._square_buffer.clear()
        return self._extract_available(final=True)

    def _consume_char(self, char: str, raw_end: int) -> None:
        if self._angle_buffer:
            self._angle_buffer.append((char, raw_end))
            if char == ">":
                self._resolve_angle_buffer()
            elif len(self._angle_buffer) > 32:
                if self._think_depth == 0:
                    self._flush_pending(self._angle_buffer)
                self._angle_buffer.clear()
            return

        if char == "<":
            self._angle_buffer.append((char, raw_end))
            return

        if self._think_depth > 0:
            return

        if self._square_buffer:
            self._square_buffer.append((char, raw_end))
            pending = "".join(item[0] for item in self._square_buffer)
            upper = pending.upper()
            if char == "]" and upper.startswith("[GEN_IMG:"):
                self._square_buffer.clear()
            elif not "[GEN_IMG:".startswith(upper) and not upper.startswith("[GEN_IMG:"):
                self._flush_pending(self._square_buffer)
                self._square_buffer.clear()
            return

        if char == "[":
            self._square_buffer.append((char, raw_end))
            return

        self._append_visible(char, raw_end)

    def _resolve_angle_buffer(self) -> None:
        text = "".join(char for char, _ in self._angle_buffer).strip().lower()
        if text == "<think>":
            self._think_depth += 1
        elif text == "</think>":
            self._think_depth = max(0, self._think_depth - 1)
        elif self._think_depth == 0:
            self._flush_pending(self._angle_buffer)
        self._angle_buffer.clear()

    def _flush_pending(self, pending: list[tuple[str, int]]) -> None:
        for char, raw_end in pending:
            self._append_visible(char, raw_end)

    def _append_visible(self, char: str, raw_end: int) -> None:
        self._visible_chars.append(char)
        self._visible_raw_ends.append(raw_end)

    def _extract_available(self, *, final: bool) -> list[SentenceChunk]:
        chunks: list[SentenceChunk] = []
        while self._visible_chars:
            cut = self._find_cut(final=final)
            if cut is None:
                break
            chunks.append(self._emit(cut))
        return chunks

    def _find_cut(self, *, final: bool) -> int | None:
        text = "".join(self._visible_chars)
        for index, char in enumerate(text):
            length = index + 1
            if char in self._STRONG_BOUNDARIES:
                return length
            if char == "." and self._period_is_boundary(text, index, final=final):
                return length
            if char in self._WEAK_BOUNDARIES and length >= self.min_chars:
                return length

        if len(text) >= self.max_chars:
            search_start = max(self.min_chars - 1, 0)
            for index in range(self.max_chars - 1, search_start - 1, -1):
                if text[index] in self._WEAK_BOUNDARIES or text[index].isspace():
                    return index + 1
            return self.max_chars

        if final and text.strip():
            return len(text)
        if final:
            self._discard_visible(len(text))
        return None

    def _period_is_boundary(self, text: str, index: int, *, final: bool) -> bool:
        if index + 1 == len(text) and not final:
            return False
        previous = text[index - 1] if index > 0 else ""
        following = text[index + 1] if index + 1 < len(text) else ""
        if previous.isdigit() and following.isdigit():
            return False
        if following == "." or previous == ".":
            return False

        prefix = text[: index + 1].lower()
        word_match = re.search(r"(?:[a-z]\.){1,4}$|[a-z]+\.$", prefix)
        if word_match and word_match.group(0) in self._ABBREVIATIONS:
            return False
        if re.search(r"(?:[A-Za-z]\.){2,}$", text[: index + 1]):
            return False
        return True

    def _emit(self, visible_count: int) -> SentenceChunk:
        raw_end = self._visible_raw_ends[visible_count - 1]
        raw_text = self._raw_text[self._raw_emit_start : raw_end]
        synthesis_text = "".join(self._visible_chars[:visible_count]).strip()
        chunk = SentenceChunk(
            raw_text=raw_text,
            synthesis_text=synthesis_text,
            start_char=self._raw_emit_start,
            end_char=raw_end,
        )
        self._raw_emit_start = raw_end
        self._discard_visible(visible_count)
        return chunk

    def _discard_visible(self, count: int) -> None:
        del self._visible_chars[:count]
        del self._visible_raw_ends[:count]
