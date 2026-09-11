"""S0 输入适配层：流式 Markdown 规范化与句子回溯映射。

分层边界（对应指南第 07 模块的修正决定）：

- Prompt 管正文语义：要求直接回答、不输出分析、不使用 Markdown；
  模型仍泄漏思考正文时由 ``output_contract`` 判失败并保留原文，
  不做关键词静默删除。
- 本模块只管格式安全：确定性去除纯格式片段，保留原文、处理后文本与
  逐字符回溯下标；规则放在工作负载适配层，不写入通用调度器，
  也不按模型名称或文本内容分支。
- ``1.`` 必须结合后文判断：仅当行首出现 ``数字+.+空白`` 或整行只剩
  该标记时按列表标记处理；``3.5``、``版本 1.2`` 等数字正文不受影响。
"""

from __future__ import annotations

import re
from typing import Any

_MARKER = re.compile(r"^(?:[-*+]|\d{1,3}[.)]|#{1,6})[ \t]+$")
_MARKER_TIGHT = re.compile(r"^(?:[-*+]|\d{1,3}[.)]|#{1,6})$")
_PURE_FORMAT = re.compile(r"[*_`~#>\-]+")
_MARKUP_CHARS = frozenset("0123456789.*-+#) \t")
_MAX_LINE_HEAD_BUF = 16


class StreamingTextNormalizer:
    """字符级流式规范化器；所有状态在实例内，跨 token 边界安全。

    同时维护 ``processed_to_raw``：每个输出字符都能回溯到原文下标，
    因此句子映射与丢弃原因都可反向定位，而不是只留字符计数。
    """

    def __init__(self) -> None:
        self.changes: list[dict[str, Any]] = []
        self.raw_parts: list[str] = []
        self.processed_parts: list[str] = []
        self.processed_to_raw: list[int] = []
        self.dropped_chars = 0
        self._at_line_start = True
        self._line_head: list[tuple[str, int]] = []
        self._drop_run: list[tuple[str, int]] = []
        self._last_space = False

    @property
    def raw_text(self) -> str:
        return "".join(self.raw_parts)

    @property
    def processed_text(self) -> str:
        return "".join(self.processed_parts)

    def feed(self, token: str) -> str:
        pieces: list[str] = []
        for char in token:
            raw_index = len(self.raw_parts)
            self.raw_parts.append(char)
            pieces.append(self._consume(char, raw_index))
        return "".join(pieces)

    def finish(self) -> str:
        return self._decide_line_head() + self._flush_drop_run()

    def stats(self) -> dict[str, Any]:
        raw = self.raw_text
        return {
            "raw_text": raw,
            "processed_text": self.processed_text,
            "raw_chars": len(raw),
            "processed_chars": len(self.processed_parts),
            "dropped_chars": self.dropped_chars,
            "change_count": len(self.changes),
            "changes": list(self.changes),
        }

    def _consume(self, char: str, raw_index: int) -> str:
        if self._at_line_start:
            if char == "\n":
                out = self._resolve_line_head(line_end=True)
                self._at_line_start = True
                self._line_head = []
                self._last_space = False
                return out + self._emit(char, raw_index)
            if char in _MARKUP_CHARS and len(self._line_head) < _MAX_LINE_HEAD_BUF:
                self._line_head.append((char, raw_index))
                return ""
            if len(self._line_head) >= _MAX_LINE_HEAD_BUF:
                buffered = self._resolve_line_head(line_end=False)
                self._at_line_start = False
                return buffered + self._consume(char, raw_index)
            self._line_head.append((char, raw_index))
            return self._decide_line_head()
        if char == "\n":
            self._at_line_start = True
            self._line_head = []
            self._last_space = False
            return self._flush_drop_run() + self._emit(char, raw_index)
        if char in "*_`":
            self._drop_run.append((char, raw_index))
            return ""
        if char in " \t":
            if self._last_space:
                self.dropped_chars += 1
                return ""
            self._last_space = True
            return self._emit(char, raw_index)
        self._last_space = False
        return self._flush_drop_run() + self._emit(char, raw_index)

    def _emit(self, char: str, raw_index: int) -> str:
        self.processed_parts.append(char)
        self.processed_to_raw.append(raw_index)
        return char

    def _resolve_line_head(self, *, line_end: bool) -> str:
        """行首缓冲结算；整行只剩纯标记时丢弃，避免纯格式片段独立成句。"""

        if not self._line_head:
            return ""
        chars = "".join(char for char, _ in self._line_head)
        if line_end:
            stripped = chars.rstrip(" \t")
            if _MARKER.fullmatch(chars) or _MARKER_TIGHT.fullmatch(stripped):
                self._register_drop("list_marker_removed", chars)
                self._line_head = []
                self._at_line_start = False
                return ""
            if _PURE_FORMAT.fullmatch(stripped):
                self._register_drop("markdown_emphasis_removed", chars)
                self._line_head = []
                self._at_line_start = False
                return ""
        return self._flush_line_head()

    def _decide_line_head(self) -> str:
        if not self._line_head:
            return ""
        chars = "".join(char for char, _ in self._line_head)
        candidate = chars[:-1] if chars[-1:] not in _MARKUP_CHARS else chars
        if candidate and _MARKER.fullmatch(candidate):
            self._register_drop("list_marker_removed", candidate)
            self._line_head = self._line_head[len(candidate) :]
            return self._flush_line_head()
        if candidate and _PURE_FORMAT.fullmatch(candidate):
            self._register_drop("markdown_emphasis_removed", candidate)
            self._line_head = self._line_head[len(candidate) :]
            return self._flush_line_head()
        return self._flush_line_head()

    def _flush_line_head(self) -> str:
        buffered = self._line_head
        self._line_head = []
        self._at_line_start = False
        return "".join(self._emit(char, index) for char, index in buffered)

    def _flush_drop_run(self) -> str:
        if not self._drop_run:
            return ""
        chars = "".join(char for char, _ in self._drop_run)
        self._drop_run = []
        self._register_drop("markdown_emphasis_removed", chars)
        return ""

    def _register_drop(self, reason: str, raw: str) -> None:
        self.dropped_chars += len(raw)
        self.changes.append({"reason": reason, "raw": raw, "processed": ""})


def map_sentences_to_raw(
    normalizer: StreamingTextNormalizer,
    sentence_texts: list[str],
) -> list[dict[str, Any]]:
    """把每句（规范化后文本）映射回原文字符区间，保序且允许显式失败。"""

    processed = normalizer.processed_text
    index_map = normalizer.processed_to_raw
    cursor = 0
    mapping: list[dict[str, Any]] = []
    for chunk_id, text in enumerate(sentence_texts):
        if not text:
            mapping.append(
                {
                    "chunk_id": chunk_id,
                    "mapped": False,
                    "reason": "empty_normalized_text",
                }
            )
            continue
        start = processed.find(text, cursor)
        if start < 0:
            start = processed.find(text)
        if start < 0:
            mapping.append(
                {
                    "chunk_id": chunk_id,
                    "mapped": False,
                    "reason": "not_found_in_processed_text",
                    "text": text,
                }
            )
            continue
        end = start + len(text)
        mapping.append(
            {
                "chunk_id": chunk_id,
                "mapped": True,
                "text": text,
                "processed_start": start,
                "processed_end": end,
                "raw_start": index_map[start],
                "raw_end": index_map[end - 1] + 1,
            }
        )
        cursor = end
    return mapping
