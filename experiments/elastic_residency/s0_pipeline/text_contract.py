"""S0 输入适配层：流式 Markdown 规范化与输出契约检查。

分层边界（对应指南第 07 模块的修正决定）：

- Prompt 管正文语义：要求直接回答、不输出分析、不使用 Markdown；
  若模型仍输出思考正文，由契约检查显式判失败，**不做关键词静默删除**。
- 本模块只管格式安全：确定性去除纯格式片段，保留原文、处理后文本
  与丢弃原因，供回溯审计；规则不写入通用调度器。
- ``1.`` 必须结合后文判断：仅当行首出现 ``数字+.+空白`` 时按列表标记
  处理；``3.5``、``版本 1.2`` 等数字正文不受影响。
"""

from __future__ import annotations

import re
from typing import Any

_MARKER = re.compile(r"^(?:[-*+]|\d{1,3}[.)]|#{1,6})[ \t]$")
_PURE_FORMAT = re.compile(r"[*_`~#>]+")
_MARKUP_CHARS = frozenset("0123456789.*-+#) \t")
_MAX_LINE_HEAD_BUF = 16


class StreamingTextNormalizer:
    """字符级流式规范化器；所有状态在实例内，跨 token 安全。"""

    def __init__(self) -> None:
        self.changes: list[dict[str, Any]] = []
        self.raw_parts: list[str] = []
        self.dropped_chars = 0
        self._at_line_start = True
        self._line_head_buf: list[str] = []
        self._drop_run = ""
        self._last_space = False

    @property
    def raw_text(self) -> str:
        return "".join(self.raw_parts)

    def feed(self, token: str) -> str:
        self.raw_parts.append(token)
        return "".join(self._consume(ch) for ch in token)

    def finish(self) -> str:
        out = self._decide_line_head(flush=True) or ""
        out += self._flush_drop_run()
        return out

    def stats(self) -> dict[str, Any]:
        return {
            "raw_chars": len(self.raw_text),
            "dropped_chars": self.dropped_chars,
            "change_count": len(self.changes),
            "changes": list(self.changes),
        }

    def _consume(self, ch: str) -> str:
        if self._at_line_start:
            if ch == "\n":
                # 行首缓冲在换行处结算：纯格式标记丢弃并记录，
                # 其余（数字正文、连字符等）原样保留给分句器
                out = ""
                if self._line_head_buf:
                    buf = "".join(self._line_head_buf)
                    if _MARKER.fullmatch(buf):
                        self._register_drop("list_marker_removed", buf)
                    elif _PURE_FORMAT.fullmatch(buf):
                        self._register_drop("markdown_emphasis_removed", buf)
                    else:
                        out = buf
                    self._line_head_buf = []
                return out + "\n"
            if ch in _MARKUP_CHARS and len(self._line_head_buf) < _MAX_LINE_HEAD_BUF:
                self._line_head_buf.append(ch)
                return ""
            if len(self._line_head_buf) >= _MAX_LINE_HEAD_BUF:
                buffered = "".join(self._line_head_buf)
                self._line_head_buf = []
                self._at_line_start = False
                return buffered + self._consume(ch)
            self._line_head_buf.append(ch)
            return self._decide_line_head(flush=True)
        if ch == "\n":
            self._at_line_start = True
            self._line_head_buf = []
            self._last_space = False
            return self._flush_drop_run() + "\n"
        if ch in "*_`":
            self._drop_run += ch
            return ""
        if ch in " \t":
            if self._last_space:
                self.dropped_chars += 1
                return ""
            self._last_space = True
            return " "
        self._last_space = False
        return self._flush_drop_run() + ch

    def _decide_line_head(self, *, flush: bool) -> str:
        buf = "".join(self._line_head_buf)
        if not flush:
            if _MARKER.fullmatch(buf):
                self._register_drop("list_marker_removed", buf)
                self._line_head_buf = []
                self._at_line_start = False
                self._last_space = True
                return ""
            return ""
        if not self._line_head_buf:
            return ""
        candidate = buf[:-1] if buf[-1:] not in _MARKUP_CHARS else buf
        if candidate and _MARKER.fullmatch(candidate):
            self._register_drop("list_marker_removed", candidate)
            rest = buf[len(candidate) :]
            self._line_head_buf = []
            self._at_line_start = False
            return rest
        if candidate and _PURE_FORMAT.fullmatch(candidate):
            self._register_drop("markdown_emphasis_removed", candidate)
            rest = buf[len(candidate) :]
            self._line_head_buf = []
            self._at_line_start = False
            return rest
        self._line_head_buf = []
        self._at_line_start = False
        return buf

    def _flush_drop_run(self) -> str:
        if not self._drop_run:
            return ""
        run = self._drop_run
        self._drop_run = ""
        self._register_drop("markdown_emphasis_removed", run)
        return ""

    def _register_drop(self, reason: str, raw: str) -> None:
        self.dropped_chars += len(raw)
        self.changes.append(
            {"reason": reason, "raw": raw, "processed": ""}
        )


def assess_output_contract(
    *,
    normalizer_stats: dict[str, Any],
    sentences: list[str],
) -> dict[str, Any]:
    """输出契约检查：判定失败并保留证据，不删除任何内容。

    阈值 0.3 为 pilot 后的初始值，正式冻结前须经用户确认；
    契约失败只影响门禁判定，不改写或丢弃任何文本。
    """

    raw_chars = int(normalizer_stats.get("raw_chars", 0))
    dropped = int(normalizer_stats.get("dropped_chars", 0))
    ratio = dropped / raw_chars if raw_chars > 0 else 0.0
    residual_markers = sum(
        1
        for text in sentences
        if re.match(r"^\s*(?:[-*+]\s+|\d{1,3}[.)]\s+|#{1,6}\s+|\*\*)", text)
    )
    failed = raw_chars > 0 and (ratio > 0.3 or residual_markers > 0)
    return {
        "raw_chars": raw_chars,
        "dropped_chars": dropped,
        "dropped_char_ratio": round(ratio, 4),
        "residual_marker_sentences": residual_markers,
        "failed": failed,
        "threshold": {
            "dropped_char_ratio_max": 0.3,
            "residual_markers_max": 0,
            "status": "pilot_initial_pending_freeze",
        },
    }
