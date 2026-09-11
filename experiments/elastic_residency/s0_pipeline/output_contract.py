"""S0 输出契约：思考正文、空输出与格式残留的判定与证据落盘。

原则（指南第 07 模块 E2）：

- **不静默删除**：思考正文、纯格式片段与空输出都只判定失败并保留
  原文、处理后文本、句子映射和失败原因，不按关键词删除任何文本。
- **确定性**：判据是固定正则与阈值，不依赖模型名称、不依赖随机性。
- 阈值仍为 pilot 初始值，正式冻结前须经用户确认，状态写进产物。
"""

from __future__ import annotations

import re
from typing import Any

from experiments.elastic_residency.s0_pipeline.text_contract import (
    StreamingTextNormalizer,
    map_sentences_to_raw,
)

#: 输出契约阈值；pilot 初始值，正式冻结前须经用户确认。
DROPPED_CHAR_RATIO_MAX = 0.3
RESIDUAL_MARKER_MAX = 0

_RESIDUAL_MARKER = re.compile(r"^\s*(?:[-*+]\s+|\d{1,3}[.)]\s+|#{1,6}\s+|\*\*)")

#: 思考正文判据：只用于判定失败并保留证据，不触发任何删除。
_THINKING_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("think_tag", re.compile(r"<\s*/?\s*think\s*>", re.IGNORECASE)),
    (
        "reasoning_heading",
        re.compile(
            r"^\W{0,4}(?:thinking process|thought process|reasoning process"
            r"|chain[- ]of[- ]thought|step[- ]by[- ]step (?:analysis|reasoning)"
            r"|思考过程|分析过程|推理过程|分析步骤|思路)\s*[:：]",
            re.IGNORECASE,
        ),
    ),
    (
        "plan_step_heading",
        re.compile(
            r"^\W{0,4}\d{1,2}[.)]\s*\**\s*"
            r"(?:analyze|analysis|identify|draft|outline|plan)\b",
            re.IGNORECASE,
        ),
    ),
)


def detect_thinking_body(raw_text: str) -> dict[str, Any] | None:
    """检测思考正文；只返回判据，不做任何删除。"""

    for code, pattern in _THINKING_RULES:
        match = pattern.search(raw_text)
        if match is not None:
            return {"code": code, "matched": match.group(0)[:120]}
    return None


def assess_output_contract(
    *,
    normalizer_stats: dict[str, Any],
    sentences: list[str],
) -> dict[str, Any]:
    """输出契约检查：判定失败并保留全部证据，不删除任何内容。"""

    raw_text = str(normalizer_stats.get("raw_text", ""))
    processed_text = str(normalizer_stats.get("processed_text", ""))
    raw_chars = int(normalizer_stats.get("raw_chars", 0) or len(raw_text))
    dropped = int(normalizer_stats.get("dropped_chars", 0))
    ratio = dropped / raw_chars if raw_chars > 0 else 0.0
    residual_markers = sum(1 for text in sentences if _RESIDUAL_MARKER.match(text))
    thinking = detect_thinking_body(raw_text)

    failures: list[dict[str, Any]] = []
    if raw_chars == 0:
        failures.append(
            {"code": "empty_output", "detail": "模型未产生任何文本，无证据可判"}
        )
    else:
        if thinking is not None:
            failures.append(
                {
                    "code": "thinking_body_detected",
                    "pattern": thinking["code"],
                    "matched": thinking["matched"],
                    "detail": "保留原文并判失败，不做关键词静默删除",
                }
            )
        if not sentences:
            failures.append({"code": "no_sentence_emitted", "detail": "无可合成句"})
        if ratio > DROPPED_CHAR_RATIO_MAX:
            failures.append(
                {
                    "code": "dropped_char_ratio_exceeded",
                    "value": round(ratio, 4),
                    "max": DROPPED_CHAR_RATIO_MAX,
                }
            )
        if residual_markers > RESIDUAL_MARKER_MAX:
            failures.append(
                {
                    "code": "residual_marker_sentences",
                    "value": residual_markers,
                    "max": RESIDUAL_MARKER_MAX,
                }
            )

    return {
        "raw_text": raw_text,
        "processed_text": processed_text,
        "raw_chars": raw_chars,
        "processed_chars": int(
            normalizer_stats.get("processed_chars", len(processed_text))
        ),
        "dropped_chars": dropped,
        "dropped_char_ratio": round(ratio, 4),
        "residual_marker_sentences": residual_markers,
        "thinking_detected": thinking,
        "empty_output": raw_chars == 0,
        "change_count": int(normalizer_stats.get("change_count", 0)),
        "changes": list(normalizer_stats.get("changes", [])),
        "failure_reasons": failures,
        "failed": bool(failures),
        "threshold": {
            "dropped_char_ratio_max": DROPPED_CHAR_RATIO_MAX,
            "residual_markers_max": RESIDUAL_MARKER_MAX,
            "status": "pilot_initial_pending_freeze",
        },
    }


def evaluate_text_contract(
    *,
    normalizer: StreamingTextNormalizer,
    sentences: list[str],
) -> dict[str, Any]:
    """一次运行使用的完整契约证据：判据 + 原文 + 处理后文本 + 句子映射。"""

    contract = assess_output_contract(
        normalizer_stats=normalizer.stats(), sentences=sentences
    )
    contract["sentence_mapping"] = map_sentences_to_raw(normalizer, sentences)
    return contract
