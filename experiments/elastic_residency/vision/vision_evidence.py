"""视觉端点测量证据语义、驻留状态与验收门禁（纯计算，可离线单测）。

状态定义（互斥，写入每条 iteration）：

- ``TEXT_ONLY``：只加载语言模型，projector 未加载；prompt 为固定 token 数填充文本，
  与 C1/C2 的纯文本口径一致；
- ``TEXT_ONLY_CHAT``：只加载语言模型，但使用与视觉轮次相同的对话 prompt（不含图片），
  作为视觉轮次的同 prompt 对照；
- ``VISION_WARM_GPU`` / ``VISION_WARM_CPU``：projector 已 warm 并对固定真实图片
  实际执行，区别只在 projector 所在设备。

禁止事项：projector 未实际执行、视觉 token 为 0、图片解码失败或
``mtmd_encode`` 未返回嵌入的轮次一律判失败；不得把 mmproj 文件大小
当作运行期显存，也不得把纯文本轮次改名为视觉轮次。
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from experiments.elastic_residency.vision.vision_measurement_semantics import (
    VISION_MEASUREMENT_SEMANTICS as _SEMANTICS,
)

RESIDENCY_TEXT_ONLY = "TEXT_ONLY"
RESIDENCY_TEXT_ONLY_CHAT = "TEXT_ONLY_CHAT"
RESIDENCY_VISION_WARM_GPU = "VISION_WARM_GPU"
RESIDENCY_VISION_WARM_CPU = "VISION_WARM_CPU"

TEXT_STATES = (RESIDENCY_TEXT_ONLY, RESIDENCY_TEXT_ONLY_CHAT)
VISION_STATES = (RESIDENCY_VISION_WARM_GPU, RESIDENCY_VISION_WARM_CPU)
ALL_STATES = TEXT_STATES + VISION_STATES

# 语义字典单独成模块（字段多且有大量"它不是什么"的边界说明），此处只做再导出。
VISION_MEASUREMENT_SEMANTICS: dict[str, str] = _SEMANTICS


def image_file_evidence(path: str | Path) -> dict[str, Any]:
    """记录任意输入图片的路径、大小与 SHA-256（用户自备图片路径）。"""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"找不到图片: {resolved}")
    payload = resolved.read_bytes()
    return {
        "source": "external_file",
        "path": str(resolved),
        "file_size_bytes": len(payload),
        "file_sha256": hashlib.sha256(payload).hexdigest(),
        "pixel_payload_sha256": None,
        "spec_id": None,
        "spec_sha256": None,
    }


def qwen_chat_prompt(
    *,
    question: str,
    media_marker: str | None,
) -> str:
    """按 Qwen3.5 自带模板的默认分支显式构造 prompt（不启用 thinking）。"""

    body = f"{media_marker}\n{question}" if media_marker else question
    return (
        "<|im_start|>user\n"
        f"{body}<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"
    )


def render_model_chat_prompt(
    *,
    chat_template: str,
    question: str,
    media_marker: str | None,
    eos_token: str,
    bos_token: str,
) -> str:
    """用模型自带 ``tokenizer.chat_template`` 渲染 prompt；失败直接抛错。"""

    from jinja2.ext import loopcontrols  # noqa: PLC0415
    from jinja2.sandbox import ImmutableSandboxedEnvironment  # noqa: PLC0415

    environment = ImmutableSandboxedEnvironment(
        trim_blocks=True,
        lstrip_blocks=True,
        extensions=[loopcontrols],
    )
    body = f"{media_marker}\n{question}" if media_marker else question

    def raise_exception(message: str) -> None:
        raise ValueError(message)

    return environment.from_string(chat_template).render(
        messages=[{"role": "user", "content": body}],
        add_generation_prompt=True,
        eos_token=eos_token,
        bos_token=bos_token,
        raise_exception=raise_exception,
    )


def build_prompt_evidence(
    *,
    mode: str,
    question: str,
    media_marker: str | None,
    chat_template: str | None = None,
    eos_token: str = "",
    bos_token: str = "",
) -> dict[str, Any]:
    """构造 prompt 文本及其可追溯证据。``mode`` 只接受两种显式取值。"""

    if mode == "qwen_im":
        text = qwen_chat_prompt(question=question, media_marker=media_marker)
    elif mode == "model_template":
        if not chat_template:
            raise ValueError("model_template 模式需要模型自带 chat_template")
        text = render_model_chat_prompt(
            chat_template=chat_template,
            question=question,
            media_marker=media_marker,
            eos_token=eos_token,
            bos_token=bos_token,
        )
    else:
        raise ValueError(f"未知 prompt 构造模式: {mode}")
    return {
        "prompt_construction_mode": mode,
        "prompt_text": text,
        "prompt_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "prompt_contains_media_marker": bool(
            media_marker and media_marker in text
        ),
        "question": question,
        "media_marker": media_marker,
    }


def memory_delta(
    *,
    before: dict[str, Any],
    after: dict[str, Any],
    before_rss_bytes: int,
    after_rss_bytes: int,
) -> dict[str, Any]:
    """计算两个 direct NVML 快照与 RSS 之间的差值（缺值保持 None）。"""

    before_card = before.get("gpu_card_used_mb")
    after_card = after.get("gpu_card_used_mb")
    return {
        "gpu_card_used_before_mb": before_card,
        "gpu_card_used_after_mb": after_card,
        "gpu_card_delta_mb": (
            after_card - before_card
            if before_card is not None and after_card is not None
            else None
        ),
        "rss_before_bytes": int(before_rss_bytes),
        "rss_after_bytes": int(after_rss_bytes),
        "rss_delta_bytes": int(after_rss_bytes) - int(before_rss_bytes),
    }


def projector_execution_checks(
    *,
    session_evidence: dict[str, Any],
    chunk_evidence: dict[str, Any],
    bitmap_evidence: list[dict[str, Any]],
    expected_image_count: int = 1,
    expected_visual_tokens: int | None = None,
) -> dict[str, bool]:
    """校验 projector 是否真实执行并把视觉 token 送进语言模型。"""

    checks: dict[str, bool] = {
        "projector_context_supports_vision": (
            session_evidence.get("supports_vision") is True
        ),
        "projector_load_time_recorded": (
            isinstance(session_evidence.get("projector_load_s"), (int, float))
            and float(session_evidence["projector_load_s"]) > 0.0
        ),
        "image_chunk_count_matches_input": (
            int(chunk_evidence.get("image_chunk_count", 0)) == expected_image_count
        ),
        "visual_tokens_positive": int(chunk_evidence.get("visual_token_total", 0)) > 0,
        "visual_positions_advanced_llm_context": (
            int(chunk_evidence.get("visual_position_total", 0)) > 0
        ),
        "text_chunks_present_around_image": (
            int(chunk_evidence.get("text_chunk_count", 0)) > 0
        ),
        "all_bitmaps_decoded_as_image": bool(bitmap_evidence)
        and all(
            item["decoded_width_px"] > 0
            and item["decoded_height_px"] > 0
            and item["decoded_bytes"] > 0
            and item["is_audio"] is False
            for item in bitmap_evidence
        ),
    }
    if chunk_evidence.get("visual_embedding_available") is not None:
        checks["projector_returned_visual_embedding"] = bool(
            chunk_evidence["visual_embedding_available"]
        )
        checks["projector_encode_time_isolated"] = (
            isinstance(chunk_evidence.get("projector_encode_total_s"), (int, float))
            and float(chunk_evidence["projector_encode_total_s"]) > 0.0
        )
    if expected_visual_tokens is not None:
        checks["visual_token_count_matches_patch_expectation"] = (
            int(chunk_evidence.get("visual_token_total", 0))
            == int(expected_visual_tokens)
        )
    return checks


def output_text_checks(
    *,
    output_text: str,
    generated_token_count: int,
    expected_token_count: int,
) -> dict[str, bool]:
    """校验模型确实产生了可复查的文字输出。"""

    return {
        "generation_token_count_exact": (
            generated_token_count == expected_token_count
        ),
        "output_text_non_empty": bool(output_text.strip()),
    }
