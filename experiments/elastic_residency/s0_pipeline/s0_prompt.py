"""S0 的 prompt 构造：用模型自带模板渲染，并显式关闭 thinking。

证据与边界：

- Qwen3.5 自带模板默认分支会在 ``assistant`` 轮渲染 ``<think>``，
  导致模型先输出 ``Thinking Process:`` 计划正文；仅靠系统提示无法消除。
- 本模块按项目既有做法，在渲染结果后追加空 think 块
  ``<think>\\n\\n</think>\\n\\n``，让模型直接进入正文；该做法已由真实
  模型输出验证（2026-09-12 探针：同一模型同一 seed 下正文干净）。
- 渲染只用模型自带 ``tokenizer.chat_template``，不硬编码模型名称；
  渲染结果与指纹全部落盘，供跨批次复核。
"""

from __future__ import annotations

import hashlib
from typing import Any

#: 追加到渲染结果后的空思考块；顺序不可调换，必须紧跟 assistant 轮。
EMPTY_THINK_PREFILL = "<think>\n\n</think>\n\n"

PROMPT_RENDER_MODE = "model_template_empty_think_prefill"


def render_chat_prompt(
    *,
    chat_template: str,
    system_prompt: str,
    user_prompt: str,
    eos_token: str = "",
    bos_token: str = "",
) -> str:
    """用模型自带模板渲染 system+user，并追加空 think 块关闭 thinking。"""

    if not chat_template.strip():
        raise ValueError("模型缺少 tokenizer.chat_template，无法构造 prompt")
    from jinja2.ext import loopcontrols  # noqa: PLC0415 - 只在真实环境需要
    from jinja2.sandbox import ImmutableSandboxedEnvironment  # noqa: PLC0415

    environment = ImmutableSandboxedEnvironment(
        trim_blocks=True,
        lstrip_blocks=True,
        extensions=[loopcontrols],
    )
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    def raise_exception(message: str) -> None:
        raise ValueError(message)

    rendered = environment.from_string(chat_template).render(
        messages=messages,
        add_generation_prompt=True,
        eos_token=eos_token,
        bos_token=bos_token,
        raise_exception=raise_exception,
    )
    return f"{rendered}{EMPTY_THINK_PREFILL}"


def prompt_evidence(rendered_prompt: str) -> dict[str, Any]:
    """渲染结果的可复核证据；不含任何文本内容之外的推断。"""

    return {
        "prompt_render_mode": PROMPT_RENDER_MODE,
        "rendered_prompt": rendered_prompt,
        "rendered_prompt_sha256": hashlib.sha256(
            rendered_prompt.encode("utf-8")
        ).hexdigest(),
        "thinking_suppression": "empty_think_block_prefill",
    }


def special_token_text(llm: Any, *, eos: bool) -> str:
    """读取模型自带的 eos/bos 文本；读取失败返回空串，由调用方决定。"""

    try:
        token_id = llm.token_eos() if eos else llm.token_bos()
        if token_id is None or int(token_id) < 0:
            return ""
        return llm.detokenize([int(token_id)], special=True).decode(
            "utf-8", "ignore"
        )
    except Exception:  # noqa: BLE001 - 不同后端取值方式不同，失败即放弃 stop
        return ""
