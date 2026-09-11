"""S0 端到端流水线的参数、门禁与冻结状态。

pilot 阶段只要求「参数可记录、可复核」，不声称协议已冻结。
正式比较前必须把 07 模块表格中的待确认项逐条落到本文件并锁定。
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

#: 已冻结的四个容量可行配置（LLM 档位 : TTS 放置）。
S0_PLACEMENTS: tuple[str, ...] = ("K24:T0", "K24:T3", "K0:T0", "K24:T4")

#: 冻结的容量口径上下文长度。
S0_CONTEXT_TOKENS = 512

SCHEMA_VERSION = "edgesched.s0_pipeline.v1"

#: 统一的直接回答 prompt：四配置共用，禁止思考正文、分析与 Markdown。
#: 它只约束正文语义；模型仍泄漏思考正文时由输出契约判失败，不静默删除。
PROMPT_CONTRACT_VERSION = "s0_direct_answer_v1"
DIRECT_ANSWER_SYSTEM_PROMPT = (
    "直接回答用户问题，只输出需要朗读的最终正文。"
    "不要输出思考过程、分析步骤、计划或任何推理说明；"
    "不要使用 Markdown 列表、编号、加粗、标题或代码块；"
    "用完整自然的中文句子回答，句末使用中文句号。"
)


def text_fingerprint(value: str) -> str:
    """文本指纹：用于把 prompt/系统提示固化进产物并可复核。"""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class S0PipelineConfig:
    """一次 S0 流水线运行的完整可序列化配置。"""

    placement: str
    context_tokens: int = S0_CONTEXT_TOKENS
    generation_budget_tokens: int = 128
    max_output_tokens: int = 128
    seed: int = 42
    temperature: float = 0.0
    min_chars: int = 12
    max_chars: int = 80
    tts_queue_maxsize: int = 4
    token_queue_maxsize: int = 16
    audio_queue_maxsize: int = 4
    playback_blocksize: int = 1024
    playback_device: int | None = None
    prefer_wasapi: bool = True
    llm_cores: int = 14
    system_cores: int = 2
    sample_interval_s: float = 0.05
    gpu_sample_interval_s: float = 0.25
    gpu_warmup_timeout_s: float = 5.0
    max_gpu_sample_age_ms: float = 750.0
    max_swap_growth_mb: float = 4.0
    safety_margin_mb: float = 1024.0
    language: str = "Chinese"
    system_prompt: str = DIRECT_ANSWER_SYSTEM_PROMPT
    prompt_contract_version: str = PROMPT_CONTRACT_VERSION
    attributes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.placement not in S0_PLACEMENTS:
            raise ValueError(
                f"未冻结的放置配置 {self.placement!r}，只允许 {S0_PLACEMENTS}"
            )
        if self.context_tokens != S0_CONTEXT_TOKENS:
            raise ValueError(
                f"C512 口径只允许 context_tokens={S0_CONTEXT_TOKENS}，"
                f"实际 {self.context_tokens}"
            )
        if self.max_output_tokens < 1:
            raise ValueError("max_output_tokens 必须为正")
        if self.max_output_tokens > self.generation_budget_tokens:
            raise ValueError(
                "max_output_tokens 超过生成预算，可能越过冻结的 C512 口径"
            )
        for name in (
            "tts_queue_maxsize",
            "token_queue_maxsize",
            "audio_queue_maxsize",
            "playback_blocksize",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} 必须为正")
        if self.min_chars < 1 or self.max_chars < self.min_chars:
            raise ValueError("分句长度参数非法")

    @property
    def llm_config(self) -> str:
        return self.placement.split(":")[0]

    @property
    def tts_config(self) -> str:
        return self.placement.split(":")[1]

    @property
    def n_ctx(self) -> int:
        return self.context_tokens + self.generation_budget_tokens

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": SCHEMA_VERSION, **asdict(self)}


#: 与 07 模块表格一一对应；pilot 阶段一律为 pending，禁止预填数值。
FREEZE_STATUS: dict[str, str] = {
    "input_and_generation": "pending",
    "sentence_splitting": "pending",
    "startup_policy": "pending",
    "first_sentence_ready_slo": "pending",
    "actual_playback_slo": "pending",
    "continuity_gate": "pending",
    "comparison_rule": "pending",
}


def default_output_root(project_root: Path) -> Path:
    return (
        Path(project_root)
        / "experiment_results"
        / "elastic_residency"
        / "s0_pipeline"
    )


def parse_placement(value: str) -> tuple[str, str]:
    parts = value.split(":")
    if len(parts) != 2:
        raise ValueError(f"非法放置配置 {value!r}，应为 K档:T档")
    llm_config, tts_config = (part.strip().upper() for part in parts)
    if f"{llm_config}:{tts_config}" not in S0_PLACEMENTS:
        raise ValueError(f"未冻结的放置配置 {value!r}")
    return llm_config, tts_config
