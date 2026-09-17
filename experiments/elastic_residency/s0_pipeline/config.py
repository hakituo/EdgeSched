"""S0 端到端流水线的参数、门禁与冻结状态。

07 模块表格中的待确认项已在本文件冻结为 ``FROZEN_PROTOCOL``：
输入与生成、分句、启动策略、重复次数、输出差异与失败处理规则都已落值，
改协议必须同步改版本号并重跑，不能看完正式结果再调截止或预缓冲。

``s0_protocol_v2``（2026-09-17）相对 v1 的改动：TTS 接入后端改为加速 CUDA Graph
路径，放置维度只保留 T0（K24:T3、K24:T4 因后端不支持分部件放置而排除），
正式配置由 4 个降为 2 个（K24:T0 与 K0:T0）。
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

#: 已冻结的容量可行配置（LLM 档位 : TTS 放置）。
#: 2026-09-17 起只保留 T0 一档 TTS 放置：接入后端改为加速 CUDA Graph 路径后，
#: 后端只暴露整模型 device，没有 talker/codec/speaker 分部件放置接口，
#: 因此 T3/T4 显式排除（见 B2 的 placement_status），不为凑齐档位沿用旧后端数值。
S0_PLACEMENTS: tuple[str, ...] = ("K24:T0", "K0:T0")

#: 加速后端唯一支持的 TTS 放置：整模型驻 GPU。
TTS_PLACEMENT_T0 = "T0"
#: 旧后端组件放置维度；当前接入后端不支持，已排除。
TTS_PLACEMENTS_EXCLUDED: tuple[str, ...] = ("T1", "T2", "T3", "T4")

#: 启动策略：首句就绪即开播，或等全部句子合成完再开播。
STARTUP_FIRST_SENTENCE = "first_sentence"
STARTUP_FULL_PREBUFFER = "full_prebuffer"
S0_STARTUP_POLICIES: tuple[str, ...] = (
    STARTUP_FIRST_SENTENCE,
    STARTUP_FULL_PREBUFFER,
)
#: 正式比较冻结使用首句即播；全量预缓冲只用于启动策略小对照。
DEFAULT_STARTUP_POLICY = STARTUP_FIRST_SENTENCE

S0_PROTOCOL_VERSION = "s0_protocol_v2"

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
    startup_policy: str = DEFAULT_STARTUP_POLICY
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
        if self.startup_policy not in S0_STARTUP_POLICIES:
            raise ValueError(
                f"未冻结的启动策略 {self.startup_policy!r}，"
                f"只允许 {S0_STARTUP_POLICIES}"
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


#: 与 07 模块表格一一对应；2026-09-17 起为已冻结状态。
FREEZE_STATUS: dict[str, str] = {
    "input_and_generation": "frozen",
    "sentence_splitting": "frozen",
    "startup_policy": "frozen",
    "first_sentence_ready_slo": "frozen_as_reporting_only",
    "actual_playback_slo": "pending_device_validation",
    "continuity_gate": "frozen_scope_split",
    "comparison_rule": "frozen",
}

#: 冻结的正式 S0 协议；改任何一项都必须升 ``S0_PROTOCOL_VERSION`` 并重跑。
FROZEN_PROTOCOL: dict[str, Any] = {
    "version": S0_PROTOCOL_VERSION,
    "frozen_on": "2026-09-17",
    "input_and_generation": {
        "prompt": "请用三到五句话介绍边缘计算的优势。",
        "prompt_sha256": text_fingerprint("请用三到五句话介绍边缘计算的优势。"),
        "system_prompt_version": PROMPT_CONTRACT_VERSION,
        "seed": 42,
        "temperature": 0.0,
        "repetition_penalty": 1.05,
        "max_output_tokens": 128,
        "voice_ref": "tts_refs/ref_calm.wav",
        "tts_backend": "cuda_graph_static",
        "tts_backend_note": (
            "加速 CUDA Graph 路径，整模型驻 GPU（T0）；与 B2 画像同口径："
            "eager attention、x-vector 克隆、do_sample=False、非流式完整波形"
        ),
        "output_check": (
            "输出契约逐次判定；输出 token 数、句数与文本 SHA-256 作为协变量记录，"
            "不因长度或文本差异删除运行"
        ),
    },
    "sentence_splitting": {
        "min_chars": 12,
        "max_chars": 80,
        "strong_boundaries": "。！？!?；;\\n",
        "weak_boundaries": "，,：:",
        "tail_flush": True,
        "queue_order": "FIFO",
    },
    "startup_policy": {
        "formal": DEFAULT_STARTUP_POLICY,
        "comparison": [STARTUP_FIRST_SENTENCE, STARTUP_FULL_PREBUFFER],
        "note": (
            "正式比较用首句即播；全量预缓冲仅用于 K24:T0 启动策略小对照，"
            "同时报告首播等待代价与窗口内供给缺口，禁止逐配置调节预缓冲掩盖断流"
        ),
    },
    "placement_dimension": {
        "supported": [TTS_PLACEMENT_T0],
        "excluded": list(TTS_PLACEMENTS_EXCLUDED),
        "rule": (
            "加速后端只暴露整模型 device，没有分部件放置接口；"
            "未确认支持的放置不进入 S0 矩阵，也不沿用旧后端数值凑齐档位"
        ),
    },
    "repetitions": {
        "runs_per_placement": 5,
        "paired_order": "rotate",
        "formal_placements": list(S0_PLACEMENTS),
        "formal_total_runs": 5 * len(S0_PLACEMENTS),
    },
    "output_difference_rule": {
        "fingerprint": "逐句 raw_text 拼接后 SHA-256，另记输出 token 数",
        "tolerance": "文本 SHA 不同的运行不删除，按协变量报告并标注 confound",
        "confound_checks": ["output_tokens", "sentence_count", "audio_duration_s"],
    },
    "comparison_rule": {
        "paired_unit": "run_index",
        "primary_metrics": [
            "first_sentence_waveform_ready_s",
            "last_sentence_waveform_ready_s",
            "rtf_mean",
            "supply_fault_episode_count",
        ],
        "relative_tolerance": 0.10,
        "absolute_tolerance_s": 1.0,
        "baseline_non_mixed": "K0:T0",
        "reference_same_llm": "K24:T0",
        "note": (
            "无绝对 SLO 截止值，按配对相对容差判定 Pareto 收益；"
            "设备体验指标不参与比较，只报 pending"
        ),
    },
    "failure_handling": {
        "no_silent_drop": "失败运行必须落盘并计入分母，不得静默删除或替换",
        "fail_fast_default": True,
        "continue_on_failure": "仅用于收全证据，产物中仍标 passed=false",
        "display_missing": "缺少真实后端的运行直接失败，禁止 Mock fallback",
    },
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
