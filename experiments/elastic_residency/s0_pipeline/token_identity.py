"""S0 token 源的身份、边界与运行期统计（纯数据结构，便于单测）。

从 ``token_source`` 拆出以保持单文件规模上限；这里只定义可序列化数据
与不变量，不含任何后端依赖。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class TokenSourceError(RuntimeError):
    """token 源不可用或违反冻结边界。"""


@dataclass(frozen=True)
class TokenSourceIdentity:
    """模型身份与本次生成的 token 边界。"""

    model_path: str
    model_sha256: str
    architecture: str
    block_count: int
    n_ctx: int
    context_tokens: int
    generation_budget_tokens: int
    chat_template_sha256: str
    prompt_tokens_estimate: int
    max_output_tokens: int
    n_gpu_layers: int
    n_threads: int
    backend_kind: str = "llama_cpp_python"

    def __post_init__(self) -> None:
        if self.prompt_tokens_estimate > self.context_tokens:
            raise TokenSourceError(
                f"prompt token 数 {self.prompt_tokens_estimate} 超过冻结口径 "
                f"{self.context_tokens}"
            )
        if self.prompt_tokens_estimate + self.max_output_tokens > self.n_ctx:
            raise TokenSourceError(
                "prompt + max_output 超过 n_ctx，会越过 C512 冻结口径"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_path": self.model_path,
            "model_sha256": self.model_sha256,
            "architecture": self.architecture,
            "block_count": self.block_count,
            "n_ctx": self.n_ctx,
            "context_tokens": self.context_tokens,
            "generation_budget_tokens": self.generation_budget_tokens,
            "chat_template_sha256": self.chat_template_sha256,
            "prompt_tokens_estimate": self.prompt_tokens_estimate,
            "max_output_tokens": self.max_output_tokens,
            "n_gpu_layers": self.n_gpu_layers,
            "n_threads": self.n_threads,
            "backend_kind": self.backend_kind,
        }


@dataclass
class TokenSourceStats:
    """运行期统计；交付、阻塞与完成状态分开记录。"""

    delivered_tokens: int = 0
    first_delivery_ns: int | None = None
    last_delivery_ns: int | None = None
    bridge_max_depth: int = 0
    bridge_blocked_count: int = 0
    actual_output_tokens: int = 0
    finished: bool = False
    error: str = ""
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "delivered_tokens": self.delivered_tokens,
            "first_delivery_ns": self.first_delivery_ns,
            "last_delivery_ns": self.last_delivery_ns,
            "bridge_max_depth": self.bridge_max_depth,
            "bridge_blocked_count": self.bridge_blocked_count,
            "actual_output_tokens": self.actual_output_tokens,
            "finished": self.finished,
            "error": self.error,
            "notes": dict(self.notes),
        }
