"""LLM+Forge 共置候选与筛选画像契约。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LlmForgeCandidate:
    context_tokens: int
    llm_config: str

    @property
    def label(self) -> str:
        return f"{self.context_tokens}:{self.llm_config}"


def parse_candidate_specs(items: list[str] | None) -> tuple[LlmForgeCandidate, ...]:
    """解析 ``CONTEXT:LLM``，保持顺序并拒绝重复。"""

    candidates: list[LlmForgeCandidate] = []
    seen: set[tuple[int, str]] = set()
    for item in items or []:
        parts = item.split(":")
        if len(parts) != 2:
            raise ValueError(f"非法候选 {item!r}，应为 CONTEXT:LLM")
        context_text, llm_config = (part.strip() for part in parts)
        try:
            context_tokens = int(context_text)
        except ValueError as exc:
            raise ValueError(f"候选 context 不是整数: {item!r}") from exc
        llm_config = llm_config.upper()
        key = (context_tokens, llm_config)
        if context_tokens <= 0 or not llm_config:
            raise ValueError(f"非法候选 {item!r}，context 必须为正且配置非空")
        if key in seen:
            raise ValueError(f"候选重复: {item!r}")
        seen.add(key)
        candidates.append(LlmForgeCandidate(*key))
    return tuple(candidates)


def screened_candidates(screen: dict[str, Any]) -> tuple[LlmForgeCandidate, ...]:
    """从 C3 筛选输出读取逐 context 的 H_D 候选。"""

    result: list[LlmForgeCandidate] = []
    for context in sorted(screen["per_context"], key=int):
        h_d = screen["per_context"][context]["h_d"]
        layers = h_d.get("h_layers")
        if layers is None:
            raise ValueError(f"C{context} 没有可验证的 H_D 候选")
        config = "FULL" if h_d.get("full_feasible") else f"K{layers}"
        result.append(LlmForgeCandidate(int(context), config))
    return tuple(result)


def candidate_profile(
    screen: dict[str, Any], candidate: LlmForgeCandidate
) -> dict[str, float]:
    """读取候选所需的隔离增量、G_safe 与 GPU 容量。"""

    expected = set(screened_candidates(screen))
    if candidate not in expected:
        raise ValueError(f"候选不属于筛选 H_D 边界: {candidate.label}")
    entry = screen["per_context"][str(candidate.context_tokens)]
    return {
        "llm_isolated_increment_mb": float(entry["m_l_mb"][candidate.llm_config]),
        "forge_isolated_increment_mb": float(entry["h_d"]["m_d_mb"]),
        "global_idle_reference_mb": float(
            screen["g_safe"]["components"]["idle_gpu_baseline_high_water_mb"]
        ),
        "safety_margin_mb": float(screen["g_safe"]["value_mb"]),
        "gpu_total_mb": float(screen["gpu_total_mb"]),
    }


def build_execution_plan(
    candidates: tuple[LlmForgeCandidate, ...], *, runs: int
) -> list[tuple[int, str, int, int]]:
    if not candidates:
        raise ValueError("LLM+Forge 候选不能为空")
    if runs < 1:
        raise ValueError("runs 必须至少为 1")
    plan: list[tuple[int, str, int, int]] = []
    ordinal = 0
    for candidate in candidates:
        for run_index in range(1, runs + 1):
            ordinal += 1
            plan.append(
                (candidate.context_tokens, candidate.llm_config, run_index, ordinal)
            )
    return plan
