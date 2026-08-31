"""LLM+TTS 共置矩阵的显式候选解析与执行顺序。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ColocatedCandidate:
    context_tokens: int
    llm_config: str
    tts_config: str

    @property
    def label(self) -> str:
        return f"{self.context_tokens}:{self.llm_config}:{self.tts_config}"


def parse_candidate_specs(items: list[str] | None) -> tuple[ColocatedCandidate, ...]:
    """解析 ``CONTEXT:LLM:TTS``，保持顺序并拒绝重复。"""

    candidates: list[ColocatedCandidate] = []
    seen: set[tuple[int, str, str]] = set()
    for item in items or []:
        parts = item.split(":")
        if len(parts) != 3:
            raise ValueError(f"非法候选 {item!r}，应为 CONTEXT:LLM:TTS")
        context_text, llm_config, tts_config = (part.strip() for part in parts)
        try:
            context_tokens = int(context_text)
        except ValueError as exc:
            raise ValueError(f"候选 context 不是整数: {item!r}") from exc
        llm_config = llm_config.upper()
        tts_config = tts_config.upper()
        if context_tokens <= 0 or not llm_config or not tts_config:
            raise ValueError(f"非法候选 {item!r}，字段必须非空且 context > 0")
        key = (context_tokens, llm_config, tts_config)
        if key in seen:
            raise ValueError(f"候选重复: {item!r}")
        seen.add(key)
        candidates.append(ColocatedCandidate(*key))
    return tuple(candidates)


def build_execution_plan(
    candidates: tuple[ColocatedCandidate, ...],
    *,
    runs: int,
) -> list[tuple[int, str, str, int, int]]:
    """候选优先、轮次次之；最紧候选可放首位以配合 fail-fast。"""

    if runs < 1:
        raise ValueError("runs 必须至少为 1")
    plan: list[tuple[int, str, str, int, int]] = []
    ordinal = 0
    for candidate in candidates:
        for run_index in range(1, runs + 1):
            ordinal += 1
            plan.append(
                (
                    candidate.context_tokens,
                    candidate.llm_config,
                    candidate.tts_config,
                    run_index,
                    ordinal,
                )
            )
    return plan
