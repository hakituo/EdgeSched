"""容量筛选前合并并校验 C2/C3 LLM 驻留曲线输入。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def merge_llm_curve_summaries(
    coarse: Mapping[str, Any],
    refinement: Mapping[str, Any] | None,
    *,
    expected_config_count: int | None = None,
) -> dict[str, Any]:
    """按 context 合并互斥配置，并拒绝失败、重复或不完整的输入。"""

    sources = [coarse] + ([refinement] if refinement is not None else [])
    for source in sources:
        if source.get("passed") is not True:
            raise ValueError(
                f"LLM 曲线 {source.get('run_id', '<unknown>')} 未通过，拒绝分析"
            )
        if not isinstance(source.get("by_context_config"), Mapping):
            raise ValueError("LLM 曲线缺少 by_context_config")

    context_sets = [set(source["by_context_config"]) for source in sources]
    if any(contexts != context_sets[0] for contexts in context_sets[1:]):
        raise ValueError("C2/C3 context 集合不一致，拒绝静默合并")

    merged: dict[str, dict[str, Any]] = {}
    for context in sorted(context_sets[0], key=int):
        merged_configs: dict[str, Any] = {}
        for source in sources:
            for config, stats in source["by_context_config"][context].items():
                if config in merged_configs:
                    raise ValueError(f"C{context} 配置 {config} 在 C2/C3 中重复")
                merged_configs[config] = stats
        if (
            expected_config_count is not None
            and len(merged_configs) != expected_config_count
        ):
            raise ValueError(
                f"C{context} 合并后只有 {len(merged_configs)} 个配置，"
                f"预期 {expected_config_count}"
            )
        merged[str(context)] = merged_configs

    run_ids = [str(source.get("run_id", "")) for source in sources]
    return {
        "run_id": "+".join(run_ids),
        "source_run_ids": run_ids,
        "passed": True,
        "by_context_config": merged,
    }
