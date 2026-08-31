"""LLM+Forge 共置容量和正残差的纯计算口径。"""

from __future__ import annotations

from typing import Any


def forge_colocated_metrics(
    *,
    global_idle_reference_mb: float,
    b_llm_mb: float,
    p_colocated_mb: float,
    gpu_total_mb: float,
    safety_margin_mb: float,
    llm_isolated_increment_mb: float,
    forge_isolated_increment_mb: float,
) -> dict[str, Any]:
    """以统一增量口径计算容量和单模型加和之外的正残差。"""

    if not 0 <= global_idle_reference_mb <= b_llm_mb <= p_colocated_mb:
        raise ValueError("B_idle/B_LLM/P_colocated 顺序不合法")
    actual_colocated_delta_mb = p_colocated_mb - global_idle_reference_mb
    isolated_sum_mb = llm_isolated_increment_mb + forge_isolated_increment_mb
    positive_residual_mb = max(0.0, actual_colocated_delta_mb - isolated_sum_mb)
    safe_capacity_used_mb = actual_colocated_delta_mb + safety_margin_mb
    physical_headroom_mb = gpu_total_mb - p_colocated_mb
    safe_capacity_headroom_mb = gpu_total_mb - safe_capacity_used_mb
    return {
        "global_idle_reference_mb": global_idle_reference_mb,
        "b_llm_mb": b_llm_mb,
        "p_colocated_mb": p_colocated_mb,
        "gpu_total_mb": gpu_total_mb,
        "safety_margin_mb": safety_margin_mb,
        "llm_isolated_increment_mb": llm_isolated_increment_mb,
        "forge_isolated_increment_mb": forge_isolated_increment_mb,
        "isolated_sum_mb": isolated_sum_mb,
        "actual_colocated_delta_mb": actual_colocated_delta_mb,
        "colocated_positive_residual_mb": positive_residual_mb,
        "safe_capacity_used_mb": safe_capacity_used_mb,
        "physical_headroom_mb": physical_headroom_mb,
        "safe_capacity_headroom_mb": safe_capacity_headroom_mb,
        "capacity_admissible": (
            physical_headroom_mb >= 0.0 and safe_capacity_headroom_mb >= 0.0
        ),
        "capacity_formula": (
            "(P_colocated - B_idle_reference) + G_safe <= gpu_total and "
            "P_colocated <= gpu_total"
        ),
    }
