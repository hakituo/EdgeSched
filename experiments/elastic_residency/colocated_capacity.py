"""共置容量与正残差的纯计算口径。"""

from __future__ import annotations

from typing import Any


def colocated_metrics(
    *,
    global_idle_baseline_mb: float,
    b_llm_mb: float,
    p_colocated_mb: float,
    gpu_total_mb: float,
    safety_margin_mb: float,
    tts_isolated_increment_mb: float | None,
) -> dict[str, Any]:
    """计算共置残差与扣除空闲基线后的安全容量裁决。

    ``G_safe`` 已包含全局空闲 GPU 高水位，因此容量裁决必须先从实测
    整卡峰值 ``P_colocated`` 扣除本轮全局空闲基线。M_L(k, context) 已包含
    该 LLM 配置的 KV buffer，因此这里不接受也不叠加任何 KV 字段。
    """

    if b_llm_mb is None or p_colocated_mb is None:
        raise ValueError("B_LLM 或 P_colocated 不可用时不得计算共置指标")
    if p_colocated_mb < b_llm_mb:
        raise ValueError("P_colocated 低于 B_LLM，测量口径不一致")
    if p_colocated_mb < global_idle_baseline_mb:
        raise ValueError("P_colocated 低于全局空闲基线，测量口径不一致")

    tts_colocated_increment_mb = p_colocated_mb - b_llm_mb
    colocated_positive_residual_mb = (
        max(0.0, tts_colocated_increment_mb - tts_isolated_increment_mb)
        if tts_isolated_increment_mb is not None
        else None
    )
    colocated_card_delta_mb = p_colocated_mb - global_idle_baseline_mb
    physical_headroom_mb = gpu_total_mb - p_colocated_mb
    safe_capacity_used_mb = colocated_card_delta_mb + safety_margin_mb
    capacity_headroom_mb = gpu_total_mb - safe_capacity_used_mb
    capacity_admissible = (
        physical_headroom_mb >= 0.0 and capacity_headroom_mb >= 0.0
    )
    return {
        "global_idle_baseline_mb": global_idle_baseline_mb,
        "b_llm_mb": b_llm_mb,
        "p_colocated_mb": p_colocated_mb,
        "gpu_total_mb": gpu_total_mb,
        "safety_margin_mb": safety_margin_mb,
        "tts_colocated_increment_mb": tts_colocated_increment_mb,
        "tts_isolated_increment_mb": tts_isolated_increment_mb,
        "colocated_positive_residual_mb": colocated_positive_residual_mb,
        "colocated_card_delta_mb": colocated_card_delta_mb,
        "physical_headroom_mb": physical_headroom_mb,
        "safe_capacity_used_mb": safe_capacity_used_mb,
        "capacity_headroom_mb": capacity_headroom_mb,
        "capacity_admissible": capacity_admissible,
        "capacity_formula": (
            "(P_colocated - B_idle) + G_safe <= gpu_total and "
            "P_colocated <= gpu_total "
            "(kv 已含在实测峰值内，不额外叠加)"
        ),
        "kv_buffer_added_to_capacity": False,
    }
