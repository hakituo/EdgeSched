"""small 9B 离线筛选产物到共置复算视图的只读适配。

只做结构映射，不做任何容量计算：
small screen 用 `provisional_g_safe` + `text_llm_tts_gpu_only` 描述候选，
历史复算入口需要 `g_safe` + `per_context`，两者语义等价但键名不同。

small screen 的 Diffusion 只有主机内存诊断，没有 H_D 档位，
因此映射后 `h_d` 的档位一律保持 `None`，不得凭空填 0。
"""

from __future__ import annotations

from typing import Any

LEGACY = "legacy"
SMALL = "small"


def detect_schema(screen: dict[str, Any]) -> str:
    """判别 screen 结构；无法识别时报错而不是猜。"""
    if "per_context" in screen and "g_safe" in screen:
        return LEGACY
    if "provisional_g_safe" in screen and "text_llm_tts_gpu_only" in screen:
        return SMALL
    raise ValueError("无法识别的 screen 结构：既不是 legacy 也不是 small。")


def _profiled_configs(entry: dict[str, Any]) -> list[str]:
    """收集该上下文已画像的驻留档位，供候选合法性校验。"""
    names: set[str] = set()
    for tts_entry in entry.values():
        names.update(tts_entry.get("m_l_usage_mb", {}))
    return sorted(names)


def to_legacy_view(screen: dict[str, Any]) -> dict[str, Any]:
    """把 small screen 映射为 legacy 视图；保留已画像档位清单。"""
    per_context: dict[str, Any] = {}
    for context, entry in screen["text_llm_tts_gpu_only"].items():
        per_context[str(context)] = {
            "h_tts": {
                tts: {
                    "h_layers": value.get("h_layers"),
                    "full_feasible": bool(value.get("full_feasible")),
                    "budget_for_m_l_mb": value.get("budget_for_m_l_mb"),
                }
                for tts, value in entry.items()
            },
            "h_d": {"h_layers": None, "full_feasible": False},
            "profiled_llm_configs": _profiled_configs(entry),
        }
    provisional = screen["provisional_g_safe"]
    return {
        "analysis_name": screen.get("analysis_name", "small_stage_screen"),
        "g_safe": {
            "value_mb": provisional.get("value_mb"),
            "components": dict(provisional.get("components", {})),
            "pending_components": list(provisional.get("pending_components", [])),
        },
        "per_context": per_context,
        "screen_schema": SMALL,
    }


def load_screen(screen: dict[str, Any], schema: str = "auto") -> tuple[dict[str, Any], str]:
    """按给定或自动判别的 schema 返回可复算视图与实际 schema。"""
    resolved = detect_schema(screen) if schema == "auto" else schema
    if resolved == SMALL:
        return to_legacy_view(screen), SMALL
    return screen, LEGACY
