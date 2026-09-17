"""B2 冻结口径：句长文本、度量聚合与放置状态。

**B2 = 当前接入后端（加速整模型 GPU，T0）的 TTS 画像**：20/50/100 字 × 五轮，
逐轮记录 RTF、波形就绪时间、加载与图准备成本、准备/执行阶段内存峰值、输出长度
与正确性。旧后端（上游动态路径）的组件放置矩阵 T0–T4 与其三份产物已删除，
B2 不再引用任何旧后端数值；T1–T4 是否重测由 ``placement_status`` 按后端接口
能力判定，不支持的显式排除。

本模块只放纯数据与纯函数，便于在没有 GPU 的环境里静态验证。
"""

from __future__ import annotations

import hashlib
from typing import Any, Iterable

import numpy as np

#: B2 冻结文本；20/50/100 字，改文本必须整轮重跑。
TEXTS: dict[str, str] = {
    "chars_20": "别着急，先喝口水，再继续处理后面的任务。",
    "chars_50": (
        "每个测试句子都会保存独立音频和结构化指标，后续可以从原始结果"
        "重新计算平均值、尾延迟和峰值显存。复核。"
    ),
    "chars_100": (
        "如果长句的实时率明显高于短句，说明生成成本会随文本长度快速增加，"
        "调度器就需要限制单个语音块的最大长度。反过来，如果不同句长的实"
        "时率比较稳定，我们可以把句子切分策略主要用于控制首块等待和播放缓"
        "冲。再测。"
    ),
}
EXPECTED_LENGTHS: dict[str, int] = {"chars_20": 20, "chars_50": 50, "chars_100": 100}
WARMUP_TEXT = "这是一次不计入统计的预热，用来稳定执行路径。"

#: 当前后端支持的放置标签；整模型上 GPU。
PLACEMENT_CURRENT = "T0"
#: 旧后端组件放置维度，当前后端不支持，一律排除而不是沿用旧值。
PLACEMENT_EXCLUDED = ("T1", "T2", "T3", "T4")

#: 正确性序列：同一句重复三次并交替不同句长，用来查状态残留与串句。
CORRECTNESS_REPEAT = "chars_20"


def validate_texts() -> None:
    actual = {name: len(text) for name, text in TEXTS.items()}
    if actual != EXPECTED_LENGTHS:
        raise ValueError(f"B2 冻结文本长度错误: {actual}")


def length_sequence(rounds: int) -> list[tuple[str, str]]:
    """句长画像序列：三种句长各 ``rounds`` 轮。"""

    if rounds < 1:
        raise ValueError("rounds 必须大于 0")
    return [(name, text) for name in TEXTS for text in [TEXTS[name]] * rounds]


def correctness_sequence() -> list[tuple[str, str]]:
    """三段重复 + 交替长短，检查后端是否残留上一次的状态。"""

    return [
        (CORRECTNESS_REPEAT, TEXTS[CORRECTNESS_REPEAT]),
        (CORRECTNESS_REPEAT, TEXTS[CORRECTNESS_REPEAT]),
        (CORRECTNESS_REPEAT, TEXTS[CORRECTNESS_REPEAT]),
        ("chars_100", TEXTS["chars_100"]),
        (CORRECTNESS_REPEAT, TEXTS[CORRECTNESS_REPEAT]),
        ("chars_50", TEXTS["chars_50"]),
        (CORRECTNESS_REPEAT, TEXTS[CORRECTNESS_REPEAT]),
    ]


def to_mono(audio: Any) -> np.ndarray:
    """统一成 (samples,) float32；两条路径都可能返回 list。"""

    array = audio[0] if isinstance(audio, (list, tuple)) else audio
    return np.asarray(array, dtype=np.float32).reshape(-1)


def pcm_sha256(samples: np.ndarray) -> str:
    """对 PCM16 负载取哈希；与 WAV 头无关，便于跨后端与跨轮次比较。"""

    clipped = np.clip(samples, -1.0, 1.0)
    return hashlib.sha256((clipped * 32767.0).astype(np.int16).tobytes()).hexdigest()


def _mean(values: Iterable[float]) -> float:
    ordered = [float(value) for value in values]
    return round(sum(ordered) / len(ordered), 4) if ordered else 0.0


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """按 (path, text) 汇总；纯函数，便于静态验证。"""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(f"{row['path']}|{row['text']}", []).append(row)
    summary: dict[str, Any] = {}
    for key, items in grouped.items():
        rtf = [float(item["rtf"]) for item in items]
        elapsed = [float(item["waveform_ready_s"]) for item in items]
        audio = [float(item["audio_s"]) for item in items]
        summary[key] = {
            "rounds": len(items),
            "rtf_mean": _mean(rtf),
            "rtf_min": round(min(rtf), 4),
            "rtf_max": round(max(rtf), 4),
            "waveform_ready_mean_s": _mean(elapsed),
            "waveform_ready_min_s": round(min(elapsed), 4),
            "waveform_ready_max_s": round(max(elapsed), 4),
            "audio_mean_s": _mean(audio),
            "distinct_pcm_sha256": sorted({item["pcm_sha256"] for item in items}),
        }
    return summary


def _card_delta(
    phase_execute: dict[str, Any], idle_card_used_mb: float | None
) -> tuple[float, str | None]:
    """执行阶段相对空闲基线的整卡增量；NVML 不可用时退回 torch 保留峰值。"""

    card = phase_execute.get("gpu_card_used_mb_max")
    if card is not None and idle_card_used_mb is not None:
        return round(float(card) - float(idle_card_used_mb), 1), "direct_nvml_delta"
    reserved = phase_execute.get("torch_peak_reserved_mb")
    if reserved is not None:
        return float(reserved), "torch_peak_reserved_fallback"
    raise ValueError("既没有 NVML 整卡数据也没有 torch 保留峰值，无法给出 B2 内存画像")


def build_b2_summary(
    rows: list[dict[str, Any]],
    *,
    integration_path: str,
    phases: dict[str, Any],
    idle_card_used_mb: float | None,
    placement: str = PLACEMENT_CURRENT,
    git_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """B2 汇总：``by_config`` 保持 ``analyze_results.extract_tts_profiles`` 形状。

    只有实际接入路径进入 ``by_config``（容量筛选用），其他路径放在 ``by_path``
    仅作后端对照，不参与容量结论。``passed`` 只有在波形正确性核对通过后才为真，
    来源不干净（脏树或无 git 状态）时 ``git_provenance_complete`` 为假，下游筛选
    据此拒绝这份画像，而不是拿它当已晋升依据。
    """

    execute = phases["execute"]
    delta, delta_source = _card_delta(execute, idle_card_used_mb)
    integration = [row for row in rows if row["path"] == integration_path]
    if not integration:
        raise ValueError(f"缺少接入路径 {integration_path} 的测量记录")
    readiness = [float(row["waveform_ready_s"]) for row in integration]
    by_path = aggregate(rows)
    git_available = bool((git_state or {}).get("available"))
    return {
        "placement": placement,
        "placement_status": phases.get("placement_status"),
        "integration_path": integration_path,
        "passed": False,
        "passed_note": (
            "由主环境 ASR 正确性核对回填：correctness_check=completed_all_ok 时才为真"
        ),
        "git_provenance_complete": bool(git_available and (git_state or {}).get("dirty") is False),
        "git_state": git_state,
        "by_config": {
            placement: {
                "max_card_delta_mb": delta,
                "max_card_delta_mb_source": delta_source,
                "torch_peak_allocated_mb": execute.get("torch_peak_allocated_mb"),
                "torch_peak_reserved_mb": execute.get("torch_peak_reserved_mb"),
                "idle_card_used_mb": idle_card_used_mb,
                "waveform_ready_mean_s": _mean(readiness),
                "waveform_ready_max_s": round(max(readiness), 4),
                "load_s": phases["prepare"].get("load_s"),
                "graph_capture_s": phases["prepare"].get("graph_capture_s"),
                "prepare_rss_mb_max": phases["prepare"].get("rss_mb_max"),
                "execute_rss_mb_max": execute.get("rss_mb_max"),
                "selectable": True,
            }
        },
        "by_path": by_path,
        "phases": phases,
        "correctness_check": "pending_external_asr_pass",
        "correctness_note": (
            "波形正确性由主环境 ASR 回转核对写入同目录 correctness.json；"
            "隔离环境不装 ASR，避免为核对再改后端环境"
        ),
    }


def placement_status(capability: dict[str, Any]) -> dict[str, Any]:
    """当前后端支持的放置维度；不支持的一律排除。"""

    supported = [PLACEMENT_CURRENT] if capability.get("whole_model_device_parameter") else []
    return {
        "supported": supported,
        "excluded": list(PLACEMENT_EXCLUDED),
        "reason": (
            "加速后端只暴露整模型 device，没有 talker/codec/speaker 分部件放置接口；"
            "旧后端的组件放置数值不适用于当前接入路径"
        ),
        "cpu_whole_model": capability.get("cpu_whole_model"),
        "rule": "未确认支持的放置不进入 B2/S0 矩阵，也不沿用旧值凑齐 T0–T4",
        "capability": capability,
    }
