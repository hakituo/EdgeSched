"""V0 消融单元定义、核划分与结果汇总（纯标准库，可单测）。

每个单元只改变一个因素，相对 ``alone``（LLM 单独跑）或 ``tts_thread``（S0 同款：
同进程、TTS 持续合成、llama 28 线程）比较：

| 单元 | 改变的因素 | 用来回答 |
|---|---|---|
| alone | — | 本机隔离基线（对照 C2 画像） |
| tts_idle | TTS 已加载但不合成 | 仅驻留是否就会拖慢 |
| tts_thread | TTS 在同进程线程里持续合成 | 复现 S0 的 4 倍减速 |
| tts_thread_t14 | llama 线程 28→14（每物理核 1 个） | 是否是 SMT/线程超额订阅 |
| tts_thread_torch1 | torch 线程池 1 | 是否是 torch 线程抢核 |
| tts_proc_overlap | TTS 放到独立进程，与 LLM 共用核 | 是否是 GIL（同进程才有） |
| tts_proc_disjoint | TTS 独立进程且绑到不重叠的核 | 是否是 CPU 争用（剩下的只有 GPU 争用） |
| tts_alone | 只有 TTS 合成，没有 LLM | TTS 自身 RTF 基线（反向干扰） |
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import median
from typing import Any, Iterable

TTS_MODES = ("none", "idle", "thread", "proc_overlap", "proc_disjoint")


@dataclass(frozen=True)
class Cell:
    name: str
    run_llm: bool
    tts_mode: str
    llm_threads: int | None  # None = 与 LLM 逻辑核数相同（S0 口径）
    torch_threads: int | None  # None = 不设置（torch 默认）
    question: str

    def __post_init__(self) -> None:
        if self.tts_mode not in TTS_MODES:
            raise ValueError(f"未知 tts_mode {self.tts_mode!r}")
        if not self.run_llm and self.tts_mode not in ("thread",):
            raise ValueError("不跑 LLM 的单元只用于测 TTS 自身，必须持续合成")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


CELLS: tuple[Cell, ...] = (
    Cell("alone", True, "none", None, None, "本机隔离基线"),
    Cell("tts_idle", True, "idle", None, None, "TTS 仅驻留是否拖慢"),
    Cell("tts_thread", True, "thread", None, None, "复现 S0 共跑减速"),
    Cell("tts_thread_t14", True, "thread", 14, None, "llama 线程超额订阅"),
    Cell("tts_thread_torch1", True, "thread", None, 1, "torch 线程池抢核"),
    Cell("tts_proc_overlap", True, "proc_overlap", None, None, "同进程 GIL"),
    Cell("tts_proc_disjoint", True, "proc_disjoint", None, None, "CPU 争用 vs GPU 争用"),
    Cell("tts_alone", False, "thread", None, None, "TTS 自身 RTF 基线"),
    # 第二批（v1 结果出来后补）：区分「14 线程本身更快」与「给 TTS 留出逻辑核」
    Cell("alone_t14", True, "none", 14, None, "14 线程的隔离基线"),
    Cell("tts_proc_disjoint_t14", True, "proc_disjoint", 14, None, "核不重叠 + 14 线程：只剩 GPU 争用"),
)

CELLS_BY_NAME = {cell.name: cell for cell in CELLS}


def select_cells(names: Iterable[str] | None) -> list[Cell]:
    if not names:
        return list(CELLS)
    unknown = [name for name in names if name not in CELLS_BY_NAME]
    if unknown:
        raise ValueError(f"未知单元 {unknown}，可选 {sorted(CELLS_BY_NAME)}")
    return [CELLS_BY_NAME[name] for name in names]


def split_cores(
    sibling_groups: list[list[int]], llm_physical: int, tts_physical: int
) -> tuple[list[int], list[int]]:
    """按物理核划分：LLM 取低编号物理核（含 SMT 兄弟），TTS 取最高编号且不重叠。

    ``sibling_groups`` 每项是一个物理核的全部逻辑 CPU。
    """

    groups = sorted((sorted(group) for group in sibling_groups), key=lambda g: g[0])
    if llm_physical + tts_physical > len(groups):
        raise ValueError(
            f"物理核不足：需要 {llm_physical}+{tts_physical}，只有 {len(groups)}"
        )
    llm = sorted(cpu for group in groups[:llm_physical] for cpu in group)
    tts = sorted(cpu for group in groups[len(groups) - tts_physical :] for cpu in group)
    return llm, tts


def decode_rate(token_ns: list[int]) -> float | None:
    """首 token 之后的平均交付速度（tok/s）。"""

    if len(token_ns) < 2:
        return None
    span = (token_ns[-1] - token_ns[0]) / 1e9
    return (len(token_ns) - 1) / span if span > 0 else None


def tts_rtf_during(
    synth_records: list[dict[str, Any]], start_ns: int | None = None, end_ns: int | None = None
) -> float | None:
    """落在窗口内（按开始时刻）的合成记录的总合成时长 / 总音频时长。"""

    picked = [
        record
        for record in synth_records
        if (start_ns is None or record["start_ns"] >= start_ns)
        and (end_ns is None or record["start_ns"] <= end_ns)
    ]
    audio = sum(record["audio_s"] for record in picked)
    if audio <= 0:
        return None
    return sum((r["end_ns"] - r["start_ns"]) / 1e9 for r in picked) / audio


def summarize(cell_results: list[dict[str, Any]]) -> dict[str, Any]:
    """同名单元跨轮合并全部重复，取中位数，并给出相对 alone 的减速倍数。"""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in cell_results:
        grouped.setdefault(result["cell"]["name"], []).append(result)
    rows: dict[str, dict[str, Any]] = {}
    for name, results in grouped.items():
        reps = [rep for result in results for rep in (result.get("reps") or [])]
        rates = [rep["decode_tok_s"] for rep in reps if rep.get("decode_tok_s")]
        ttfts = [rep["ttft_s"] for rep in reps if rep.get("ttft_s") is not None]
        rtfs = [rep["tts_rtf"] for rep in reps if rep.get("tts_rtf") is not None]
        rtfs += [r["tts_alone_rtf"] for r in results if r.get("tts_alone_rtf") is not None]
        result = results[-1]
        rows[name] = {
            "passed": all(bool(r.get("passed")) for r in results),
            "error": next((r.get("error") for r in results if r.get("error")), None),
            "processes": len(results),
            "decode_tok_s_min": round(min(rates), 3) if rates else None,
            "decode_tok_s_max": round(max(rates), 3) if rates else None,
            "reps": len(rates),
            "decode_tok_s": round(median(rates), 3) if rates else None,
            "ttft_s": round(median(ttfts), 3) if ttfts else None,
            "tts_rtf": round(median(rtfs), 4) if rtfs else None,
            "llm_threads": result.get("llm_threads"),
            "torch_threads": result.get("torch_threads_effective"),
        }
    base = (rows.get("alone") or {}).get("decode_tok_s")
    for row in rows.values():
        rate = row["decode_tok_s"]
        row["slowdown_vs_alone_x"] = round(base / rate, 2) if base and rate else None
    return rows
