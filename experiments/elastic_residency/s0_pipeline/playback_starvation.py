"""断流统计：把「缺供给 callback 次数」与「独立断流次数」分开。

两个单位必须分清，否则「注入 1 秒间隙 → 25 条 starve」会被误读成 25 次断流：

- ``starve_callback_count``：缺数据的 callback 次数（旧口径，只说明有多少次
  回调拿不到样本）；
- ``starve_episode_count``：相邻缺音合并后的**独立断流次数**（报告用这个）。

每段落还要分阶段，因为缺音的成因不同：

- ``startup``：首句首样本之前的等待；
- ``inter_sentence``：两句之间（含等待尚未提交的下一句）；
- ``intra_sentence``：落在某一句的样本区间内；
- ``tail``：末句末样本之后、生产者已结束时的收尾补零；
- ``unknown``：无法归属时保留未知，**不强行绑定到句子**。

统计窗口默认取「首句首样本 → 末句末样本」，启动与收尾单独报告，
因为它们的分母含义与句中不同，混算会让比例失真。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

PHASE_STARTUP = "startup"
PHASE_INTER_SENTENCE = "inter_sentence"
PHASE_INTRA_SENTENCE = "intra_sentence"
PHASE_TAIL = "tail"
PHASE_UNKNOWN = "unknown"

ATTR_STARTUP_WAIT = "before_first_sentence"
ATTR_SENTENCE_BOUNDARY = "boundary_before_sentence"
ATTR_WITHIN_SENTENCE = "within_sentence"
ATTR_AFTER_LAST = "after_last_sentence"
ATTR_BEFORE_UNSUBMITTED = "before_next_unsubmitted_sentence"
ATTR_UNKNOWN = "unknown"

#: 启动等待与收尾补零不属于连续供给窗口，单独报告。
_OUT_OF_WINDOW_PHASES = frozenset({PHASE_STARTUP, PHASE_TAIL})

#: 真正的「供给缺口」：在应连续供给的区间内拿不到数据。
#: 无法归属的段落计入缺口但保留 unknown，不强行绑定句子。
_SUPPLY_FAULT_PHASES = frozenset(
    {PHASE_INTER_SENTENCE, PHASE_INTRA_SENTENCE, PHASE_UNKNOWN}
)

SpanTuple = tuple[int, int, int]  # (chunk_id, start_sample, end_sample)


@dataclass(frozen=True)
class StarveEpisode:
    """一段连续的缺音；``chunk_id`` 为 None 表示不可归属。"""

    episode_id: int
    start_sample: int
    end_sample: int
    missing_samples: int
    callback_count: int
    phase: str
    chunk_id: int | None
    attribution: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "start_sample": self.start_sample,
            "end_sample": self.end_sample,
            "missing_samples": self.missing_samples,
            "callback_count": self.callback_count,
            "phase": self.phase,
            "chunk_id": self.chunk_id,
            "attribution": self.attribution,
        }


class StarvationTracker:
    """按 callback 观测缺音，合并成独立断流并分阶段。"""

    def __init__(self) -> None:
        self._episodes: list[StarveEpisode] = []
        self._current: dict[str, Any] | None = None
        self.callback_starve_count = 0
        self.missing_samples_total = 0

    def observe(
        self,
        *,
        sample_position: int,
        end_position: int,
        missing_samples: int,
        spans: list[SpanTuple],
        producer_done: bool,
    ) -> int | None:
        """记录一次 callback；返回当前断流段 id，无缺音返回 None。

        段落的归属与阶段使用**首次观测时**的已提交句快照，避免后来提交的
        句子把「首句之前的等待」错误改判成句内。
        """

        if missing_samples <= 0:
            self._close()
            return None
        self.callback_starve_count += 1
        self.missing_samples_total += int(missing_samples)
        current = self._current
        if current is None or current["end_sample"] != sample_position:
            episode_id = len(self._episodes)
            self._current = {
                "episode_id": episode_id,
                "start_sample": sample_position,
                "end_sample": end_position,
                "missing_samples": int(missing_samples),
                "callback_count": 1,
                "spans": list(spans),
                "producer_done": producer_done,
            }
            return episode_id
        current["end_sample"] = end_position
        current["missing_samples"] += int(missing_samples)
        current["callback_count"] += 1
        current["producer_done"] = producer_done
        return int(current["episode_id"])

    def finalize(self, *, producer_done: bool = True) -> None:
        if self._current is not None:
            self._current["producer_done"] = producer_done
        self._close()

    def has_open_episode(self) -> bool:
        return self._current is not None

    def episodes(self) -> list[StarveEpisode]:
        return list(self._episodes)

    def summary(
        self,
        *,
        window_start: int | None,
        window_end: int | None,
    ) -> dict[str, Any]:
        """汇总；窗口内外分开统计，避免启动/收尾污染比例。"""

        in_window = 0
        fault_episodes = 0
        fault_samples = 0
        by_phase: dict[str, dict[str, int]] = {}
        for item in self._episodes:
            entry = by_phase.setdefault(
                item.phase, {"episodes": 0, "samples": 0, "callbacks": 0}
            )
            entry["episodes"] += 1
            entry["samples"] += item.missing_samples
            entry["callbacks"] += item.callback_count
            if item.phase in _SUPPLY_FAULT_PHASES:
                fault_episodes += 1
                fault_samples += item.missing_samples
            if item.phase in _OUT_OF_WINDOW_PHASES:
                # 启动等待与收尾补零不属于「连续供给窗口」，不计入窗口内缺音
                continue
            if window_start is not None and window_end is not None:
                overlap = min(item.end_sample, window_end) - max(
                    item.start_sample, window_start
                )
                if overlap > 0:
                    in_window += overlap
        window_samples = (
            max(0, window_end - window_start)
            if window_start is not None and window_end is not None
            else 0
        )
        return {
            "starve_callback_count": self.callback_starve_count,
            "starve_episode_count": len(self._episodes),
            "starve_samples_total": self.missing_samples_total,
            "supply_fault_episode_count": fault_episodes,
            "supply_fault_samples": fault_samples,
            "starve_samples_in_window": in_window,
            "window_start_sample": window_start,
            "window_end_sample": window_end,
            "window_samples": window_samples,
            "starve_ratio_in_window": (
                round(in_window / window_samples, 6) if window_samples > 0 else None
            ),
            "by_phase": by_phase,
            "episodes": [item.to_dict() for item in self._episodes],
        }

    def _close(self) -> None:
        if self._current is None:
            return
        start = int(self._current["start_sample"])
        phase, chunk_id, attribution = _classify(
            start_sample=start,
            spans=list(self._current["spans"]),
            producer_done=bool(self._current["producer_done"]),
        )
        self._episodes.append(
            StarveEpisode(
                episode_id=int(self._current["episode_id"]),
                start_sample=start,
                end_sample=int(self._current["end_sample"]),
                missing_samples=int(self._current["missing_samples"]),
                callback_count=int(self._current["callback_count"]),
                phase=phase,
                chunk_id=chunk_id,
                attribution=attribution,
            )
        )
        self._current = None


def _classify(
    *,
    start_sample: int,
    spans: list[SpanTuple],
    producer_done: bool,
) -> tuple[str, int | None, str]:
    """按全局样本流分阶段；句子在流中是连续的，句间没有额外区域。

    因此「句间」只可能表现为**已消费位置越过最后一句的末样本**，也就是在等
    下一句（可能尚未提交）；无法归属时保留 unknown，不强行绑定。
    """

    if not spans:
        return PHASE_STARTUP, None, ATTR_STARTUP_WAIT
    first_start = min(item[1] for item in spans)
    last_end = max(item[2] for item in spans)
    if start_sample < first_start:
        return PHASE_STARTUP, None, ATTR_STARTUP_WAIT
    for chunk_id, span_start, span_end in spans:
        if span_start <= start_sample < span_end:
            return PHASE_INTRA_SENTENCE, chunk_id, ATTR_WITHIN_SENTENCE
    if start_sample >= last_end:
        if producer_done:
            return PHASE_TAIL, None, ATTR_AFTER_LAST
        return PHASE_INTER_SENTENCE, None, ATTR_BEFORE_UNSUBMITTED
    return PHASE_UNKNOWN, None, ATTR_UNKNOWN
