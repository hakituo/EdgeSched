"""启动策略：首句即播 vs 全量预缓冲（随 ``s0_protocol_v1`` 冻结）。

两种策略的区别只在**何时打开输出流**，其余播放路径与测量口径完全一致：

- ``first_sentence``：消费前打开输出流，首句合成完即开播。首句就绪之前的
  空白属于 ``startup`` 阶段，落在统计窗口之外，单独报告。
- ``full_prebuffer``：先把整段回复合成完并留在内存，再打开输出流连续播放。
  代价是第一声更晚（首播等待变长），换来窗口内供给缺口趋零。

禁止为个别配置单独调节预缓冲来掩盖断流：正式比较只使用冻结的
``first_sentence``，``full_prebuffer`` 仅用于启动策略小对照，两者必须同时报告
首播等待代价与窗口内供给缺口。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from experiments.elastic_residency.s0_pipeline.config import (
    S0_STARTUP_POLICIES,
    STARTUP_FULL_PREBUFFER,
)


def ensure_player_started(player: Any) -> None:
    """幂等启动播放：打开输出流并校准设备时钟。"""

    if getattr(player, "is_opened", False):
        return
    player.open()
    player.calibrate_clock()


@dataclass
class StartupPlan:
    """按冻结策略决定音频是立即提交还是先预缓冲。"""

    policy: str
    _held: list[tuple[int, Any, float]] = field(default_factory=list)
    held_audio_s: float = 0.0
    flushed_sentences: int = 0
    flush_started_ns: int | None = None

    def __post_init__(self) -> None:
        if self.policy not in S0_STARTUP_POLICIES:
            raise ValueError(
                f"未冻结的启动策略 {self.policy!r}，只允许 {S0_STARTUP_POLICIES}"
            )

    @property
    def is_prebuffer(self) -> bool:
        return self.policy == STARTUP_FULL_PREBUFFER

    @property
    def held_count(self) -> int:
        return len(self._held)

    def hold(self, chunk_id: int, samples: Any, audio_duration_s: float) -> None:
        """预缓冲模式：暂存一句音频，不立即提交给设备。"""

        self._held.append((chunk_id, samples, float(audio_duration_s)))
        self.held_audio_s += float(audio_duration_s)

    async def flush(self, player: Any, sentences: list[dict[str, Any]]) -> int:
        """预缓冲模式：开播并逐句提交；非预缓冲模式为空操作。"""

        if not self.is_prebuffer or not self._held:
            return 0
        ensure_player_started(player)
        self.flush_started_ns = time.monotonic_ns()
        by_chunk = {int(item["chunk_id"]): item for item in sentences}
        for chunk_id, samples, _ in self._held:
            span = await player.submit(chunk_id, samples)
            target = by_chunk.get(int(chunk_id))
            if target is not None:
                target["start_sample"] = span.start_sample
                target["end_sample"] = span.end_sample
        self.flushed_sentences = len(self._held)
        self._held.clear()
        return self.flushed_sentences

    def evidence(self) -> dict[str, Any]:
        """启动策略证据；预缓冲代价必须与收益同时落盘。"""

        return {
            "policy": self.policy,
            "is_prebuffer": self.is_prebuffer,
            "prebuffered_sentences": self.flushed_sentences,
            "prebuffered_audio_s": (
                round(self.held_audio_s, 6) if self.flushed_sentences else 0.0
            ),
            "flush_started_ns": self.flush_started_ns,
            "note": (
                "先合成整段回复再开播；首播等待代价与窗口内供给缺口必须同时报告"
                if self.is_prebuffer
                else "首句波形就绪即开播；启动等待落在统计窗口之外"
            ),
        }
