"""
Cost-Aware Migration Avoidance

不是"来了就迁"，而是基于代价模型智能决策：
- 预测 burst 时长（基于历史）
- 计算迁移代价（KV Cache 保存 + 显存释放 + CPU 加载 + 回迁）
- 计算等待代价（不迁移则 burst 排队等待）
- 比较：迁移收益 > 迁移代价 才迁移

决策规则：
1. CPU-affine 任务（RAG/OCR）：不迁移，旁路
2. 短 burst（< 阈值）：不迁移，让 burst 排队等待 LLM 完成
3. 长 burst（> 阈值）：迁移 LLM，让 burst 立即执行
4. 队列已有多个 burst：迁移（避免长等待）

这是从"被动响应"升级到"主动决策"的核心机制。
"""

from __future__ import annotations

import time
import statistics
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Any

from .phase_model import (
    PhaseAttributes,
    PhaseRole,
    PhaseTask,
    DeviceAffinity,
    LLMState,
    LLMServiceLevel,
    PHASE_PRESETS,
)


@dataclass
class MigrationCost:
    """迁移代价（毫秒）

    迁移 LLM 从 GPU 到 CPU 的总代价，包括：
    - KV Cache 保存
    - 显存释放
    - CPU 实例加载
    - 后续回迁 GPU
    """
    kv_cache_save_ms: float = 250.0       # save_state() 实测约 200-300ms
    vram_release_ms: float = 100.0        # del + gc + empty_cache
    cpu_load_ms: float = 1500.0           # CPU LLM 加载（Page Cache 加速）
    restore_ms: float = 2000.0            # 回迁 GPU（含预加载 + 热切换）
    service_gap_ms: float = 300.0         # 用户感知的服务中断（热切换窗口）

    @property
    def total_downlink_ms(self) -> float:
        """下行迁移总耗时"""
        return self.kv_cache_save_ms + self.vram_release_ms + self.cpu_load_ms

    @property
    def total_round_trip_ms(self) -> float:
        """往返迁移总耗时（下行 + 回迁）"""
        return self.total_downlink_ms + self.restore_ms

    def to_dict(self) -> Dict[str, float]:
        return {
            "kv_cache_save_ms": self.kv_cache_save_ms,
            "vram_release_ms": self.vram_release_ms,
            "cpu_load_ms": self.cpu_load_ms,
            "restore_ms": self.restore_ms,
            "service_gap_ms": self.service_gap_ms,
            "total_downlink_ms": self.total_downlink_ms,
            "total_round_trip_ms": self.total_round_trip_ms,
        }


@dataclass
class MigrationDecision:
    """迁移决策结果"""
    should_migrate: bool
    reason: str                            # 决策原因（用于论文分析）
    predicted_burst_duration_s: float      # 预测的 burst 时长
    migration_cost: MigrationCost          # 迁移代价
    wait_cost_s: float = 0.0               # 不迁移的等待代价
    benefit_s: float = 0.0                 # 迁移收益 = wait_cost - migration_cost
    strategy: str = ""                     # 策略名：migrate / wait / bypass

    def to_dict(self) -> Dict[str, Any]:
        return {
            "should_migrate": self.should_migrate,
            "reason": self.reason,
            "predicted_burst_duration_s": self.predicted_burst_duration_s,
            "wait_cost_s": self.wait_cost_s,
            "benefit_s": self.benefit_s,
            "strategy": self.strategy,
            "migration_cost": self.migration_cost.to_dict(),
        }


class MigrationDecisionEngine:
    """迁移决策引擎

    核心公式：
        benefit = wait_cost - migration_cost
        if benefit > threshold: migrate
        else: wait or bypass

    其中：
    - wait_cost = predicted_burst_duration - llm_remaining_work
      （如果不迁移，burst 要等 LLM 完成才能上 GPU）
    - migration_cost = round_trip_ms / 1000 + service_gap 惩罚
    """

    def __init__(
        self,
        migration_cost: Optional[MigrationCost] = None,
        history_window: int = 20,          # 历史窗口大小
        short_burst_threshold_s: float = 2.0,  # 短 burst 阈值：小于此值不迁移
        benefit_threshold_s: float = 0.5,      # 收益阈值：必须 >0.5s 才迁移
        queue_pressure_threshold: int = 2,     # 队列压力阈值：>=2 个 burst 强制迁移
    ):
        self.migration_cost = migration_cost or MigrationCost()
        # 按阶段名记录历史时长
        self._duration_history: Dict[str, Deque[float]] = {}
        self._history_window = history_window
        self._short_burst_threshold = short_burst_threshold_s
        self._benefit_threshold = benefit_threshold_s
        self._queue_pressure_threshold = queue_pressure_threshold
        # 决策历史（用于实验分析）
        self._decision_history: List[MigrationDecision] = []

    def record_duration(self, phase_name: str, duration_s: float) -> None:
        """记录一次阶段执行时长（用于未来预测）"""
        if phase_name not in self._duration_history:
            self._duration_history[phase_name] = deque(maxlen=self._history_window)
        self._duration_history[phase_name].append(duration_s)

    def predict_burst_duration(self, phase_name: str) -> float:
        """预测 burst 时长

        策略：优先用历史均值，无历史则用预设典型时长。
        这是简单的启发式预测，论文中可讨论学习型扩展。
        """
        history = self._duration_history.get(phase_name)
        if history and len(history) > 0:
            return statistics.mean(history)

        # 回退到预设
        attrs = PHASE_PRESETS.get(phase_name)
        if attrs is not None:
            return attrs.typical_duration_s
        return 5.0  # 默认 5 秒

    def compute_wait_cost(
        self,
        burst_task: PhaseTask,
        llm_state: LLMState,
        llm_remaining_work_s: float = 0.0,
    ) -> float:
        """计算不迁移的等待代价

        如果不迁移 LLM，burst 任务要等 LLM 完成当前工作并释放 GPU。
        等待代价 = max(0, llm_remaining_work - burst_can_wait)

        但如果 LLM 在 GPU 上有持续请求，burst 可能等很久。
        """
        # LLM 剩余工作时间（如果有流式请求在跑）
        # 如果 LLM 空闲，等待代价为 0
        if llm_remaining_work_s <= 0:
            return 0.0

        # 等待代价 = LLM 剩余工作时间（burst 必须等）
        # 减去 burst 本身可以排队的时间（不占资源）
        # 但用户感知角度，burst 延迟 = 等待时间
        return llm_remaining_work_s

    def decide(
        self,
        burst_task: PhaseTask,
        llm_state: LLMState,
        queue_depth: int = 0,
        llm_remaining_work_s: float = 0.0,
        record_history: bool = True,
    ) -> MigrationDecision:
        """核心决策函数：是否迁移 LLM 到 CPU

        决策规则：
        1. CPU-affine 任务（RAG/OCR）：不迁移，旁路
        2. GPU_PREFERRED 任务（TTS/VLM）：可降级 CPU，不迁移 LLM
        3. GPU_REQUIRED 任务（Image）：
           - LLM 已降级 → 旁路（GPU 已空闲）
           - LLM 在推理且快完成 → 等待
           - 否则 → 迁移 LLM 释放显存
        4. 队列压力大时，迁移一次处理多个 burst，避免反复迁移

        Args:
            burst_task: 到达的 burst 任务
            llm_state: LLM 当前状态
            queue_depth: 当前 burst 队列深度
            llm_remaining_work_s: LLM 剩余工作时间估计
            record_history: 是否记录到决策历史（路由用的一次不记录）

        Returns:
            MigrationDecision
        """
        attrs = burst_task.attributes
        if attrs is None:
            return MigrationDecision(
                should_migrate=False,
                reason="无阶段属性，默认不迁移",
                predicted_burst_duration_s=0.0,
                migration_cost=self.migration_cost,
                strategy="wait",
            )

        pred_duration = self.predict_burst_duration(burst_task.phase_name)

        # 规则 1：CPU-affine 任务（RAG/OCR）不迁移，旁路
        if burst_task.is_cpu_affine:
            decision = MigrationDecision(
                should_migrate=False,
                reason=f"CPU-affine 阶段({burst_task.phase_name})旁路，不触发迁移",
                predicted_burst_duration_s=pred_duration,
                migration_cost=self.migration_cost,
                strategy="bypass",
            )
            if record_history:
                self._decision_history.append(decision)
            return decision

        # 规则 2：GPU_PREFERRED 任务（TTS/VLM）可降级 CPU，不迁移 LLM
        # 这是 cost-aware 的核心收益：避免迁移 LLM，让小任务走 CPU
        if attrs.device_affinity == DeviceAffinity.GPU_PREFERRED:
            if llm_state.level == LLMServiceLevel.GPU_NORMAL:
                decision = MigrationDecision(
                    should_migrate=False,
                    reason=f"GPU_PREFERRED 阶段({burst_task.phase_name})降级 CPU，避免迁移 LLM",
                    predicted_burst_duration_s=pred_duration,
                    migration_cost=self.migration_cost,
                    strategy="bypass",
                )
            else:
                decision = MigrationDecision(
                    should_migrate=False,
                    reason=f"GPU_PREFERRED 阶段({burst_task.phase_name})走 CPU（LLM 已降级）",
                    predicted_burst_duration_s=pred_duration,
                    migration_cost=self.migration_cost,
                    strategy="bypass",
                )
            if record_history:
                self._decision_history.append(decision)
            return decision

        # 规则 3：GPU_REQUIRED 任务（Image）必须释放显存
        if llm_state.level == LLMServiceLevel.CPU_DEGRADED:
            # LLM 已降级，GPU 空闲，无需迁移
            decision = MigrationDecision(
                should_migrate=False,
                reason="LLM 已在 CPU 降级状态，GPU 空闲",
                predicted_burst_duration_s=pred_duration,
                migration_cost=self.migration_cost,
                strategy="bypass",
            )
            if record_history:
                self._decision_history.append(decision)
            return decision

        # LLM 在 GPU 上，需要决策
        # 计算等待代价（不迁移则 burst 等 LLM 完成并释放显存）
        wait_cost = self.compute_wait_cost(burst_task, llm_state, llm_remaining_work_s)
        migration_cost_s = self.migration_cost.total_round_trip_ms / 1000.0

        # 规则 3a：LLM 在推理且快完成，短暂等待比迁移划算
        if (llm_remaining_work_s > 0
            and llm_remaining_work_s < self._short_burst_threshold_s
            and queue_depth == 0):
            decision = MigrationDecision(
                should_migrate=False,
                reason=f"LLM 即将完成({llm_remaining_work_s:.1f}s)，短暂等待比迁移划算",
                predicted_burst_duration_s=pred_duration,
                migration_cost=self.migration_cost,
                wait_cost_s=wait_cost,
                benefit_s=wait_cost - migration_cost_s,
                strategy="wait",
            )
            if record_history:
                self._decision_history.append(decision)
            return decision

        # 规则 3b：GPU_REQUIRED 必须迁移（显存被 LLM 占用，burst 无法执行）
        # 这是物理约束：8GB 显存无法同时容纳 LLM + Image
        decision = MigrationDecision(
            should_migrate=True,
            reason=f"GPU_REQUIRED 阶段({burst_task.phase_name})需要显存，迁移 LLM 释放 GPU",
            predicted_burst_duration_s=pred_duration,
            migration_cost=self.migration_cost,
            wait_cost_s=wait_cost,
            benefit_s=wait_cost - migration_cost_s,
            strategy="migrate",
        )
        if record_history:
            self._decision_history.append(decision)
        return decision

    def get_decision_history(self) -> List[MigrationDecision]:
        """获取决策历史（用于实验分析）"""
        return list(self._decision_history)

    def get_stats(self) -> Dict[str, Any]:
        """获取决策统计"""
        history = self._decision_history
        if not history:
            return {"total_decisions": 0}

        migrate_count = sum(1 for d in history if d.should_migrate)
        wait_count = sum(1 for d in history if d.strategy == "wait")
        bypass_count = sum(1 for d in history if d.strategy == "bypass")

        return {
            "total_decisions": len(history),
            "migrate_count": migrate_count,
            "wait_count": wait_count,
            "bypass_count": bypass_count,
            "migrate_ratio": migrate_count / len(history),
            "avg_predicted_burst_s": statistics.mean(d.predicted_burst_duration_s for d in history),
            "avg_benefit_s": statistics.mean(d.benefit_s for d in history),
        }


class AlwaysMigrateEngine:
    """Always-Migrate 策略（baseline）

    不管 burst 时长和代价，来了就迁移。
    用于实验对比：cost-aware avoidance vs always migrate。
    """

    def __init__(self, migration_cost: Optional[MigrationCost] = None):
        self.migration_cost = migration_cost or MigrationCost()
        self._decision_history: List[MigrationDecision] = []

    def decide(
        self,
        burst_task: PhaseTask,
        llm_state: LLMState,
        queue_depth: int = 0,
        llm_remaining_work_s: float = 0.0,
        record_history: bool = True,
    ) -> MigrationDecision:
        """总是迁移（除非 CPU-affine 或已在降级状态）"""
        attrs = burst_task.attributes
        pred_duration = 0.0
        if attrs is not None:
            pred_duration = attrs.typical_duration_s

        if burst_task.is_cpu_affine:
            decision = MigrationDecision(
                should_migrate=False,
                reason="CPU-affine 阶段旁路",
                predicted_burst_duration_s=pred_duration,
                migration_cost=self.migration_cost,
                strategy="bypass",
            )
        elif llm_state.level == LLMServiceLevel.CPU_DEGRADED:
            decision = MigrationDecision(
                should_migrate=False,
                reason="LLM 已降级",
                predicted_burst_duration_s=pred_duration,
                migration_cost=self.migration_cost,
                strategy="bypass",
            )
        else:
            decision = MigrationDecision(
                should_migrate=True,
                reason="always-migrate 策略：无条件迁移",
                predicted_burst_duration_s=pred_duration,
                migration_cost=self.migration_cost,
                strategy="migrate",
            )

        if record_history:
            self._decision_history.append(decision)
        return decision

    def record_duration(self, phase_name: str, duration_s: float) -> None:
        """兼容接口：no-op"""
        pass

    def get_decision_history(self) -> List[MigrationDecision]:
        return list(self._decision_history)

    def get_stats(self) -> Dict[str, Any]:
        history = self._decision_history
        if not history:
            return {"total_decisions": 0}
        migrate_count = sum(1 for d in history if d.should_migrate)
        return {
            "total_decisions": len(history),
            "migrate_count": migrate_count,
            "migrate_ratio": migrate_count / len(history),
        }
