"""
GPU Lease / VRAM Lease 机制

把 GPU 使用权建模成短期租约（lease）。
- Image 这种任务申请 non-preemptible GPU lease
- LLM 是 resident lease holder，但可以被降级（degraded）
- TTS/RAG/OCR 不申请 GPU lease

这是形式化 GPU 资源管理的核心抽象。
借鉴分布式系统的 lease 思想，但应用到单机多模型 GPU 调度。
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Callable, Awaitable, Any

from .phase_model import PhaseRole, LLMServiceLevel, LLMState


class LeaseType(Enum):
    """租约类型"""
    RESIDENT = "resident"      # 常驻租约（LLM 持有，可降级释放）
    BURST = "burst"            # 突发租约（Image 持有，不可抢占）
    SHARED = "shared"          # 共享租约（多个小任务共享 GPU）


@dataclass
class GPULease:
    """GPU 租约：描述一个任务对 GPU 资源的占用

    每个租约有明确的持有者、类型、显存需求、预计时长。
    租约的获取/释放/降级是调度器决策的核心。

    注意：使用 kw_only 避免默认值字段顺序冲突。
    """
    lease_id: str = field(default_factory=lambda: f"lease_{uuid.uuid4().hex[:10]}")
    holder: str = ""                          # 持有者标识（task_id 或 "llm_resident"）
    holder_role: PhaseRole = PhaseRole.BURST_GPU_PINNED  # 持有者角色
    lease_type: LeaseType = LeaseType.BURST
    preemptible: bool = False             # 是否可被抢占
    acquired_at: float = field(default_factory=time.time)
    duration_pred_s: float = 0.0          # 预计持有时长（秒）
    vram_required_mb: int = 0             # 显存需求（MB）
    released: bool = False
    released_at: Optional[float] = None

    @property
    def elapsed_s(self) -> float:
        """已持有时长"""
        end = self.released_at or time.time()
        return end - self.acquired_at

    @property
    def is_overdue(self) -> bool:
        """是否超时（超过预测时长）"""
        if self.duration_pred_s <= 0:
            return False
        return self.elapsed_s > self.duration_pred_s * 1.5  # 50% 容差


class GPULeaseManager:
    """GPU 租约管理器

    管理所有 GPU 租约的获取、释放、降级。
    核心约束：8GB 显存下，同时只能有一个 GPU 重任务（互斥）。

    决策规则：
    1. RESIDENT 租约（LLM）可被降级释放（迁移到 CPU）
    2. BURST 租约（Image）不可抢占，必须等待完成
    3. SHARED 租约可与其他 SHARED 共存（如果显存允许）
    """

    def __init__(self, vram_total_mb: int = 8192):
        self._vram_total = vram_total_mb
        self._active_leases: Dict[str, GPULease] = {}
        self._lock = asyncio.Lock()
        # 事件回调
        self._on_lease_acquired: Optional[Callable[[GPULease], Awaitable[None]]] = None
        self._on_lease_released: Optional[Callable[[GPULease], Awaitable[None]]] = None

    def set_callbacks(
        self,
        on_acquired: Optional[Callable[[GPULease], Awaitable[None]]] = None,
        on_released: Optional[Callable[[GPULease], Awaitable[None]]] = None,
    ) -> None:
        """设置租约事件回调（用于触发实际迁移/加载）"""
        self._on_lease_acquired = on_acquired
        self._on_lease_released = on_released

    @property
    def active_leases(self) -> List[GPULease]:
        """当前活跃租约列表"""
        return [l for l in self._active_leases.values() if not l.released]

    @property
    def vram_used_mb(self) -> int:
        """当前显存占用"""
        return sum(l.vram_required_mb for l in self.active_leases)

    @property
    def vram_free_mb(self) -> int:
        """剩余显存"""
        return self._vram_total - self.vram_used_mb

    def get_resident_lease(self) -> Optional[GPULease]:
        """获取当前常驻租约（LLM）"""
        for l in self.active_leases:
            if l.lease_type == LeaseType.RESIDENT:
                return l
        return None

    def get_burst_lease(self) -> Optional[GPULease]:
        """获取当前突发租约（Image 等）"""
        for l in self.active_leases:
            if l.lease_type == LeaseType.BURST:
                return l
        return None

    def has_gpu_burst_running(self) -> bool:
        """是否有 GPU 突发任务在运行"""
        return self.get_burst_lease() is not None

    def can_acquire(self, vram_required_mb: int, lease_type: LeaseType) -> bool:
        """检查是否能获取租约（不实际获取）

        规则：
        - BURST 租约需要独占 GPU（8GB 下无法与 RESIDENT 共存）
        - SHARED 租约可与其他 SHARED 共存（如果显存允许）
        """
        if lease_type == LeaseType.SHARED:
            # 共享租约：检查显存
            return self.vram_free_mb >= vram_required_mb

        # BURST / RESIDENT：需要独占
        # 如果已有 BURST 在跑，不能获取新的 BURST/RESIDENT
        if self.get_burst_lease() is not None:
            return False

        # 如果有 RESIDENT（LLM 在 GPU），BURST 需要先让 RESIDENT 降级
        # 这里只检查"物理可行性"：释放 RESIDENT 后显存够不够
        resident = self.get_resident_lease()
        if resident is not None:
            # 假设 RESIDENT 会被降级释放
            free_after_release = self.vram_free_mb + resident.vram_required_mb
            return free_after_release >= vram_required_mb

        # 没有 RESIDENT，直接检查显存
        return self.vram_free_mb >= vram_required_mb

    async def acquire(self, lease: GPULease) -> bool:
        """获取租约

        Returns:
            True 如果获取成功（可能触发了 RESIDENT 降级）
            False 如果无法获取（需要排队等待）
        """
        async with self._lock:
            if not self.can_acquire(lease.vram_required_mb, lease.lease_type):
                return False

            # 如果是 BURST 且有 RESIDENT 在 GPU，需要先降级 RESIDENT
            if lease.lease_type == LeaseType.BURST:
                resident = self.get_resident_lease()
                if resident is not None:
                    # 标记 RESIDENT 为降级（实际迁移由回调执行）
                    await self._downgrade_resident(resident)

            self._active_leases[lease.lease_id] = lease

            if self._on_lease_acquired is not None:
                await self._on_lease_acquired(lease)

            return True

    async def release(self, lease_id: str) -> Optional[GPULease]:
        """释放租约"""
        async with self._lock:
            lease = self._active_leases.get(lease_id)
            if lease is None or lease.released:
                return None

            lease.released = True
            lease.released_at = time.time()

            if self._on_lease_released is not None:
                await self._on_lease_released(lease)

            return lease

    async def _downgrade_resident(self, resident: GPULease) -> None:
        """降级常驻租约（LLM 迁移到 CPU）

        实际迁移由 phase_scheduler 通过回调执行。
        这里只标记租约状态。
        """
        resident.released = True
        resident.released_at = time.time()
        # 注意：实际迁移逻辑由 phase_scheduler 处理
        # 这里通过 on_lease_released 回调触发

    def get_state_snapshot(self) -> Dict[str, Any]:
        """获取状态快照（用于实验记录）"""
        return {
            "vram_total_mb": self._vram_total,
            "vram_used_mb": self.vram_used_mb,
            "vram_free_mb": self.vram_free_mb,
            "active_lease_count": len(self.active_leases),
            "has_resident": self.get_resident_lease() is not None,
            "has_burst": self.get_burst_lease() is not None,
            "leases": [
                {
                    "lease_id": l.lease_id,
                    "holder": l.holder,
                    "holder_role": l.holder_role.value,
                    "lease_type": l.lease_type.value,
                    "vram_required_mb": l.vram_required_mb,
                    "duration_pred_s": l.duration_pred_s,
                    "elapsed_s": l.elapsed_s,
                }
                for l in self.active_leases
            ],
        }
