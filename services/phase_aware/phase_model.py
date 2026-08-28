"""
Phase-Aware Workflow Model

把本地 AI 助手的多模态任务建模成多阶段 workflow。
核心抽象不是 "LLM vs Image"，而是阶段的属性维度：
  - resident/migratable（常驻可迁移）
  - GPU-exclusive/CPU-affine（设备亲和）
  - preemptible/non-preemptible（可抢占性）
  - upstream/downstream（workflow 中的位置）

这样任何新任务（视频生成、ASR 等）都能映射到属性维度，泛化到 N 个阶段。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Callable, Awaitable, Any


class PhaseRole(Enum):
    """阶段角色：定义任务在 workflow 中的角色

    对应论文中的 "phase-level resource complementarity" 抽象。
    每个角色有不同的调度规则，不是非此即彼。
    """
    RESIDENT_MIGRATABLE = "resident_migratable"
    # 常驻可迁移：LLM。GPU 优先，可降级 CPU，需要持续可用

    BURST_GPU_PINNED = "burst_gpu_pinned"
    # GPU 突发不可迁移：图像生成。GPU 必需，不可抢占，可排队

    BURST_GPU_PREFERRED = "burst_gpu_preferred"
    # GPU 偏好可降级：VLM/TTS。GPU 更快但可走 CPU，可与 CPU 任务并行

    BURST_CPU_LIGHT = "burst_cpu_light"
    # CPU 轻任务：RAG/OCR。CPU 即可，不申请 GPU 租约，可 backfill


class DeviceAffinity(Enum):
    """设备亲和性"""
    GPU_REQUIRED = "gpu_required"      # 必须 GPU（边缘可用时延定义）
    GPU_PREFERRED = "gpu_preferred"    # 偏好 GPU，可降级 CPU
    CPU_ONLY = "cpu_only"             # 只用 CPU
    CPU_PREFERRED = "cpu_preferred"   # 偏好 CPU（如小模型 TTS）


class LLMServiceLevel(Enum):
    """LLM 服务等级（Degradation-Aware Scheduling 的三态）

    不是简单的 "在 GPU / 在 CPU" 二元状态，
    而是 GPU_NORMAL / CPU_DEGRADED / RESTORING 三态切换。
    """
    GPU_NORMAL = "gpu_normal"        # GPU 正常服务
    CPU_DEGRADED = "cpu_degraded"    # CPU 降级服务（迁移后）
    RESTORING = "restoring"          # 正在回迁 GPU


@dataclass
class PhaseAttributes:
    """阶段属性：描述一个阶段任务的资源与调度特性

    这是 Phase-Aware 模型的核心。每个任务类型映射到一组属性，
    调度器基于属性做决策，而不是基于具体任务名。
    """
    role: PhaseRole
    device_affinity: DeviceAffinity
    preemptible: bool = False          # 是否可被高优先级任务抢占
    queueable: bool = True             # 是否可排队等待
    migratable: bool = False           # 是否可迁移（GPU↔CPU）
    vram_usage_mb: int = 0             # 显存占用预估（MB）
    typical_duration_s: float = 1.0    # 典型执行时长（秒）
    is_downstream: bool = False        # 是否是下游阶段（如 TTS 在 LLM 之后）
    is_upstream: bool = False          # 是否是上游阶段（如 RAG 在 LLM 之前）
    can_overlap_with_gpu: bool = False # 是否可与 GPU 独占任务并行（CPU 任务）


@dataclass
class LLMState:
    """LLM 服务状态快照

    用于迁移决策。记录当前服务等级、设备、迁移历史。
    """
    level: LLMServiceLevel = LLMServiceLevel.GPU_NORMAL
    current_device: str = "gpu"                # "gpu" / "cpu"
    last_migration_time: float = 0.0
    migration_count: int = 0
    total_degraded_time_s: float = 0.0
    last_degraded_start: float = 0.0

    def enter_degraded(self) -> None:
        """进入降级状态"""
        self.level = LLMServiceLevel.CPU_DEGRADED
        self.current_device = "cpu"
        self.last_migration_time = time.time()
        self.last_degraded_start = time.time()
        self.migration_count += 1

    def enter_restoring(self) -> None:
        """进入回迁状态

        注意：回迁期间 LLM 仍处于"非 GPU 正常"状态，对用户而言仍是降级。
        因此在这里累加降级时间，避免 enter_normal 时因 level 已变而漏算。
        """
        if self.level == LLMServiceLevel.CPU_DEGRADED and self.last_degraded_start > 0:
            self.total_degraded_time_s += time.time() - self.last_degraded_start
            self.last_degraded_start = 0.0
        self.level = LLMServiceLevel.RESTORING

    def enter_normal(self) -> None:
        """恢复正常状态"""
        # 兜底：如果没经过 RESTORING 直接回 NORMAL（异常路径），也补算降级时间
        if self.level == LLMServiceLevel.CPU_DEGRADED and self.last_degraded_start > 0:
            self.total_degraded_time_s += time.time() - self.last_degraded_start
            self.last_degraded_start = 0.0
        self.level = LLMServiceLevel.GPU_NORMAL
        self.current_device = "gpu"


@dataclass
class PhaseTask:
    """阶段任务：一个具体的阶段执行实例

    每个任务属于一个 workflow，有自己的属性、优先级、执行回调。
    """
    task_id: str = field(default_factory=lambda: f"phase_{uuid.uuid4().hex[:10]}")
    phase_name: str = ""                          # e.g. "rag", "llm", "tts", "image"
    attributes: Optional[PhaseAttributes] = None
    priority: int = 1                             # 0=CRITICAL, 1=HIGH, 2=MEDIUM, 3=LOW
    submit_time: float = field(default_factory=time.time)
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    status: str = "pending"                       # pending/running/completed/failed/cancelled
    # workflow 关联
    workflow_id: Optional[str] = None
    phase_index: int = 0                          # 在 workflow 中的阶段索引
    depends_on: List[str] = field(default_factory=list)  # 依赖的阶段名
    # 执行回调：返回结果
    execute_fn: Optional[Callable[[], Awaitable[Any]]] = None
    # 执行结果
    result: Any = None
    error: Optional[str] = None
    # 实际执行时长（由调度器填充）
    actual_duration_s: float = 0.0
    # 实际执行设备
    executed_on_device: str = ""                  # "gpu" / "cpu"

    @property
    def needs_gpu(self) -> bool:
        """是否需要 GPU（必需或偏好）"""
        if self.attributes is None:
            return False
        return self.attributes.device_affinity in (
            DeviceAffinity.GPU_REQUIRED,
            DeviceAffinity.GPU_PREFERRED,
        )

    @property
    def is_cpu_affine(self) -> bool:
        """是否是 CPU 亲和任务（可 backfill）"""
        if self.attributes is None:
            return False
        return self.attributes.device_affinity in (
            DeviceAffinity.CPU_ONLY,
            DeviceAffinity.CPU_PREFERRED,
        )


@dataclass
class Workflow:
    """多阶段 workflow：描述一个完整的本地 AI 助手请求流

    例如：RAG 检索 -> LLM 生成 -> TTS 播报 + 图像生成（并行）
    """
    workflow_id: str = field(default_factory=lambda: f"wf_{uuid.uuid4().hex[:10]}")
    phases: List[PhaseTask] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    # workflow 级别状态
    status: str = "pending"                       # pending/running/completed/failed
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    def add_phase(self, task: PhaseTask) -> None:
        """添加阶段"""
        task.workflow_id = self.workflow_id
        task.phase_index = len(self.phases)
        self.phases.append(task)

    def get_phase(self, phase_name: str) -> Optional[PhaseTask]:
        """按名称获取阶段"""
        for p in self.phases:
            if p.phase_name == phase_name:
                return p
        return None

    @property
    def total_duration_s(self) -> float:
        """workflow 总执行时长（端到端）"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0


# ============================================================
# 阶段预设：常见本地 AI 助手任务的属性映射
# ============================================================

PHASE_PRESETS: Dict[str, PhaseAttributes] = {
    "llm": PhaseAttributes(
        role=PhaseRole.RESIDENT_MIGRATABLE,
        device_affinity=DeviceAffinity.GPU_PREFERRED,
        preemptible=False,
        queueable=False,
        migratable=True,
        vram_usage_mb=6700,           # 4.7GB 权重 + 1-2GB KV Cache
        typical_duration_s=3.0,       # 流式生成，但单次响应约 3s
        is_downstream=False,
        is_upstream=False,
        can_overlap_with_gpu=False,
    ),
    "image": PhaseAttributes(
        role=PhaseRole.BURST_GPU_PINNED,
        device_affinity=DeviceAffinity.GPU_REQUIRED,
        preemptible=False,            # GPU 任务不可硬中断
        queueable=True,
        migratable=False,
        vram_usage_mb=6700,           # SDXL Turbo fp16
        typical_duration_s=7.0,       # 512x512, 20 steps
        is_downstream=False,
        is_upstream=False,
        can_overlap_with_gpu=False,
    ),
    "tts": PhaseAttributes(
        role=PhaseRole.BURST_GPU_PREFERRED,
        device_affinity=DeviceAffinity.CPU_PREFERRED,  # 0.6B 小模型 CPU 可接受
        preemptible=True,
        queueable=True,
        migratable=True,
        vram_usage_mb=1500,           # Qwen3-TTS 0.6B
        typical_duration_s=2.0,
        is_downstream=True,           # 在 LLM 之后
        is_upstream=False,
        can_overlap_with_gpu=True,    # 可与图像并行（走 CPU）
    ),
    "rag": PhaseAttributes(
        role=PhaseRole.BURST_CPU_LIGHT,
        device_affinity=DeviceAffinity.CPU_ONLY,
        preemptible=True,
        queueable=True,
        migratable=False,
        vram_usage_mb=0,
        typical_duration_s=0.8,       # embedding 检索
        is_downstream=False,
        is_upstream=True,             # 在 LLM 之前
        can_overlap_with_gpu=True,
    ),
    "ocr": PhaseAttributes(
        role=PhaseRole.BURST_CPU_LIGHT,
        device_affinity=DeviceAffinity.CPU_ONLY,
        preemptible=True,
        queueable=True,
        migratable=False,
        vram_usage_mb=0,
        typical_duration_s=1.2,
        is_downstream=False,
        is_upstream=True,
        can_overlap_with_gpu=True,
    ),
    "vlm": PhaseAttributes(
        role=PhaseRole.BURST_GPU_PREFERRED,
        device_affinity=DeviceAffinity.GPU_PREFERRED,
        preemptible=True,
        queueable=True,
        migratable=True,
        vram_usage_mb=3500,           # 小型 VLM
        typical_duration_s=2.5,
        is_downstream=False,
        is_upstream=False,
        can_overlap_with_gpu=False,
    ),
}


def make_phase_task(
    phase_name: str,
    execute_fn: Optional[Callable[[], Awaitable[Any]]] = None,
    priority: int = 1,
    depends_on: Optional[List[str]] = None,
) -> PhaseTask:
    """便捷工厂：根据 phase_name 从预设创建 PhaseTask"""
    attrs = PHASE_PRESETS.get(phase_name)
    if attrs is None:
        raise ValueError(f"未知阶段名: {phase_name}，可用: {list(PHASE_PRESETS.keys())}")
    return PhaseTask(
        phase_name=phase_name,
        attributes=attrs,
        priority=priority,
        execute_fn=execute_fn,
        depends_on=depends_on or [],
    )
