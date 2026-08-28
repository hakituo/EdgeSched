"""
Phase-Aware Local AI Scheduling 模块

把本地 AI 助手建模成多阶段 workflow，根据阶段依赖、资源亲和性和迁移代价，
决定哪些阶段串行、哪些阶段并行、什么时候迁移。

核心组件：
- phase_model: 阶段属性、角色、任务定义
- gpu_lease: GPU 使用权租约管理
- cost_model: 迁移代价函数 + 决策引擎
- backfill_scheduler: CPU 任务 backfilling
- phase_scheduler: 主调度器
- cpp_adapter: C++ 引擎对接适配器
"""

from .phase_model import (
    PhaseRole,
    DeviceAffinity,
    PhaseAttributes,
    PhaseTask,
    Workflow,
    LLMServiceLevel,
    LLMState,
    PHASE_PRESETS,
    make_phase_task,
)
from .gpu_lease import GPULease, LeaseType, GPULeaseManager
from .cost_model import (
    MigrationCost,
    MigrationDecision,
    MigrationDecisionEngine,
)
from .backfill_scheduler import BackfillScheduler
from .phase_scheduler import (
    PhaseAwareScheduler,
    SchedulerPolicy,
    SchedulerConfig,
    MigrationPolicy,
)

__all__ = [
    "PhaseRole",
    "DeviceAffinity",
    "PhaseAttributes",
    "PhaseTask",
    "Workflow",
    "LLMServiceLevel",
    "LLMState",
    "PHASE_PRESETS",
    "make_phase_task",
    "GPULease",
    "LeaseType",
    "GPULeaseManager",
    "MigrationCost",
    "MigrationDecision",
    "MigrationDecisionEngine",
    "BackfillScheduler",
    "PhaseAwareScheduler",
    "SchedulerPolicy",
    "SchedulerConfig",
    "MigrationPolicy",
]

# 延迟导入 cpp_adapter（避免在无 C++ 引擎时报错）
def create_cpp_phase_scheduler(config, cpp_engine=None):
    """创建对接 C++ 引擎的 Phase-Aware 调度器"""
    from .cpp_adapter import create_cpp_phase_scheduler as _create
    return _create(config, cpp_engine)
