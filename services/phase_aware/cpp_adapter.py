"""
Phase-Aware C++ 引擎对接适配器

将 Python 层的 PhaseAwareScheduler 与 C++ 调度引擎对接：
1. 通过 scheduler_py 模块创建 C++ Phase-Aware 策略
2. 注入真实迁移回调（offload_llm_to_cpu / restore_llm_to_gpu）
3. 提供统一接口，让实验脚本无需关心底层实现

使用方式：
    from services.phase_aware.cpp_adapter import create_cpp_phase_scheduler
    scheduler = create_cpp_phase_scheduler(config)
"""

import os
import sys
import asyncio
import logging
from typing import Optional, Callable, Awaitable

from services.phase_aware import (
    PhaseAwareScheduler,
    SchedulerConfig,
    SchedulerPolicy,
    MigrationPolicy,
    MigrationCost,
)

logger = logging.getLogger(__name__)


def _try_import_scheduler_py():
    """尝试导入 C++ 调度器绑定模块"""
    try:
        import scheduler_py
        return scheduler_py
    except ImportError:
        # 尝试通过 wrapper 加载
        try:
            from services import cpp_scheduler_wrapper
            return cpp_scheduler_wrapper.get_module()
        except (ImportError, AttributeError):
            return None


def _try_get_cpp_engine():
    """尝试获取 C++ 调度引擎单例"""
    try:
        from services.cpp_scheduler_engine import cpp_scheduler_engine
        if cpp_scheduler_engine.is_initialized():
            return cpp_scheduler_engine
    except (ImportError, AttributeError):
        pass
    return None


def create_cpp_phase_scheduler(
    config: SchedulerConfig,
    cpp_engine=None,
) -> PhaseAwareScheduler:
    """创建对接 C++ 引擎的 Phase-Aware 调度器

    Args:
        config: 调度器配置
        cpp_engine: C++ 调度引擎实例（可选，默认自动获取）

    Returns:
        PhaseAwareScheduler，已注入真实迁移回调
    """
    scheduler = PhaseAwareScheduler(config)

    # 尝试获取 C++ 引擎
    if cpp_engine is None:
        cpp_engine = _try_get_cpp_engine()

    if cpp_engine is not None:
        # 注入真实迁移回调
        async def migrate_fn(urgent=False):
            """真实迁移：LLM GPU→CPU"""
            try:
                result = await asyncio.get_running_loop().run_in_executor(
                    None, lambda: cpp_engine.offload_llm_to_cpu(urgent=urgent)
                )
                logger.info(f"LLM 迁移到 CPU 完成 (urgent={urgent}): {result}")
                return result
            except Exception as e:
                logger.error(f"LLM 迁移到 CPU 失败: {e}")
                return False

        async def restore_fn():
            """真实回迁：LLM CPU→GPU"""
            try:
                result = await asyncio.get_running_loop().run_in_executor(
                    None, lambda: cpp_engine.restore_llm_to_gpu()
                )
                logger.info(f"LLM 回迁到 GPU 完成: {result}")
                return result
            except Exception as e:
                logger.error(f"LLM 回迁到 GPU 失败: {e}")
                return False

        scheduler.set_migration_callbacks(migrate_fn, restore_fn)
        logger.info("Phase-Aware 调度器已对接 C++ 引擎（真实迁移回调）")
    else:
        logger.warning("未找到 C++ 引擎，使用 mock 迁移回调")

    # 尝试设置 C++ 层的 Phase-Aware 策略
    scheduler_py = _try_import_scheduler_py()
    if scheduler_py is not None and cpp_engine is not None:
        try:
            # 创建 C++ Phase-Aware 配置
            cpp_config = scheduler_py.PhaseAwareConfig()
            cpp_config.policy = {
                SchedulerPolicy.SEQUENTIAL: scheduler_py.SchedulerPolicy.SEQUENTIAL,
                SchedulerPolicy.V1_DUAL_ROLE: scheduler_py.SchedulerPolicy.V1_DUAL_ROLE,
                SchedulerPolicy.V2_PHASE_AWARE: scheduler_py.SchedulerPolicy.V2_PHASE_AWARE,
            }.get(config.policy, scheduler_py.SchedulerPolicy.V2_PHASE_AWARE)

            cpp_config.migration_policy = {
                MigrationPolicy.ALWAYS_MIGRATE: scheduler_py.MigrationPolicy.ALWAYS_MIGRATE,
                MigrationPolicy.COST_AWARE: scheduler_py.MigrationPolicy.COST_AWARE,
            }.get(config.migration_policy, scheduler_py.MigrationPolicy.COST_AWARE)

            cpp_config.vram_total_mb = config.vram_total_mb
            cpp_config.migration_cost.kv_cache_save_ms = config.migration_cost.kv_cache_save_ms
            cpp_config.migration_cost.vram_release_ms = config.migration_cost.vram_release_ms
            cpp_config.migration_cost.cpu_load_ms = config.migration_cost.cpu_load_ms
            cpp_config.migration_cost.restore_ms = config.migration_cost.restore_ms
            cpp_config.migration_cost.service_gap_ms = config.migration_cost.service_gap_ms

            # 创建 C++ 策略并注入到 C++ 调度器
            cpp_strategy = scheduler_py.DefaultPhaseAwareStrategy(cpp_config)
            cpp_scheduler = cpp_engine.get_scheduler()
            if cpp_scheduler is not None:
                cpp_scheduler.setPhaseStrategy(cpp_strategy)
                logger.info("C++ Phase-Aware 策略已设置")
        except Exception as e:
            logger.warning(f"设置 C++ Phase-Aware 策略失败: {e}")

    return scheduler
