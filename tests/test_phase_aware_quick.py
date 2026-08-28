"""
Phase-Aware 调度器快速功能测试

验证核心流程：
1. 调度器启动/停止
2. GPU burst 任务执行 + 迁移决策
3. CPU backfill 任务并行执行
4. 决策引擎 cost-aware vs always-migrate
"""

import asyncio
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.phase_aware import (
    PhaseAwareScheduler,
    SchedulerConfig,
    SchedulerPolicy,
    MigrationPolicy,
    PhaseRole,
    make_phase_task,
    PHASE_PRESETS,
    MigrationCost,
)


def small_cost() -> MigrationCost:
    """小迁移代价（用于快速测试）"""
    return MigrationCost(
        kv_cache_save_ms=20,
        vram_release_ms=10,
        cpu_load_ms=50,
        restore_ms=80,
        service_gap_ms=10,
    )


async def mock_execute(duration_s: float, label: str = ""):
    """模拟任务执行"""
    await asyncio.sleep(duration_s)
    return f"done({label})"


async def test_basic_gpu_burst():
    """测试 1：单个 GPU burst 任务触发迁移"""
    print("\n=== 测试 1：单个 GPU burst 任务 ===")
    scheduler = PhaseAwareScheduler(SchedulerConfig(
        policy=SchedulerPolicy.V2_PHASE_AWARE,
        migration_policy=MigrationPolicy.COST_AWARE,
        migration_cost=small_cost(),
    ))
    await scheduler.start()

    # 提交一个 image 任务（长 burst，应该触发迁移）
    task = make_phase_task("image", execute_fn=lambda: mock_execute(0.3, "image"))
    t0 = time.time()
    result = await scheduler.submit_phase(task)
    elapsed = time.time() - t0

    print(f"  结果: {result}")
    print(f"  耗时: {elapsed:.2f}s")
    print(f"  LLM 迁移次数: {scheduler.llm_state.migration_count}")
    print(f"  决策统计: {scheduler.decision_engine.get_stats()}")

    await scheduler.stop()
    assert result == "done(image)"
    assert scheduler.llm_state.migration_count == 1
    print("  ✓ 通过")


async def test_cpu_backfill_parallel():
    """测试 2：CPU 任务与 GPU 任务并行（backfill）"""
    print("\n=== 测试 2：CPU backfill 与 GPU 并行 ===")
    scheduler = PhaseAwareScheduler(SchedulerConfig(
        policy=SchedulerPolicy.V2_PHASE_AWARE,
        migration_policy=MigrationPolicy.COST_AWARE,
        max_concurrent_cpu=2,
        migration_cost=small_cost(),
    ))
    await scheduler.start()

    # 同时提交：image（GPU）+ rag + tts（CPU）
    image_task = make_phase_task("image", execute_fn=lambda: mock_execute(0.5, "image"))
    rag_task = make_phase_task("rag", execute_fn=lambda: mock_execute(0.3, "rag"))
    tts_task = make_phase_task("tts", execute_fn=lambda: mock_execute(0.4, "tts"))

    t0 = time.time()
    # 并行提交
    results = await asyncio.gather(
        scheduler.submit_phase(image_task),
        scheduler.submit_phase(rag_task),
        scheduler.submit_phase(tts_task),
    )
    elapsed = time.time() - t0

    print(f"  结果: {results}")
    print(f"  总耗时: {elapsed:.2f}s")
    print(f"  Backfill 统计: {scheduler.backfill_scheduler.get_stats().to_dict()}")

    stats = scheduler.get_stats()
    print(f"  GPU 任务数: {stats['gpu_tasks']}, CPU 任务数: {stats['cpu_tasks']}")

    await scheduler.stop()
    # 总耗时 ≈ 迁移(0.08s) + image(0.5s) + 回迁(0.08s) ≈ 0.66s
    # rag/tts 在 backfill 并行，应被隐藏
    assert elapsed < 1.0
    print("  ✓ 通过")


async def test_cost_aware_vs_always_migrate():
    """测试 3：cost-aware 对短 burst 不迁移，always-migrate 总是迁移"""
    print("\n=== 测试 3：cost-aware vs always-migrate ===")

    # cost-aware：rag（CPU-affine）不迁移，tts（GPU_PREFERRED）降级 CPU 不迁移
    cost_aware = PhaseAwareScheduler(SchedulerConfig(
        migration_policy=MigrationPolicy.COST_AWARE,
        migration_cost=small_cost(),
    ))
    await cost_aware.start()

    rag_task = make_phase_task("rag", execute_fn=lambda: mock_execute(0.1, "rag"))
    await cost_aware.submit_phase(rag_task)
    print(f"  cost-aware (rag): 迁移次数={cost_aware.llm_state.migration_count}")
    assert cost_aware.llm_state.migration_count == 0

    tts_task = make_phase_task("tts", execute_fn=lambda: mock_execute(0.1, "tts"))
    await cost_aware.submit_phase(tts_task)
    print(f"  cost-aware (tts): 迁移次数={cost_aware.llm_state.migration_count}")
    assert cost_aware.llm_state.migration_count == 0  # TTS 降级 CPU，不迁移
    await cost_aware.stop()

    # always-migrate：image 触发迁移
    always_mig = PhaseAwareScheduler(SchedulerConfig(
        migration_policy=MigrationPolicy.ALWAYS_MIGRATE,
        migration_cost=small_cost(),
    ))
    await always_mig.start()
    img_task = make_phase_task("image", execute_fn=lambda: mock_execute(0.1, "image"))
    await always_mig.submit_phase(img_task)
    print(f"  always-migrate (image): 迁移次数={always_mig.llm_state.migration_count}")
    assert always_mig.llm_state.migration_count == 1
    await always_mig.stop()

    print("  ✓ 通过")


async def test_workflow_n_phase():
    """测试 4：N 阶段 workflow（RAG -> LLM -> TTS + Image 并行）"""
    print("\n=== 测试 4：N 阶段 workflow ===")
    from services.phase_aware import Workflow

    scheduler = PhaseAwareScheduler(SchedulerConfig(
        policy=SchedulerPolicy.V2_PHASE_AWARE,
        migration_policy=MigrationPolicy.COST_AWARE,
        migration_cost=small_cost(),
    ))
    await scheduler.start()

    wf = Workflow()
    # RAG（上游，CPU）-> LLM（GPU）-> TTS（下游，CPU）+ Image（GPU，并行）
    wf.add_phase(make_phase_task("rag", execute_fn=lambda: mock_execute(0.2, "rag")))
    wf.add_phase(make_phase_task("llm", execute_fn=lambda: mock_execute(0.3, "llm"), depends_on=["rag"]))
    wf.add_phase(make_phase_task("tts", execute_fn=lambda: mock_execute(0.3, "tts"), depends_on=["llm"]))
    wf.add_phase(make_phase_task("image", execute_fn=lambda: mock_execute(0.4, "image"), depends_on=["llm"]))

    t0 = time.time()
    result = await scheduler.submit_workflow(wf)
    elapsed = time.time() - t0

    print(f"  workflow 结果: {result}")
    print(f"  总耗时: {elapsed:.2f}s")
    print(f"  统计: {scheduler.get_stats()}")

    await scheduler.stop()
    print("  ✓ 通过")


async def main():
    print("Phase-Aware 调度器功能测试")
    await test_basic_gpu_burst()
    await test_cpu_backfill_parallel()
    await test_cost_aware_vs_always_migrate()
    await test_workflow_n_phase()
    print("\n=== 所有测试通过 ===")


if __name__ == "__main__":
    asyncio.run(main())
