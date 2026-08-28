"""
实验 1：Always-Migrate vs Cost-Aware Avoidance

对比两种迁移策略：
- always_migrate: 来了就迁（v1 行为）
- cost_aware: 代价感知（v2 行为，GPU_PREFERRED/CPU 任务不迁移）

场景：提交混合 burst 任务流（image + vlm + rag）
- always_migrate: 每个 GPU 任务（image + vlm）都迁移 LLM
- cost_aware: 只有 GPU_REQUIRED（image）迁移，VLM 降级 CPU，RAG 旁路

核心差异：
- GPU_PREFERRED 任务（VLM）在两种策略下的处理不同
  - always_migrate: 进入 GPU burst 队列，触发 LLM 迁移
  - cost_aware: 降级 CPU，走 backfill，不迁移 LLM

指标：
- 迁移次数（cost_aware 应显著更少）
- 总耗时（cost_aware 应更短，避免不必要迁移）
- LLM 降级时间（cost_aware 应更短）
- 决策分布（bypass/migrate/wait）

输出：experiment_results/phase_aware/exp1_migration_avoidance.json
"""

import asyncio
import json
import os
import sys
import time
import statistics
from dataclasses import dataclass, asdict
from typing import Dict, Any, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from services.phase_aware import (
    PhaseAwareScheduler,
    SchedulerConfig,
    SchedulerPolicy,
    MigrationPolicy,
    make_phase_task,
    MigrationCost,
)


# ============================================================
# 实验配置
# ============================================================

# 时长缩放因子：真实场景时长 * TIME_SCALE = 实验时长
# 用于加速实验，同时保持各阶段时长与迁移代价的比例关系
TIME_SCALE = 0.3

# 真实迁移代价（基于实测，已按 TIME_SCALE 缩放以保持比例）
REAL_MIGRATION_COST = MigrationCost(
    kv_cache_save_ms=250 * TIME_SCALE,
    vram_release_ms=100 * TIME_SCALE,
    cpu_load_ms=1500 * TIME_SCALE,
    restore_ms=2000 * TIME_SCALE,
    service_gap_ms=300 * TIME_SCALE,
)
# 真实迁移代价（用于论文报告）
REAL_MIGRATION_COST_FULL = MigrationCost(
    kv_cache_save_ms=250,
    vram_release_ms=100,
    cpu_load_ms=1500,
    restore_ms=2000,
    service_gap_ms=300,
)

# 任务时长（mock，基于真实场景预估，已按比例缩短以加速实验）
TASK_DURATIONS = {
    "image": 7.0 * TIME_SCALE,   # SDXL Turbo 512x512 20 steps
    "vlm": 2.5 * TIME_SCALE,     # 小型 VLM 推理
    "tts": 2.0 * TIME_SCALE,     # Qwen3-TTS 0.6B CPU
    "rag": 0.8 * TIME_SCALE,     # embedding 检索
    "ocr": 1.2 * TIME_SCALE,     # OCR
}
# 真实时长（用于论文报告）
REAL_TASK_DURATIONS = {
    "image": 7.0,
    "vlm": 2.5,
    "tts": 2.0,
    "rag": 0.8,
    "ocr": 1.2,
}

REPEAT_COUNT = 3   # 重复次数
RANDOM_SEED = 42


@dataclass
class Exp1Result:
    """实验 1 结果"""
    strategy: str
    repeat: int
    total_time_s: float
    migration_count: int
    llm_degraded_time_s: float
    decision_stats: Dict[str, Any]
    backfill_stats: Dict[str, Any]
    task_count: int


async def mock_execute(duration_s: float, label: str = ""):
    """模拟任务执行"""
    await asyncio.sleep(duration_s)
    return f"done({label})"


def build_burst_workload() -> List[Dict]:
    """构建混合 burst 工作负载

    模拟用户在聊天中同时发起多个请求：
    - 1 个 image（GPU_REQUIRED，两种策略都迁移）
    - 2 个 vlm（GPU_PREFERRED，always_migrate 迁移 / cost_aware 降级 CPU）
    - 2 个 rag（CPU_ONLY，两种策略都旁路）
    """
    return [
        {"phase": "image", "duration": TASK_DURATIONS["image"]},
        {"phase": "vlm", "duration": TASK_DURATIONS["vlm"]},
        {"phase": "vlm", "duration": TASK_DURATIONS["vlm"]},
        {"phase": "rag", "duration": TASK_DURATIONS["rag"]},
        {"phase": "rag", "duration": TASK_DURATIONS["rag"]},
    ]


async def run_single_experiment(
    strategy: str,
    workload: List[Dict],
) -> Exp1Result:
    """运行单次实验"""
    if strategy == "always_migrate":
        config = SchedulerConfig(
            policy=SchedulerPolicy.V1_DUAL_ROLE,
            migration_policy=MigrationPolicy.ALWAYS_MIGRATE,
            migration_cost=REAL_MIGRATION_COST,
            max_concurrent_cpu=2,
        )
    else:  # cost_aware
        config = SchedulerConfig(
            policy=SchedulerPolicy.V2_PHASE_AWARE,
            migration_policy=MigrationPolicy.COST_AWARE,
            migration_cost=REAL_MIGRATION_COST,
            max_concurrent_cpu=2,
        )

    scheduler = PhaseAwareScheduler(config)
    await scheduler.start()

    # 并行提交所有任务
    tasks = []
    for item in workload:
        phase = item["phase"]
        duration = item["duration"]
        task = make_phase_task(
            phase,
            execute_fn=lambda d=duration, l=phase: mock_execute(d, l),
        )
        tasks.append(scheduler.submit_phase(task))

    t0 = time.time()
    await asyncio.gather(*tasks)
    total_time = time.time() - t0

    stats = scheduler.get_stats()
    result = Exp1Result(
        strategy=strategy,
        repeat=0,
        total_time_s=total_time,
        migration_count=stats["llm_migration_count"],
        llm_degraded_time_s=stats["llm_total_degraded_time_s"],
        decision_stats=stats["decision_stats"],
        backfill_stats=stats["backfill_stats"],
        task_count=stats["total_tasks"],
    )

    await scheduler.stop()
    return result


async def run_experiment():
    """运行完整实验"""
    print("=" * 60)
    print("实验 1：Always-Migrate vs Cost-Aware Avoidance")
    print("=" * 60)

    workload = build_burst_workload()
    print(f"\n工作负载: {len(workload)} 个任务")
    for item in workload:
        print(f"  - {item['phase']}: {item['duration']}s")

    strategies = ["always_migrate", "cost_aware"]
    all_results: Dict[str, List[Exp1Result]] = {s: [] for s in strategies}

    for strategy in strategies:
        print(f"\n--- 策略: {strategy} ---")
        for i in range(REPEAT_COUNT):
            result = await run_single_experiment(strategy, workload)
            result.repeat = i
            all_results[strategy].append(result)
            print(f"  轮次 {i+1}: 耗时={result.total_time_s:.2f}s, "
                  f"迁移={result.migration_count}, "
                  f"降级时间={result.llm_degraded_time_s:.2f}s")

    # 汇总统计
    summary = {}
    for strategy, results in all_results.items():
        times = [r.total_time_s for r in results]
        migrations = [r.migration_count for r in results]
        degraded = [r.llm_degraded_time_s for r in results]

        summary[strategy] = {
            "total_time_s": {
                "mean": statistics.mean(times),
                "stdev": statistics.stdev(times) if len(times) > 1 else 0,
                "min": min(times),
                "max": max(times),
            },
            "migration_count": {
                "mean": statistics.mean(migrations),
                "min": min(migrations),
                "max": max(migrations),
            },
            "llm_degraded_time_s": {
                "mean": statistics.mean(degraded),
                "min": min(degraded),
                "max": max(degraded),
            },
            "decision_stats": results[-1].decision_stats,
            "backfill_stats": results[-1].backfill_stats,
        }

    # 计算改进
    if "always_migrate" in summary and "cost_aware" in summary:
        am_time = summary["always_migrate"]["total_time_s"]["mean"]
        ca_time = summary["cost_aware"]["total_time_s"]["mean"]
        am_mig = summary["always_migrate"]["migration_count"]["mean"]
        ca_mig = summary["cost_aware"]["migration_count"]["mean"]
        am_deg = summary["always_migrate"]["llm_degraded_time_s"]["mean"]
        ca_deg = summary["cost_aware"]["llm_degraded_time_s"]["mean"]

        summary["improvement"] = {
            "time_reduction_pct": (am_time - ca_time) / am_time * 100 if am_time > 0 else 0,
            "migration_reduction_pct": (am_mig - ca_mig) / am_mig * 100 if am_mig > 0 else 0,
            "degraded_time_reduction_pct": (am_deg - ca_deg) / am_deg * 100 if am_deg > 0 else 0,
        }

    # 输出结果
    output = {
        "experiment": "exp1_migration_avoidance",
        "description": "Always-Migrate vs Cost-Aware Avoidance",
        "config": {
            "repeat_count": REPEAT_COUNT,
            "time_scale": TIME_SCALE,
            "migration_cost_scaled": REAL_MIGRATION_COST.to_dict(),
            "migration_cost_real": REAL_MIGRATION_COST_FULL.to_dict(),
            "task_durations_scaled": TASK_DURATIONS,
            "task_durations_real": REAL_TASK_DURATIONS,
            "workload": workload,
        },
        "summary": summary,
        "raw_results": {
            strategy: [asdict(r) for r in results]
            for strategy, results in all_results.items()
        },
    }

    # 保存结果
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "experiment_results", "phase_aware")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "exp1_migration_avoidance.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # 打印汇总
    print("\n" + "=" * 60)
    print("实验 1 汇总")
    print("=" * 60)
    for strategy in strategies:
        s = summary[strategy]
        print(f"\n{strategy}:")
        print(f"  总耗时: {s['total_time_s']['mean']:.2f}s ± {s['total_time_s']['stdev']:.2f}s")
        print(f"  迁移次数: {s['migration_count']['mean']:.1f}")
        print(f"  LLM 降级时间: {s['llm_degraded_time_s']['mean']:.2f}s")
        print(f"  决策统计: {s['decision_stats']}")

    if "improvement" in summary:
        imp = summary["improvement"]
        print(f"\nCost-Aware 改进:")
        print(f"  耗时降低: {imp['time_reduction_pct']:.1f}%")
        print(f"  迁移减少: {imp['migration_reduction_pct']:.1f}%")
        print(f"  降级时间减少: {imp['degraded_time_reduction_pct']:.1f}%")

    print(f"\n结果已保存: {output_path}")
    return output


if __name__ == "__main__":
    asyncio.run(run_experiment())
