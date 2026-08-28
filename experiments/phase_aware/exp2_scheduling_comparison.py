"""
实验 2：Sequential / V1 Dual-Role / V2 Phase-Aware Backfilling 对比

对比三种调度策略：
- sequential: 所有任务串行执行（baseline，无并行无迁移）
- v1_dual_role: 二元互斥，来了就迁，CPU 任务不 backfill（串行）
- v2_phase_aware: phase-aware + cost-aware + backfilling（最优）

场景：并行提交混合任务流（image + vlm + rag + tts）
- sequential: 所有任务串行，总时间 = 所有任务时长之和
- v1_dual_role: GPU 任务迁移串行，CPU 任务也串行（不 backfill）
- v2_phase_aware: GPU 任务迁移，CPU 任务 backfill 并行

核心展示：
- v1 → v2 的进化：从"非此即彼"到"phase-aware 互补"
- backfill 的价值：隐藏 CPU 任务的等待时间

指标：
- 总耗时（v2 应最短）
- 迁移次数（sequential=0, v1>v2）
- LLM 降级时间
- backfill 隐藏的时间（v2 独有）

输出：experiment_results/phase_aware/exp2_scheduling_comparison.json
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

TIME_SCALE = 0.3

# 迁移代价（按 TIME_SCALE 缩放）
MIGRATION_COST = MigrationCost(
    kv_cache_save_ms=250 * TIME_SCALE,
    vram_release_ms=100 * TIME_SCALE,
    cpu_load_ms=1500 * TIME_SCALE,
    restore_ms=2000 * TIME_SCALE,
    service_gap_ms=300 * TIME_SCALE,
)
MIGRATION_COST_FULL = MigrationCost(
    kv_cache_save_ms=250,
    vram_release_ms=100,
    cpu_load_ms=1500,
    restore_ms=2000,
    service_gap_ms=300,
)

# 任务时长
TASK_DURATIONS = {
    "image": 7.0 * TIME_SCALE,
    "vlm": 2.5 * TIME_SCALE,
    "tts": 2.0 * TIME_SCALE,
    "rag": 0.8 * TIME_SCALE,
}
REAL_TASK_DURATIONS = {
    "image": 7.0,
    "vlm": 2.5,
    "tts": 2.0,
    "rag": 0.8,
}

REPEAT_COUNT = 3


@dataclass
class Exp2Result:
    """实验 2 结果"""
    strategy: str
    repeat: int
    total_time_s: float
    migration_count: int
    llm_degraded_time_s: float
    backfill_hidden_time_s: float
    task_count: int
    gpu_task_count: int
    cpu_task_count: int
    backfilled_task_count: int


async def mock_execute(duration_s: float, label: str = ""):
    """模拟任务执行"""
    await asyncio.sleep(duration_s)
    return f"done({label})"


def build_mixed_workload() -> List[Dict]:
    """构建混合工作负载

    模拟用户同时发起多种请求：
    - 1 image (GPU_REQUIRED)
    - 1 vlm (GPU_PREFERRED)
    - 2 tts (CPU_PREFERRED)
    - 2 rag (CPU_ONLY)
    """
    return [
        {"phase": "image", "duration": TASK_DURATIONS["image"]},
        {"phase": "vlm", "duration": TASK_DURATIONS["vlm"]},
        {"phase": "tts", "duration": TASK_DURATIONS["tts"]},
        {"phase": "tts", "duration": TASK_DURATIONS["tts"]},
        {"phase": "rag", "duration": TASK_DURATIONS["rag"]},
        {"phase": "rag", "duration": TASK_DURATIONS["rag"]},
    ]


def make_config(strategy: str) -> SchedulerConfig:
    """根据策略名创建调度器配置"""
    if strategy == "sequential":
        return SchedulerConfig(
            policy=SchedulerPolicy.SEQUENTIAL,
            migration_policy=MigrationPolicy.COST_AWARE,  # 不影响，sequential 不迁移
            migration_cost=MIGRATION_COST,
            max_concurrent_cpu=2,
        )
    elif strategy == "v1_dual_role":
        return SchedulerConfig(
            policy=SchedulerPolicy.V1_DUAL_ROLE,
            migration_policy=MigrationPolicy.ALWAYS_MIGRATE,
            migration_cost=MIGRATION_COST,
            max_concurrent_cpu=2,  # 不影响，v1 不 backfill
        )
    else:  # v2_phase_aware
        return SchedulerConfig(
            policy=SchedulerPolicy.V2_PHASE_AWARE,
            migration_policy=MigrationPolicy.COST_AWARE,
            migration_cost=MIGRATION_COST,
            max_concurrent_cpu=2,
        )


async def run_single_experiment(strategy: str, workload: List[Dict]) -> Exp2Result:
    """运行单次实验"""
    config = make_config(strategy)
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
    result = Exp2Result(
        strategy=strategy,
        repeat=0,
        total_time_s=total_time,
        migration_count=stats["llm_migration_count"],
        llm_degraded_time_s=stats["llm_total_degraded_time_s"],
        backfill_hidden_time_s=stats["backfill_stats"].get("total_backfill_time_hidden_s", 0.0),
        task_count=stats["total_tasks"],
        gpu_task_count=stats["gpu_tasks"],
        cpu_task_count=stats["cpu_tasks"],
        backfilled_task_count=stats["backfilled_tasks"],
    )

    await scheduler.stop()
    return result


async def run_experiment():
    """运行完整实验"""
    print("=" * 60)
    print("实验 2：Sequential / V1 Dual-Role / V2 Phase-Aware 对比")
    print("=" * 60)

    workload = build_mixed_workload()
    print(f"\n工作负载: {len(workload)} 个任务")
    for item in workload:
        print(f"  - {item['phase']}: {item['duration']:.2f}s")

    strategies = ["sequential", "v1_dual_role", "v2_phase_aware"]
    all_results: Dict[str, List[Exp2Result]] = {s: [] for s in strategies}

    for strategy in strategies:
        print(f"\n--- 策略: {strategy} ---")
        for i in range(REPEAT_COUNT):
            result = await run_single_experiment(strategy, workload)
            result.repeat = i
            all_results[strategy].append(result)
            print(f"  轮次 {i+1}: 耗时={result.total_time_s:.2f}s, "
                  f"迁移={result.migration_count}, "
                  f"降级={result.llm_degraded_time_s:.2f}s, "
                  f"backfill隐藏={result.backfill_hidden_time_s:.2f}s")

    # 汇总统计
    summary = {}
    for strategy, results in all_results.items():
        times = [r.total_time_s for r in results]
        migrations = [r.migration_count for r in results]
        degraded = [r.llm_degraded_time_s for r in results]
        hidden = [r.backfill_hidden_time_s for r in results]

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
            "backfill_hidden_time_s": {
                "mean": statistics.mean(hidden),
                "min": min(hidden),
                "max": max(hidden),
            },
            "task_count": results[-1].task_count,
            "gpu_task_count": results[-1].gpu_task_count,
            "cpu_task_count": results[-1].cpu_task_count,
            "backfilled_task_count": results[-1].backfilled_task_count,
        }

    # 计算改进（相对 sequential baseline）
    if "sequential" in summary:
        seq_time = summary["sequential"]["total_time_s"]["mean"]
        for strategy in ["v1_dual_role", "v2_phase_aware"]:
            if strategy in summary:
                s_time = summary[strategy]["total_time_s"]["mean"]
                summary[strategy]["speedup_vs_sequential"] = seq_time / s_time if s_time > 0 else 0
                summary[strategy]["time_reduction_pct"] = (seq_time - s_time) / seq_time * 100 if seq_time > 0 else 0

    # v2 vs v1 改进
    if "v1_dual_role" in summary and "v2_phase_aware" in summary:
        v1_time = summary["v1_dual_role"]["total_time_s"]["mean"]
        v2_time = summary["v2_phase_aware"]["total_time_s"]["mean"]
        summary["v2_vs_v1"] = {
            "speedup": v1_time / v2_time if v2_time > 0 else 0,
            "time_reduction_pct": (v1_time - v2_time) / v1_time * 100 if v1_time > 0 else 0,
        }

    # 输出结果
    output = {
        "experiment": "exp2_scheduling_comparison",
        "description": "Sequential / V1 Dual-Role / V2 Phase-Aware Backfilling",
        "config": {
            "repeat_count": REPEAT_COUNT,
            "time_scale": TIME_SCALE,
            "migration_cost_scaled": MIGRATION_COST.to_dict(),
            "migration_cost_real": MIGRATION_COST_FULL.to_dict(),
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
    output_path = os.path.join(output_dir, "exp2_scheduling_comparison.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # 打印汇总
    print("\n" + "=" * 60)
    print("实验 2 汇总")
    print("=" * 60)
    for strategy in strategies:
        s = summary[strategy]
        print(f"\n{strategy}:")
        print(f"  总耗时: {s['total_time_s']['mean']:.2f}s ± {s['total_time_s']['stdev']:.2f}s")
        print(f"  迁移次数: {s['migration_count']['mean']:.1f}")
        print(f"  LLM 降级时间: {s['llm_degraded_time_s']['mean']:.2f}s")
        print(f"  backfill 隐藏时间: {s['backfill_hidden_time_s']['mean']:.2f}s")
        print(f"  GPU任务: {s['gpu_task_count']}, CPU任务: {s['cpu_task_count']}, "
              f"backfill任务: {s['backfilled_task_count']}")
        if "speedup_vs_sequential" in s:
            print(f"  vs sequential: 加速比 {s['speedup_vs_sequential']:.2f}x, "
                  f"耗时降低 {s['time_reduction_pct']:.1f}%")

    if "v2_vs_v1" in summary:
        v2v1 = summary["v2_vs_v1"]
        print(f"\nv2 vs v1: 加速比 {v2v1['speedup']:.2f}x, 耗时降低 {v2v1['time_reduction_pct']:.1f}%")

    print(f"\n结果已保存: {output_path}")
    return output


if __name__ == "__main__":
    asyncio.run(run_experiment())
