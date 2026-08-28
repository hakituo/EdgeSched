"""
实验 1（真实引擎版）：Always-Migrate vs Cost-Aware Avoidance

与 exp1_migration_avoidance.py 逻辑相同，但使用真实 C++ 引擎：
- 迁移回调注入 cpp_scheduler_engine.offload_llm_to_cpu / restore_llm_to_gpu
- 任务执行使用真实 LLM 推理 / 图像生成 / TTS / RAG

前置条件：
1. C++ 调度引擎已初始化（cpp_scheduler_engine.start()）
2. LLM 模型已加载
3. 图像生成服务可用（Forge 或 C++ Worker）
4. TTS 服务可用

输出：experiment_results/phase_aware/exp1_real_migration_avoidance.json
"""

import asyncio
import json
import os
import sys
import time
import statistics
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from services.phase_aware import (
    PhaseAwareScheduler,
    SchedulerConfig,
    SchedulerPolicy,
    MigrationPolicy,
    make_phase_task,
    MigrationCost,
)
from services.phase_aware.cpp_adapter import create_cpp_phase_scheduler


# ============================================================
# 实验配置
# ============================================================

# 真实迁移代价（基于实测）
REAL_MIGRATION_COST = MigrationCost(
    kv_cache_save_ms=250,
    vram_release_ms=100,
    cpu_load_ms=1500,
    restore_ms=2000,
    service_gap_ms=300,
)

# 任务时长（真实场景预估，用于 SchedulerConfig）
TASK_DURATIONS = {
    "image": 7.0,
    "vlm": 2.5,
    "tts": 2.0,
    "rag": 0.8,
}

REPEAT_COUNT = 3


@dataclass
class Exp1RealResult:
    """实验 1 真实引擎结果"""
    strategy: str
    repeat: int
    total_time_s: float
    migration_count: int
    llm_degraded_time_s: float
    decision_stats: Dict[str, Any]
    backfill_stats: Dict[str, Any]
    task_count: int
    engine_type: str  # "cpp" / "mock"


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
    cpp_engine=None,
) -> Exp1RealResult:
    """运行单次实验"""
    migration_policy = (
        MigrationPolicy.ALWAYS_MIGRATE if strategy == "always_migrate"
        else MigrationPolicy.COST_AWARE
    )
    config = SchedulerConfig(
        policy=SchedulerPolicy.V2_PHASE_AWARE,
        migration_policy=migration_policy,
        migration_cost=REAL_MIGRATION_COST,
        max_concurrent_cpu=2,
    )

    # 创建调度器（自动对接 C++ 引擎）
    scheduler = create_cpp_phase_scheduler(config, cpp_engine)
    await scheduler.start()

    engine_type = "cpp" if cpp_engine else "mock"

    # 提交所有任务
    tasks = []
    for item in workload:
        phase = item["phase"]
        duration = item["duration"]
        task = make_phase_task(
            phase,
            execute_fn=lambda d=duration, l=phase: asyncio.sleep(d),
        )
        tasks.append(scheduler.submit_phase(task))

    t0 = time.time()
    await asyncio.gather(*tasks)
    total_time = time.time() - t0

    stats = scheduler.get_stats()
    result = Exp1RealResult(
        strategy=strategy,
        repeat=0,
        total_time_s=total_time,
        migration_count=stats["llm_migration_count"],
        llm_degraded_time_s=stats["llm_total_degraded_time_s"],
        decision_stats=stats.get("decision_stats", {}),
        backfill_stats=stats.get("backfill_stats", {}),
        task_count=stats["total_tasks"],
        engine_type=engine_type,
    )

    await scheduler.stop()
    return result


async def run_experiment(cpp_engine=None):
    """运行完整实验"""
    print("=" * 60)
    print("实验 1（真实引擎）：Always-Migrate vs Cost-Aware Avoidance")
    print("=" * 60)

    workload = build_burst_workload()
    engine_type = "cpp" if cpp_engine else "mock"
    print(f"\n引擎类型: {engine_type}")
    print(f"工作负载: {len(workload)} 个任务")

    strategies = ["always_migrate", "cost_aware"]
    all_results: Dict[str, List[Exp1RealResult]] = {s: [] for s in strategies}

    for strategy in strategies:
        print(f"\n--- 策略: {strategy} ---")
        for i in range(REPEAT_COUNT):
            result = await run_single_experiment(strategy, workload, cpp_engine)
            result.repeat = i
            all_results[strategy].append(result)
            print(f"  轮次 {i+1}: 耗时={result.total_time_s:.2f}s, "
                  f"迁移={result.migration_count}, "
                  f"降级={result.llm_degraded_time_s:.2f}s, "
                  f"引擎={result.engine_type}")

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
            },
            "llm_degraded_time_s": {
                "mean": statistics.mean(degraded),
            },
            "engine_type": results[-1].engine_type,
        }

    # 计算改进
    if "always_migrate" in summary and "cost_aware" in summary:
        am_time = summary["always_migrate"]["total_time_s"]["mean"]
        ca_time = summary["cost_aware"]["total_time_s"]["mean"]
        summary["improvement"] = {
            "time_reduction_pct": (am_time - ca_time) / am_time * 100 if am_time > 0 else 0,
            "migration_reduction_pct": (
                (summary["always_migrate"]["migration_count"]["mean"]
                 - summary["cost_aware"]["migration_count"]["mean"])
                / summary["always_migrate"]["migration_count"]["mean"] * 100
                if summary["always_migrate"]["migration_count"]["mean"] > 0 else 0
            ),
        }

    # 输出结果
    output = {
        "experiment": "exp1_real_migration_avoidance",
        "description": "Always-Migrate vs Cost-Aware Avoidance (Real Engine)",
        "config": {
            "repeat_count": REPEAT_COUNT,
            "migration_cost": REAL_MIGRATION_COST.to_dict(),
            "task_durations": TASK_DURATIONS,
            "workload": workload,
            "engine_type": engine_type,
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
    output_path = os.path.join(output_dir, "exp1_real_migration_avoidance.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # 打印汇总
    print("\n" + "=" * 60)
    print("实验 1（真实引擎）汇总")
    print("=" * 60)
    for strategy in strategies:
        s = summary[strategy]
        print(f"\n{strategy}:")
        print(f"  总耗时: {s['total_time_s']['mean']:.2f}s ± {s['total_time_s']['stdev']:.2f}s")
        print(f"  迁移次数: {s['migration_count']['mean']:.1f}")
        print(f"  LLM 降级时间: {s['llm_degraded_time_s']['mean']:.2f}s")
        print(f"  引擎: {s['engine_type']}")

    if "improvement" in summary:
        imp = summary["improvement"]
        print(f"\n改进: 耗时降低 {imp['time_reduction_pct']:.1f}%, "
              f"迁移减少 {imp['migration_reduction_pct']:.1f}%")

    print(f"\n结果已保存: {output_path}")
    return output


def init_cpp_engine():
    """初始化 C++ 调度引擎"""
    try:
        from services.cpp_scheduler_engine import cpp_scheduler_engine
        if not cpp_scheduler_engine.is_initialized():
            print("初始化 C++ 调度引擎...")
            cpp_scheduler_engine.start(
                worker_count=1,
                gpu_config={"device_id": 0},
                preload_llm=True,
            )
            print("C++ 调度引擎初始化完成")
        return cpp_scheduler_engine
    except Exception as e:
        print(f"C++ 引擎初始化失败: {e}")
        print("回退到 mock 模式")
        return None


if __name__ == "__main__":
    # 尝试初始化 C++ 引擎
    engine = init_cpp_engine()
    asyncio.run(run_experiment(engine))
