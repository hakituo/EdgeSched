"""
实验 3：N 阶段 Workflow 场景（RAG/OCR -> LLM -> TTS + Image）

直接回应导师第 2 点：泛化到 N 个任务。

构建完整的多阶段 workflow：
  阶段1（上游，并行）: RAG + OCR（CPU，为 LLM 提供上下文）
  阶段2（中间）: LLM（依赖 RAG+OCR，GPU，生成回复）
  阶段3（下游，并行）: TTS + Image（依赖 LLM，TTS 走 CPU，Image 走 GPU）

对比三种策略在 N 阶段 workflow 下的表现：
- sequential: 所有阶段串行（RAG→OCR→LLM→TTS→Image）
- v1_dual_role: 二元互斥，CPU 任务不 backfill
- v2_phase_aware: phase-aware + cost-aware + backfill

核心展示：
- N 阶段 workflow 的资源互补性：RAG/OCR（CPU）可与 LLM 前置并行，
  TTS（CPU）可与 Image（GPU）后置并行
- v2 的 phase-aware 调度能自动发现并行机会，v1 不能

指标：
- 端到端耗时（v2 应最短）
- 各阶段执行轨迹（展示并行/串行）
- 迁移次数、backfill 隐藏时间

输出：experiment_results/phase_aware/exp3_n_phase_workflow.json
"""

import asyncio
import json
import os
import sys
import time
import statistics
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from services.phase_aware import (
    PhaseAwareScheduler,
    SchedulerConfig,
    SchedulerPolicy,
    MigrationPolicy,
    make_phase_task,
    MigrationCost,
    Workflow,
    PhaseTask,
)


# ============================================================
# 实验配置
# ============================================================

TIME_SCALE = 0.3

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

# 各阶段时长
PHASE_DURATIONS = {
    "rag": 0.8 * TIME_SCALE,    # RAG 检索
    "ocr": 1.2 * TIME_SCALE,    # OCR 识别
    "llm": 3.0 * TIME_SCALE,    # LLM 生成
    "tts": 2.0 * TIME_SCALE,    # TTS 合成
    "image": 7.0 * TIME_SCALE,  # 图像生成
}
REAL_PHASE_DURATIONS = {
    "rag": 0.8,
    "ocr": 1.2,
    "llm": 3.0,
    "tts": 2.0,
    "image": 7.0,
}

REPEAT_COUNT = 3


@dataclass
class Exp3Result:
    """实验 3 结果"""
    strategy: str
    repeat: int
    total_time_s: float
    migration_count: int
    llm_degraded_time_s: float
    backfill_hidden_time_s: float
    phase_timeline: List[Dict[str, Any]] = field(default_factory=list)


async def mock_execute(duration_s: float, label: str = ""):
    """模拟任务执行"""
    await asyncio.sleep(duration_s)
    return f"done({label})"


def build_n_phase_workflow() -> Workflow:
    """构建 N 阶段 workflow

    结构：
      阶段1（上游，并行）: RAG + OCR → 为 LLM 提供上下文
      阶段2（中间）: LLM → 依赖 RAG+OCR，生成回复
      阶段3（下游，并行）: TTS + Image → 依赖 LLM，输出结果

    这体现了"至少 RAG/OCR -> LLM -> TTS + Image"的 N 阶段场景。
    """
    wf = Workflow()

    # 阶段1：上游 CPU 任务（并行）
    wf.add_phase(make_phase_task(
        "rag",
        execute_fn=lambda: mock_execute(PHASE_DURATIONS["rag"], "rag"),
    ))
    wf.add_phase(make_phase_task(
        "ocr",
        execute_fn=lambda: mock_execute(PHASE_DURATIONS["ocr"], "ocr"),
    ))

    # 阶段2：LLM（依赖 rag + ocr）
    wf.add_phase(make_phase_task(
        "llm",
        execute_fn=lambda: mock_execute(PHASE_DURATIONS["llm"], "llm"),
        depends_on=["rag", "ocr"],
    ))

    # 阶段3：下游并行（TTS 走 CPU，Image 走 GPU）
    wf.add_phase(make_phase_task(
        "tts",
        execute_fn=lambda: mock_execute(PHASE_DURATIONS["tts"], "tts"),
        depends_on=["llm"],
    ))
    wf.add_phase(make_phase_task(
        "image",
        execute_fn=lambda: mock_execute(PHASE_DURATIONS["image"], "image"),
        depends_on=["llm"],
    ))

    return wf


def make_config(strategy: str) -> SchedulerConfig:
    """根据策略名创建调度器配置"""
    if strategy == "sequential":
        return SchedulerConfig(
            policy=SchedulerPolicy.SEQUENTIAL,
            migration_policy=MigrationPolicy.COST_AWARE,
            migration_cost=MIGRATION_COST,
            max_concurrent_cpu=2,
        )
    elif strategy == "v1_dual_role":
        return SchedulerConfig(
            policy=SchedulerPolicy.V1_DUAL_ROLE,
            migration_policy=MigrationPolicy.ALWAYS_MIGRATE,
            migration_cost=MIGRATION_COST,
            max_concurrent_cpu=2,
        )
    else:  # v2_phase_aware
        return SchedulerConfig(
            policy=SchedulerPolicy.V2_PHASE_AWARE,
            migration_policy=MigrationPolicy.COST_AWARE,
            migration_cost=MIGRATION_COST,
            max_concurrent_cpu=2,
        )


def extract_timeline(scheduler: PhaseAwareScheduler) -> List[Dict[str, Any]]:
    """提取执行轨迹（用于绘制甘特图）"""
    traces = scheduler.get_all_traces()
    timeline = []
    for t in traces:
        timeline.append({
            "task_id": t.task_id,
            "phase": t.phase_name,
            "strategy": t.strategy,
            "start_time": round(t.start_time, 4),
            "end_time": round(t.end_time, 4),
            "duration_s": round(t.duration_s, 4),
            "device": t.executed_on_device,
            "backfilled": t.backfilled,
        })
    # 按开始时间排序
    timeline.sort(key=lambda x: x["start_time"])
    return timeline


async def run_single_experiment(strategy: str) -> Exp3Result:
    """运行单次实验"""
    config = make_config(strategy)
    scheduler = PhaseAwareScheduler(config)
    await scheduler.start()

    workflow = build_n_phase_workflow()

    t0 = time.time()
    result = await scheduler.submit_workflow(workflow)
    total_time = time.time() - t0

    stats = scheduler.get_stats()
    timeline = extract_timeline(scheduler)

    exp_result = Exp3Result(
        strategy=strategy,
        repeat=0,
        total_time_s=total_time,
        migration_count=stats["llm_migration_count"],
        llm_degraded_time_s=stats["llm_total_degraded_time_s"],
        backfill_hidden_time_s=stats["backfill_stats"].get("total_backfill_time_hidden_s", 0.0),
        phase_timeline=timeline,
    )

    await scheduler.stop()
    return exp_result


async def run_experiment():
    """运行完整实验"""
    print("=" * 60)
    print("实验 3：N 阶段 Workflow（RAG/OCR -> LLM -> TTS + Image）")
    print("=" * 60)

    workflow = build_n_phase_workflow()
    print(f"\nWorkflow 结构（{len(workflow.phases)} 个阶段）:")
    for p in workflow.phases:
        deps = p.depends_on if p.depends_on else "无"
        print(f"  - {p.phase_name}: 依赖={deps}, "
              f"设备亲和={p.attributes.device_affinity.value}, "
              f"角色={p.attributes.role.value}")

    print(f"\n各阶段时长（缩放后）:")
    for name, dur in PHASE_DURATIONS.items():
        print(f"  - {name}: {dur:.2f}s（真实 {REAL_PHASE_DURATIONS[name]:.1f}s）")

    strategies = ["sequential", "v1_dual_role", "v2_phase_aware"]
    all_results: Dict[str, List[Exp3Result]] = {s: [] for s in strategies}

    for strategy in strategies:
        print(f"\n--- 策略: {strategy} ---")
        for i in range(REPEAT_COUNT):
            result = await run_single_experiment(strategy)
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
            },
            "llm_degraded_time_s": {
                "mean": statistics.mean(degraded),
            },
            "backfill_hidden_time_s": {
                "mean": statistics.mean(hidden),
            },
            # 保留最后一轮的执行轨迹用于绘制甘特图
            "timeline_last_run": results[-1].phase_timeline,
        }

    # 计算改进
    if "sequential" in summary:
        seq_time = summary["sequential"]["total_time_s"]["mean"]
        for strategy in ["v1_dual_role", "v2_phase_aware"]:
            if strategy in summary:
                s_time = summary[strategy]["total_time_s"]["mean"]
                summary[strategy]["speedup_vs_sequential"] = seq_time / s_time if s_time > 0 else 0
                summary[strategy]["time_reduction_pct"] = (seq_time - s_time) / seq_time * 100 if seq_time > 0 else 0

    if "v1_dual_role" in summary and "v2_phase_aware" in summary:
        v1_time = summary["v1_dual_role"]["total_time_s"]["mean"]
        v2_time = summary["v2_phase_aware"]["total_time_s"]["mean"]
        summary["v2_vs_v1"] = {
            "speedup": v1_time / v2_time if v2_time > 0 else 0,
            "time_reduction_pct": (v1_time - v2_time) / v1_time * 100 if v1_time > 0 else 0,
        }

    # 输出结果
    output = {
        "experiment": "exp3_n_phase_workflow",
        "description": "N-Phase Workflow: RAG/OCR -> LLM -> TTS + Image",
        "config": {
            "repeat_count": REPEAT_COUNT,
            "time_scale": TIME_SCALE,
            "migration_cost_scaled": MIGRATION_COST.to_dict(),
            "migration_cost_real": MIGRATION_COST_FULL.to_dict(),
            "phase_durations_scaled": PHASE_DURATIONS,
            "phase_durations_real": REAL_PHASE_DURATIONS,
            "workflow_structure": [
                {
                    "phase": p.phase_name,
                    "depends_on": p.depends_on,
                    "device_affinity": p.attributes.device_affinity.value,
                    "role": p.attributes.role.value,
                }
                for p in workflow.phases
            ],
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
    output_path = os.path.join(output_dir, "exp3_n_phase_workflow.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # 打印汇总
    print("\n" + "=" * 60)
    print("实验 3 汇总")
    print("=" * 60)
    for strategy in strategies:
        s = summary[strategy]
        print(f"\n{strategy}:")
        print(f"  总耗时: {s['total_time_s']['mean']:.2f}s ± {s['total_time_s']['stdev']:.2f}s")
        print(f"  迁移次数: {s['migration_count']['mean']:.1f}")
        print(f"  LLM 降级时间: {s['llm_degraded_time_s']['mean']:.2f}s")
        print(f"  backfill 隐藏时间: {s['backfill_hidden_time_s']['mean']:.2f}s")
        if "speedup_vs_sequential" in s:
            print(f"  vs sequential: 加速比 {s['speedup_vs_sequential']:.2f}x, "
                  f"耗时降低 {s['time_reduction_pct']:.1f}%")

    if "v2_vs_v1" in summary:
        v2v1 = summary["v2_vs_v1"]
        print(f"\nv2 vs v1: 加速比 {v2v1['speedup']:.2f}x, 耗时降低 {v2v1['time_reduction_pct']:.1f}%")

    # 打印 v2 的执行轨迹（展示并行）
    print(f"\n--- v2_phase_aware 执行轨迹（最后一轮）---")
    v2_timeline = summary["v2_phase_aware"]["timeline_last_run"]
    for t in v2_timeline:
        print(f"  [{t['start_time']:.2f}-{t['end_time']:.2f}] {t['phase']:6s} "
              f"on {t['device']:3s} (strategy={t['strategy']}, "
              f"backfill={t['backfilled']})")

    print(f"\n结果已保存: {output_path}")
    return output


if __name__ == "__main__":
    asyncio.run(run_experiment())
