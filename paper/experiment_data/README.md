# EdgeSched 实验数据与复现指南

> **最后更新**: 2026-05-05
> **实验环境**: WSL2 (Ubuntu-24.04-E), RTX 5070 Laptop (8GB), Ryzen 9 8940HX, 32GB DDR5

---

## 1. 文件夹结构

本目录保存论文当前稿件使用的数据：正文聚合结果主要位于 `01_paper_main/`，辅助结果位于 `02_auxiliary/`，原始迭代结果位于 `03_raw_iterations/`。

```
experiment_data/
├── 01_paper_main/                       # 论文正文直接引用的聚合结果
│   ├── migration/                       # 迁移主实验与消融聚合结果
│   ├── crash_recovery_comparison.json   # 崩溃恢复对比实验
│   ├── baseline_cold_start_results.json # 冷启动 vs. Atomic Hot-Swap
│   ├── baseline_lru_thrashing_results.json
│   ├── realistic_stress_test.json       # 10 分钟真实混合负载压力测试
│   └── real_workload_trace.json         # 真实交互会话轨迹
├── 02_auxiliary/                        # 辅助说明性实验
├── 03_raw_iterations/                   # 原始迭代结果
├── figures/                             # 生成的论文图表 (PDF/PNG)
└── tables/                              # 生成的 LaTeX 表格 (.tex)
```

---

## 2. 论文图表数据源映射

当前仓库保留了论文使用的最终图表与表格产物。

| 论文图表 (Evaluation Section) | 对应数据文件 | 说明 |
|-------------------|--------------|------|
| Migration figures/tables | `01_paper_main/migration/*.json` | 主迁移实验与消融聚合结果 |
| Crash recovery figure/table | `01_paper_main/crash_recovery_comparison.json` | 有/无 StateBackplane 的恢复语义对比 |
| Cold-start recovery figure/table | `01_paper_main/baseline_cold_start_results.json` | 冷重启 vs. Atomic Hot-Swap |
| LRU-style baseline table | `01_paper_main/baseline_lru_thrashing_results.json` | OS 级分页/LRU 风格行为 |
| Real workload trace figure | `01_paper_main/real_workload_trace.json` | 真实交互会话时间线 |
| Realistic stress figure | `01_paper_main/realistic_stress_test.json` | 10 分钟真实混合负载压力测试 |
| Mock overhead reference | `02_auxiliary/baseline_comparison.json` | 仅用于说明实现开销 |

---

## 3. 关键实验数据摘要

### 资源迁移 (Resource Migration) - N=5 均值
> 数据来源: `01_paper_main/migration/edgesched_aggregated.json`

*   **GPU TTFT (Baseline)**: `112.9 ms` (±12.2 ms)
    *   *说明*: 在图像生成模型驻留显存情况下的基线性能。
*   **Migration Stutter**: `8774.3 ms` (±1768.8 ms)
    *   *说明*: 突发任务抢占 GPU 时的服务中断时间。
*   **CPU TTFT (Coexistence)**: `200.0 ms` (±19.6 ms)
    *   *说明*: 迁移到 CPU 后的稳态响应速度。
*   **Restore Gap**: `0.0 ms`
    *   *说明*: 回迁过程中服务完全无中断 (Atomic Hot-Swap)。
*   **Queue Wait Time**: `10024.8 ms`
    *   *说明*: 互斥锁机制生效，第二项任务正确排队等待。

### 崩溃恢复 (Crash Recovery)
> 数据来源: `crash_recovery_comparison.json`

*   **Recovery Rate**: `100%` (vs 0% without backplane)
*   **Task Loss**: `0` tasks
*   **Recovery Overhead**: `~3 ms` 的额外启动开销；达到语义可用恢复约为 `44 ms`

---

## 4. 复现与图表生成

### 生成图表
当前仓库仅保留已生成的图表结果；如果后续重跑实验，建议以 `01_paper_main/` 中的聚合 JSON 为正文同步基准。

```bash
# figure/table generation script is maintained outside this workspace
```

### 数据更新流程
1.  如果重新运行实验，请将新生成的 JSON 结果覆盖到 `01_paper_main/` 对应文件。
2.  重新生成 `figures/` 与 `tables/` 中的产物。
3.  检查 `figures/` 下的新图表。
