# EdgeSched 实验指南

本目录包含实验框架、论文主实验、语义验证、压力测试和图表生成脚本。实验的第一原则是区分 Mock、真实 LLM、真实 Forge 和论文正文数据。

## 目录索引

| 目录 | 内容 |
|---|---|
| `framework/` | 路径、配置、统计、随机种子、环境采集、Mock/真实适配器 |
| `migration/` | EdgeSched 非对称迁移、对称迁移和消融策略 |
| `baselines/` | cold start、LRU thrashing、manual switching |
| `admission/` | 准入策略和动态任务画像对比 |
| `recovery/` | 崩溃恢复主实验 |
| `semantic_validation/` | 取消语义、StateBackplane 恢复语义 |
| `stress/` | 大规模与真实混合负载压力 |
| `workload/` | 真实交互 workload 和 trace |
| `phase_aware/` | 迁移避免、策略对比和 N 阶段 workflow |
| `overhead/` | 实现开销、综合实验、论文消融套件 |
| `charts/` | JSON 到 PNG/PDF/CSV/LaTeX 图表 |
| `scaling/` | Forge batch scaling 特性曲线 (T(b)/吞吐/加速比) |

## 推荐入口

### 1. 无 Forge 实验

```bash
python experiments/run_no_forge_experiments.py --iterations 5
python experiments/run_no_forge_experiments.py \
  --experiments lru_thrashing,crash_recovery --iterations 5
```

集合包含：`lru_thrashing`、`crash_recovery`、`stress_test`、`baseline_comparison`。

### 2. Forge 实验

```bash
python experiments/run_forge_experiments.py --iterations 5
python experiments/run_forge_experiments.py --experiments migration --iterations 5
```

集合包含：`edgesched_migration`、`symmetric_migration`、`fast_down_naive_up`、`naive_down_optimized_up`、`real_workload`、`realistic_stress`。

### 3. 统一论文编排

```bash
python experiments/run_all_experiments.py --experiments all --iterations 5
python experiments/run_all_experiments.py --experiments core --iterations 5
```

`run_all_experiments.py` 的 `all` 当前包含迁移、cold start、LRU、恢复、压力、实现开销和真实 workload；Forge 不可用时不要使用 `--skip-forge-check` 把回退结果当作真实结果。

### 4. 性能套件

```bash
python experiments/run_performance_suite.py --workload mock
python experiments/run_performance_suite.py --workload real
```

Mock 用于验证异步链路、统计字段、背压和回归；Real 才用于真实吞吐、尾延迟、显存峰值和 trace。运行时产物默认写入 `experiment_results/`。

### 5. Phase-Aware

```bash
python experiments/phase_aware/exp1_migration_avoidance.py
python experiments/phase_aware/exp2_scheduling_comparison.py
python experiments/phase_aware/exp3_n_phase_workflow.py
```

`exp1_real_migration_avoidance.py` 涉及真实迁移，必须在 WSL2 模型环境单独核对，不要与纯策略模拟混合报告。

## 环境变量

高频变量包括：

- `XIAOYOU_LLM_MODEL_PATH`：论文实验 GGUF 路径。
- `XIAOYOU_BENCH_LLM_PATH`：benchmark 适配器 GGUF 路径。
- `XIAOYOU_BENCH_N_CTX`、`XIAOYOU_BENCH_N_GPU_LAYERS`：LLM 运行参数。
- `XIAOYOU_FORGE_BASE_URL`：Forge API，通常是本机/WSL2 的 7860 端口。
- `XIAOYOU_USE_CPP_SCHEDULER`、`XIAOYOU_USE_CPP_FOR_LLM`、`XIAOYOU_USE_CPP_FOR_IMAGE`：C++ 路径开关。
- `XIAOYOU_ALLOW_CPP_LLM_WORKER`：显式允许实验性 C++ LLM worker。
- `XIAOYOU_ALLOW_MOCK_FORGE`：只用于受控调试；启用后结果不得晋升为真实 Forge 证据。

完整环境和路径见 [项目上下文](../docs/PROJECT_CONTEXT.md#7-环境与同步边界)。

## 结果与论文数据

```text
experiment_results/                      当前工作区运行产物
paper/experiment_data/03_raw_iterations/ 原始迭代
paper/experiment_data/02_auxiliary/      辅助数据
paper/experiment_data/01_paper_main/     当前论文正文聚合数据
paper/figures/                            当前图表
paper/tables/                             当前 LaTeX 表格
```

正文数据晋升必须满足：真实后端未回退、参数冻结、多次迭代完整、manifest 存在、聚合口径一致、图表重建成功、正文引用可追溯。不要从 `paper/previous/` 或 `archive_legacy_*` 复制结果覆盖正文。

## 快速验证

```powershell
..\..\venv_core\Scripts\python.exe -m pytest \
  tests\test_chat_service.py tests\test_phase_aware_quick.py -q

..\..\venv_core\Scripts\python.exe tests\verify_paper_experiments.py
```

真实实验请在 `/home/leslie/mvp_core` 中激活 `/home/leslie/.venvs/mvp_core-cu128` 后运行，并保存完整日志和 manifest。

