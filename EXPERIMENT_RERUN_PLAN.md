# MVP Core 待重跑实验清单与覆盖性评估

## 当前口径（已保留）

目前 `experiment_results/` 只保留“动态资源迁移”四项可直接引用的真实指标源文件：

- `experiment_results/supplementary/resource_migration_all.json`

四项真实指标：
- `llm_gpu_ttft_ms`
- `llm_cpu_ttft_ms`
- `migration_time_ms`
- `restore_time_ms`

## 真实模型 vs Mock：按论文主线的取舍

你的论文主线是“单 GPU 资源受限下，多模态混合负载的隔离/控制/恢复”，其中：

- 需要真实模型的，只有“模型本体开销会改变结论”的部分：TTFT、加载/卸载、迁移/回迁时间窗等
- 只验证调度控制面语义/状态机语义的部分，用 Mock 更科学：更可复现、变量更少、不会把模型性能误写成调度性能

### 必须用真实模型跑（否则关键数值不成立）

- 动态资源迁移（`resource_migration_experiment.py`）
  - 原因：TTFT 与迁移/回迁耗时本质是“真实加载/卸载 + 真实推理栈”带来的端到端现象，Mock 无法替代
  - 论文里可直接引用的真实指标：`llm_gpu_ttft_ms`、`llm_cpu_ttft_ms`、`migration_time_ms`、`restore_time_ms`

### 建议只用 Mock 跑（论文结论是语义正确性，不依赖真实推理）

- 崩溃恢复（`crash_recovery_experiment.py`）
  - 原因：要证明的是“有背板可恢复任务状态、无背板必丢任务 + Running 闭合”，不需要真实推理
  - 真实 LLM 会引入额外变量（模型路径、加载耗时、CUDA/CPU 环境差异），反而削弱结论可信度

- 状态背板验证（`state_backplane_recovery_experiment.py`）
  - 原因：纯 mmap 读写一致性/延迟基准，与模型无关

- 压力测试（`large_scale_stress_test.py`）
  - 原因：要验证调度控制面在高并发下的稳定性/背压行为与资源监控，不需要真实模型

- 基线对比（`baseline_comparison.py`）
  - 原因：目的是解释“纯调度开销”，用 Mock 任务更公平（避免模型把对比淹没）

- 调度器开销（`tests/verify_scheduler_overhead.py` 的不 preload 口径）
  - 原因：`import/start` 纯调度器成本与模型无关，应该固定为 Mock/不加载模型

### 两阶段（先 Mock 验证语义，必要时再用真实模型补“数值”）

- 取消语义（`cancellation_semantics_experiment.py`）
  - 语义验证（必须做）：Mock/真实都行，重点是“取消后不再输出 + 迟到结果丢弃”
  - 数值引用（可选补强）：如果论文要写“LLM 流式取消延迟是 XX ms”，那一轮必须在明确启用真实 LLM 后端下跑；脚本已落盘 `backend_info` 用于证明口径

- 调度器开销（`tests/verify_scheduler_overhead.py --preload`）
  - 如果要给出“模型 preload 的时间/RSS 增量”可再跑一次 `--preload`（这部分属于模型开销，不是调度器本体）

## 待重跑实验清单（生成结果文件）

### 1) 取消语义（核心）

- 脚本：`legacy/mvp_core/experiments/cancellation/cancellation_semantics_experiment.py`
- 目标：验证 LLM 流式取消 + 图像队列取消/迟到结果丢弃
- 结果文件：`experiment_results/core/cancellation_semantics.json`
- 覆盖点：已记录 `backend_info`（平台/Python/调度后端/模型路径存在性），可直接用于区分 Real vs Mock 的口径

### 2) 崩溃恢复（核心 + 对照）

- 脚本：`legacy/mvp_core/experiments/recovery/crash_recovery_experiment.py`
- 目标：验证有/无状态背板的恢复语义（有背板可恢复并闭合运行中任务；无背板必丢崩溃前任务）
- 结果文件：
  - `experiment_results/core/crash_recovery_with_backplane.json`
  - `experiment_results/pending/crash_recovery_without_backplane.json`
  - `experiment_results/supplementary/crash_recovery_comparison.json`（使用 `--comparison` 生成）
- 风险点（建议先改脚本再跑）：
  - 子进程里默认模型路径硬编码为 Linux 路径，跨平台不可复现
  - 为了制造 RUNNING 窗口而在子进程里加载 LLM（`llama_cpp`）成本过高且引入额外变量；恢复语义实验不需要真实推理
  - 会生成 `crash_test_*_state_backplane.dat` 临时文件，容易污染结果目录

### 3) 状态背板验证（核心）

- 脚本：`legacy/mvp_core/experiments/backplane/state_backplane_recovery_experiment.py`
- 目标：验证 mmap 持久化一致性 + 基准读写性能 + 恢复后状态一致性
- 结果文件：`experiment_results/core/state_backplane_recovery.json`
- 风险点（建议先改脚本再跑）：会生成带时间戳的 `.dat` 临时文件，建议跑完自动清理或转移到临时目录

### 4) 压力测试（补充）

- 脚本：`legacy/mvp_core/experiments/scalability/large_scale_stress_test.py`
- 目标：高并发/长时间稳定性（控制面、背压、资源监控）
- 结果文件：`experiment_results/supplementary/large_scale_stress_test.json`
- 风险点：会产生日志文件；建议统一“只保留 JSON，日志按需留档”

### 5) 基线对比（补充）

- 脚本：`legacy/mvp_core/experiments/comparison/baseline_comparison.py`
- 目标：Naive Async / ThreadPool / Semaphore vs EdgeSched 的调度开销与收益解释
- 结果文件：`experiment_results/supplementary/baseline_comparison.json`

### 6) 调度器开销（补充，但很关键的口径文件）

- 脚本：`legacy/mvp_core/tests/verify_scheduler_overhead.py`
- 目标：分离“调度器引入成本”与“模型加载成本”（import/start/preload 的时间与 RSS 增量）
- 结果文件：`experiment_results/supplementary/scheduler_overhead.json`
- 注意：需要确保使用与论文主实验一致的 Python 版本与同一套运行环境口径

## 覆盖性结论（脚本能否覆盖论文需求）

可以覆盖的需求（脚本已存在且指标口径可落地）：
- 动态资源迁移：`resource_migration_experiment.py`（并且已产出可用数据源）
- 取消语义：`cancellation_semantics_experiment.py`
- 崩溃恢复语义：`crash_recovery_experiment.py`（建议先做一次“去真实 LLM 依赖”的收敛改造）
- 状态背板正确性/性能：`state_backplane_recovery_experiment.py`
- 控制面稳定性与背压：`large_scale_stress_test.py`
- 基线对比：`baseline_comparison.py`
- 调度器开销口径：`verify_scheduler_overhead.py`

主要缺口（如果你要“论文更硬”或“答辩更稳”会需要补）：
- 端到端链路实验：从 WebSocket/API 入口到调度器再到输出（当前更偏“调度与控制面实验”，不是完整产品链路压测）
- 统一的结果元数据规范：除个别脚本外，平台/Python/调度后端/模型路径等信息并未在所有结果里一致落盘
- 目录清洁策略：多处脚本会生成 `.dat`/`.log` 临时文件，容易让结果目录再次变乱

## 建议的“先改再跑”最小改动清单

- 崩溃恢复实验：用 Mock 长任务替代 `llama_cpp` 真实加载，移除 Linux 路径硬编码，临时 `.dat` 统一落到临时目录并在成功后删除
- 状态背板实验：临时 `.dat` 统一落到临时目录并在成功后删除
- 结果文件统一写入元数据：`platform`、`python`、`scheduler_backend`、`model_path_exists`、`seed`、脚本参数
