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
| `tts/` | Qwen3-TTS Base 独立 CPU/GPU smoke 与 10 块 prompt 复用验收 |
| `elastic_residency/` | 弹性驻留实验契约、物理核租约、资源与调度 jitter 采样；后续承载 T0–T3、LLM 驻留和动态核心预算 |

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

### 6. 独立 TTS

```bash
python experiments/tts/smoke_qwen3_tts_base.py --device cuda:0 --ref-text '准确转写'
python experiments/tts/smoke_qwen3_tts_base.py --device cpu --cpu-threads 8 --x-vector-only
python experiments/tts/profile_qwen3_tts.py --device cuda:0 --x-vector-only
python experiments/tts/profile_qwen3_tts_gpu_sentence_lengths.py \
  --device cuda:0 --x-vector-only --rounds 3 \
  --git-root /mnt/d/AI/xiaoyou-core/legacy/mvp_core
```

默认 ICL 必须提供参考音频准确转写；`--x-vector-only` 仅用于显式消融。smoke 产物位于 `experiment_results/tts/smoke/<run_id>/`，句长画像位于 `experiment_results/tts/profiling/gpu_sentence_length/<run_id>/`。指标使用 `first_chunk_ready`，不声称第一音频采样延迟。这里验证独立句子级 TTS，不证明 LLM→TTS 流水线或 Diffusion 共置性能。

### 7. 弹性驻留 A1 基础设施

```bash
python -m experiments.elastic_residency.smoke_a1 \
  --run-id <new_run_id> --duration 2.0 \
  --source-git-commit <windows_source_commit> \
  --source-git-dirty false \
  --source-git-status-entry-count 0
```

A1 产物位于 `experiment_results/elastic_residency/a1/<run_id>/`，包含冻结配置、环境 manifest、单轮原始 JSONL、事件、统一资源时间线、JSON/CSV 汇总和失败记录。WSL2 运行副本按规则不带 `.git`，因此正式运行必须把 Windows 源仓库 commit/dirty 快照显式传入；干净源码还必须声明状态项为 0。dirty 源码只有同时提供 `--source-git-snapshot <archive_or_patch>`（可选再传声明哈希）并把快照复制进运行目录后，provenance 才完整。首个 CPU 样本会标为无效并从聚合排除，资源时间线开始前会等待首个成功 GPU 样本；GPU 可用性/新鲜度、资源与事件循环 jitter、swap、affinity 和核心归还均属于 `passed` 硬门槛。该 smoke 不运行 TTS/LLM，因此不能作为后续性能目标或无 underrun 的证据。

当前通过硬门槛的 A1 记录为 `a1_smoke_20260829_verified_v7`，源 commit 为 `ae68608`。它只用于验收实验基础设施。

### 8. 弹性驻留 B1 TTS 组件 smoke

```bash
python -m experiments.elastic_residency.tts_component_offload \
  --run-id <new_run_id> --configs T0 T1 T2 T3 \
  --source-git-commit <clean_commit> \
  --source-git-dirty false --source-git-status-entry-count 0
```

每个配置在独立子进程中加载真实 Qwen3-TTS，记录 talker、speech codec、speaker encoder 和 prompt 的实际设备，分别测量 tokenize、talker、code transfer、codec decode，并保存整卡/进程估计/PyTorch 显存、CPU/RSS 和非静音 WAV 检查。若 WSL 驱动把进程显存报告为 `[N/A]`，结果必须保留 unavailable 原因，只能把相对空闲整卡基线的增量标为 estimate。当前通过记录为 `b1_tts_components_20260829_verified_v2`，源 commit 为 `38f4d81`。它是单文本冷态可行性 smoke，不代替 B2 多轮矩阵。

WSL2 音频优先使用 WSLg Pulse；若当前会话无法枚举输出设备，可自动或显式回退到 Windows WASAPI：

```bash
python -m experiments.elastic_residency.audio_playback_probe \
  --wav <pcm16_wav> --backend auto --output-json <summary.json>
python -m experiments.elastic_residency.audio_playback_probe \
  --wav <pcm16_wav> --backend windows --output-json <summary.json>
```

探针记录源/输出采样率、设备、host API、callback 数、状态、underflow 和最大 callback gap；不支持的 24 kHz 单声道会显式协商到设备格式。静音 smoke 只验证输出链路，正式 underrun 结论必须来自真实 TTS chunk 和并发工作流。

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
