# EdgeSched / MVP Core 项目索引

**用途**：给 Codex 和维护者提供低成本导航。  
**最后核对**：2026-08-31（以当前目录实际文件为准）

## 先判断你要做什么

| 任务 | 首读 | 主要代码 | 验证/证据 |
|---|---|---|---|
| 理解项目 | [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) | `main.py`, `services/`, `cpp_scheduler/` | `tests/`, `paper/experiment_data/` |
| Python 调度器 | [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md#4-核心模块索引) | `services/task_scheduler.py`（门面）+ `task_models.py` / `python_worker.py` / `scheduler_image_pipeline.py` / `scheduler_llm_pipeline.py` | `tests/test_chat_service.py`, `tests/verify_todo_optimizations.py` |
| 准入/背压 | [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md#5-关键运行数据流) | `services/admission_controller.py`, `presentation/websocket/handler.py` | `experiments/admission/`, `tests/test_chat_service.py` |
| 显存迁移 | [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md#5-关键运行数据流) | `services/resource_manager.py`, `services/cpp_scheduler_engine.py`（门面）+ `cpp_engine_migration.py` / `cpp_engine_llm.py` / `cpp_engine_image.py` / `cpp_engine_utils.py` | `experiments/migration/` |
| 独立 TTS | [TTS_SUBSYSTEM.md](TTS_SUBSYSTEM.md)、[GPU 句长画像](TTS_GPU_SENTENCE_LENGTH_PROFILE_20260828.md) | `services/tts/`, `data/adapters/qwen3_tts_adapter.py` | `tests/tts/`, `experiments/tts/` |
| 弹性驻留可行性 | [执行计划](ELASTIC_RESIDENCY_EXPERIMENT_EXECUTION_PLAN_20260828.md)、[结果目录](ELASTIC_RESIDENCY_RESULT_CATALOG_20260830.md)、[论文实验指南索引](../paper/thesis_experiment_guide/README.md)、[当前门禁](../paper/thesis_experiment_guide/10_gates_and_current_status.md) | `experiments/elastic_residency/`；C2/C3 direct NVML 240 条性能曲线、KV 48 条容量矩阵、T0/T3 与 Forge 同后端校准、完整 16 档筛选及单组合共置 smoke 均已完成，当前准备 6 候选×5 轮正式共置矩阵 | `tests/elastic_residency/`、`experiment_results/elastic_residency/`，不得自动晋升正文 |
| Phase-Aware | `.trae/skills/xiaoyou-mvp-core/SKILL.md` | `services/phase_aware/`, `cpp_scheduler/core/phase_strategy.*` | `tests/test_phase_aware_quick.py`, `experiments/phase_aware/` |
| GPU 计算份额（SM-A） | [CPU/SM 分配方法](../paper/thesis_experiment_guide/05_cpu_and_gpu_allocation.md)、[实现路线](../paper/thesis_experiment_guide/06_implementation_and_roadmap.md)、[当前门禁](../paper/thesis_experiment_guide/10_gates_and_current_status.md) | `experiments/elastic_residency/sm/` 子包；入口 `sm/sm_capability.py` | WSL 正式记录 `sm_a_20260830_direct_nvml_v1`：基线成功、36 SM、0 failure/fallback/cleanup failure，但 MPS binary 不存在，暂定 NO_GO；SM-B/C 暂停，等待原生 Linux 恢复验证 |
| 崩溃恢复 | `docs/PROJECT_CONTEXT.md` | `services/state_backplane.py`, `services/task_scheduler.py` | `experiments/recovery/`, `experiments/semantic_validation/` |
| C++ 调度器 | [cpp_scheduler/README.md](../cpp_scheduler/README.md) | `cpp_scheduler/core/`, `workers/`, `queue/`, `bindings/` | `cpp_scheduler/verify_bindings.py`, `tests/test_cpp_scheduler_minimal.py` |
| 跑实验 | [experiments/README.md](../experiments/README.md) | `experiments/run_*.py`, 各实验子目录 | `experiment_results/`, manifest |
| 论文重写与图表 | [paper/README.md](../paper/README.md)、[实验指南索引](../paper/thesis_experiment_guide/README.md)、[新版骨架](../paper/rewrite/OUTLINE.md) | `paper/thesis_experiment_guide/`, `paper/rewrite/`；方法公式 `paper/edgesched_phase_aware_formulation.tex`；架构图 `paper/system_architecture.tex`；`paper/THESIS_EXPERIMENT_GUIDE_DRAFT.md` 仅为兼容入口 | `paper/experiment_data/01_paper_main/`（当前尚未晋升新数据） |
| WSL2 同步/环境 | [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md#7-环境与同步边界) | `scripts/sync_to_wsl2_safe.ps1`, `cpp_scheduler/build_wsl2.sh` | WSL2 实际运行日志 |

## 入口速查

| 入口 | 作用 | 备注 |
|---|---|---|
| `main.py` | FastAPI + `/ws` + `/health` 演示服务 | 启动全局调度器；缺模型时回退 Mock |
| `experiments/run_all_experiments.py` | 论文实验统一编排 | 部分项目要求 Forge 与真实模型 |
| `experiments/run_no_forge_experiments.py` | 无 Forge 实验集合 | 适合先验证 LRU、恢复、压力与实现开销 |
| `experiments/run_forge_experiments.py` | Forge 相关真实实验 | 运行前确认 Forge API 和模型可用 |
| `experiments/run_performance_suite.py` | Mock/Real 性能套件 | 产物写入 `experiment_results/`，不自动成为论文正文数据 |
| `experiments/tts/smoke_qwen3_tts_base.py` | Qwen3-TTS Base CPU/GPU 独立 smoke | 句子级流式；不代表 LLM→TTS 端到端结果 |
| `experiments/tts/profile_qwen3_tts_gpu_sentence_lengths.py` | GPU 短/中/长句正式 profiling | 预热后 5 文本 × 3 档 × 3 轮，生成原始记录、汇总、拟合与图表 |
| `experiments/elastic_residency/smoke_a1.py` | A1 实验基础设施硬门禁 | GPU 预热/新鲜度、jitter、CPU 首样本、swap、affinity、核心归还与 Git provenance |
| `experiments/elastic_residency/tts_component_offload.py` | B1 TTS T0–T3 单文本 smoke | 独立子进程、实际组件设备、code 传输、非静音 WAV 与分阶段 CUDA 峰值；v2 已通过 |
| `experiments/elastic_residency/tts_component_offload_matrix.py` | B2 TTS 正式矩阵 | 20/50/100 字 × 5 轮、逐配置预热、原始 WAV/哈希、分组统计与 Pareto；direct NVML v2 已完成并通过 30/30 条技术核验，尚未晋升论文正文数据 |
| `experiments/elastic_residency/tts_gpu_snapshot.py` | TTS 同步 GPU 快照 | direct NVML 整卡/进程 unavailable 语义，并记录查询时间与新鲜度保守上界；禁止 CLI fallback |
| `experiments/elastic_residency/tts_runtime_evidence.py` | TTS 矩阵运行期证据 | 固定 deadline 采集 swap 原始时间线；校验同步 direct NVML 查询耗时上界，并把门禁写回 iteration/child/summary |
| `experiments/elastic_residency/llm_residency_curve.py` | C1/C2 LLM 驻留 profiler | 真实层数、精确 prompt/decode、阶段资源和 KV；C2 支持多上下文轮转、严格续跑、checkpoint 与 fail-fast；v3 起新增 `full_state_serialized_bytes`/`kv_sequence_state_bytes`/`kv_position_*`/`kv_gpu_buffer_mb` 等字段（见 `kv_measurement.py`），历史 C2/C3 数据不含这些字段且仍然有效 |
| `experiments/elastic_residency/kv_measurement.py` | KV 测量语义与采集共享模块 | 显式拆分 GPU 预分配 KV buffer / 已用 token 范围 / 单序列序列化状态 / `save_state()` 完整负载 / 整卡增量五个概念；`LlamaInitLogCapture` 解析 llama.cpp 初始化日志，缺字段保持 None 禁止填 0 |
| `experiments/elastic_residency/llm_kv_cache_calibration.py` | KV 容量与语义校准（独立入口） | `c3_kv_cache_calibration_20260830_direct_nvml_v1` 已完成 16 configs × 3 contexts × 1 run = 48/48 条；以 `n_ctx=context+128` 匹配 C2/C3 推理后状态，不做性能计时 |
| `experiments/elastic_residency/llm_tts_colocated.py` | LLM+TTS 真实共置显存测量 | direct NVML 整卡时间线、`B_LLM` 基线、TTS 分阶段 torch peak、`P_colocated` 峰值、`colocated_positive_residual_mb` 残差与容量裁决；容量公式不重复叠加 KV，正式组合支持 5 轮重复 |
| `experiments/elastic_residency/gpu_telemetry.py` | 论文 GPU 遥测 | 通过 ctypes 直接调用 NVML；正式路径禁止 CLI 子进程，显存末位查询保证时间戳语义，并记录设备级显存及 per-process unavailable 语义 |
| `experiments/elastic_residency/forge_b1_reprofile.py` | Forge b=1 同日复测（C 阶段收尾） | 依赖已启动的 Forge HTTP API；产出空闲高水位、常驻基线、逐轮峰值与 `m_d_incremental_estimate_mb`，供 H_D 冻结 |
| `experiments/elastic_residency/forge_runtime_evidence.py` | Forge 运行期有效性门禁 | 对所有参与容量计算的 idle/resident/measurement 窗口逐窗校验周期 GPU 样本年龄，并保留 Forge/WSL2 swap 增长容忍语义 |
| `experiments/elastic_residency/analysis_input_merge.py` | C2/C3 完整曲线输入门禁 | 按 context 合并粗测与细化结果，拒绝失败、context 不一致、重复档位和非 16 档输入 |
| `experiments/elastic_residency/analyze_results.py` | C 阶段 G_safe/H_TTS/H_D 筛选分析 | 纯标准库；合并 C2+C3 完整 16 档并读取 B2/Forge summary，输出 G_safe 分量、逐上下文 H 与 k±2/k±4 临界补测建议；只做容量筛选不做共置结论 |
| `experiments/elastic_residency/colocated_runtime_evidence.py` | LLM+TTS 共置运行期证据门禁 | 对 TTS 执行窗口的周期 GPU 样本、参与 `B_LLM/P_colocated` 的同步快照和全生命周期 swap 增长执行硬门 |
| `experiments/elastic_residency/colocated_plan.py` | LLM+TTS 显式候选矩阵 | 解析 `CONTEXT:LLM:TTS` 候选并按给定顺序生成重复轮次，避免筛选结果被扩成无关笛卡尔积；最紧候选可前置 fail-fast |
| `experiments/elastic_residency/sm/sm_capability.py` | SM-A GPU 计算份额能力探针 | 独立实验基础设施，不改生产调度器；WSL 正式 probe 已因 `mps_binary_unavailable` 判定 NO_GO，当前只能保留普通 CUDA 不受控共享语义 |
| `paper/compile_en.bat` | Windows 英文论文编译 | LaTeX 产物属于生成文件 |
| `cpp_scheduler/build_wsl2.sh` | WSL2 原生扩展构建 | 使用 `mvp_core-cu128` 环境 |

## 真源优先级

1. 当前源码、测试断言和运行日志。
2. 当前实验 JSON、manifest、生成脚本。
3. `paper/experiment_data/README.md` 与人工晋升后的 `01_paper_main/`。
4. 本索引、`PROJECT_CONTEXT.md` 和各目录 README。
5. `docs/archive/` 与 `paper/archive/` 中的历史评估、旧稿和旧数据，仅供追溯。

## 常见误区

- `experiment_results/` 中有 JSON，不等于论文已采用该结果。
- Mock、NVRAM 模拟和 mmap 延迟不能写成真实硬件 NVRAM 结论。
- C++ 扩展能导入，不等于真实 GGUF 推理、KV Cache 保存恢复或端到端迁移已经通过。
- Windows 文件存在，不等于 WSL2 运行副本和模型目录已同步。
- 父仓库 `.gitignore` 当前忽略整个 `/legacy/`；本目录修改默认不会出现在父仓库 Git 状态中。
- Python 文件超过 300 行即违反 `AGENTS.md` 的规模上限；应按职责拆模块，而不是继续在同一文件里追加。
