# EdgeSched / MVP Core 项目索引

**2026-10-26 当前状态**：主开发环境为本机原生 Linux `/home/leslie/mvp_core`（2026-09-25 起，编辑、运行和 git 提交均在此处）。S0 v2 正式比较 `s0_formal_20260920_223105_8510199`（`K24:T0` vs `K0:T0` 各 5 轮，10/10、0 failure）显示混合在主指标上全面占优（首句 4.384 vs 10.740 s、应用层供给缺口 0 vs 6 段），但输出文本/句数/token 未受控、设备时钟仍 pending，**不登记 Gate B Go**，结论维持“收缩主张”，planner 实现暂停。SM-A 已于 2026-10-13 在原生 Linux 转 **GO**（`sm_a_20261013_native_linux_v6`，实测有效 SM 36/26/18/8），SM-B 可按门禁启动，SM-C/D/E 待排。H_D 仍 pending；S0 设备层与 BGE 缓存补测仍待完成。

9B 选型、small 曲线与视觉画像的既有证据见 [选型与验收](NINE_B_MODEL_COMPARISON.md)、[16 档曲线与 KV 验收](SMALL_9B_CURVES_20260907.md)、[视觉阶段画像](SMALL_9B_VISION_PROFILE_20260907.md)；目录整理记录见 [文件夹归类地图](FOLDER_MAP.md)。

**用途**：给 Codex 和维护者提供低成本导航。  
**最后核对**：2026-10-26（以当前目录实际文件为准）

## 先判断你要做什么

| 任务 | 首读 | 主要代码 | 验证/证据 |
|---|---|---|---|
| 理解项目 | [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) | `main.py`, `services/`, `cpp_scheduler/` | `tests/`, `paper/experiment_data/` |
| Python 调度器 | [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md#4-核心模块索引) | `services/task_scheduler.py`（门面）+ `task_models.py` / `python_worker.py` / `scheduler_image_pipeline.py` / `scheduler_llm_pipeline.py` | `tests/test_chat_service.py`, `tests/verify_todo_optimizations.py` |
| 准入/背压 | [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md#5-关键运行数据流) | `services/admission_controller.py`, `presentation/websocket/handler.py` | `experiments/admission/`, `tests/test_chat_service.py` |
| 显存迁移 | [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md#5-关键运行数据流) | `services/resource_manager.py`, `services/cpp_scheduler_engine.py`（门面）+ `cpp_engine_migration.py` / `cpp_engine_llm.py` / `cpp_engine_image.py` / `cpp_engine_utils.py` | `experiments/migration/` |
| 独立 TTS | [TTS_SUBSYSTEM.md](TTS_SUBSYSTEM.md)、[GPU 句长画像](TTS_GPU_SENTENCE_LENGTH_PROFILE_20260828.md) | `services/tts/`, `data/adapters/qwen3_tts_adapter.py`；推理后端对照 `experiments/tts/qwen3_tts_backends.py` + `experiments/tts/profile_qwen3_tts_cudagraph.py`，启动器 `scripts/experiments/run_tts_backend_compare.ps1` | `tests/tts/`（含 `test_tts_cudagraph_probe.py`、`test_tts_backend_compare_launcher.ps1`）、`experiments/tts/`；2026-09-17 同环境 A/B `tts_backend_both_eager_20260917_072516`：CUDA Graph 把 20/50/100 字 RTF 从 1.7754/1.7257/1.7622 降到 **0.3168/0.3067/0.3032**（5.6–5.8×），用仓库外隔离环境跑；**不是位精确、逐次音频不一致（输出长度最大差约 10%）**，尚未接入 `s0_pipeline` |
| 弹性驻留可行性 | [执行计划](ELASTIC_RESIDENCY_EXPERIMENT_EXECUTION_PLAN_20260828.md)、[结果目录](ELASTIC_RESIDENCY_RESULT_CATALOG_20260830.md)、[论文实验指南索引](../paper/thesis_experiment_guide/README.md)、[当前门禁](../paper/thesis_experiment_guide/10_gates_and_current_status.md) | `experiments/elastic_residency/`；9B 为后续主模型，FULL 文本/视觉端点 2/2 已通过；历史 4B 多模态端点门、TEXT_ONLY C2 90/90、C3 150/150、KV 48/48 已通过，完整 16 档+B2 筛出 T0/T3 均可 FULL；4B 文本共置 30/30（`llm_tts_colocated_4b_text_only_20260904_direct_nvml_v1`）已冻结 G_safe/H_TTS；4B VISION_WARM 多轮画像 5/5 已记录；H_D 已按 16G 论文环境以负面边界证据收案（16G 下 K16 swap 405.15 MiB 失败，20G 同代码诊断 3/3 证明根因为主机内存容量），与 9B 构成跨尺度负面证据；CPU 核数画像入口 `auxiliary_core_budget.py` 已实现，用户执行 R_sys=2 曲线已完成 40/40；新增 BGE 候选缓存模式待用户补测，D3 并发验收待完成 | `tests/elastic_residency/`、`experiment_results/elastic_residency/`，9B 与新模型证据分开且不得自动晋升正文 |
| S0 端到端流水线（协议 v2，已接入加速后端） | [E2 协议 `s0_protocol_v2`](../paper/thesis_experiment_guide/07_experiment_stages_and_matrix.md#e2无图像的-llmtts-流水线)、[当前门禁](../paper/thesis_experiment_guide/10_gates_and_current_status.md) | `experiments/elastic_residency/s0_pipeline/`；入口 `s0_parent.py`（配置 × 启动策略轮转编排）、`s0_child.py`（单轮真实执行）、`s0_backends.py`（真实 LLM/TTS 加载，TTS 可选 `--tts-backend cuda_graph_static`（当前接入，加速 CUDA Graph，整模型 GPU/放置 T0）或 `upstream_dynamic`（旧后端，保留 T0–T4 分部件放置））；加速后端只存在于隔离 venv `faster-qwen3-tts/.venv`（依赖 transformers 5.x，装不进主环境），S0 为保持 LLM 与 TTS **同进程共置**默认用该环境解释器（已补装 `llama-cpp-python==0.3.35`(CUDA 编译) 与 `sounddevice==0.5.5`）；`s0_protocol_v2` 把正式配置由 4 档降为 2 档（`K24:T0`、`K0:T0`），`K24:T3`/`K24:T4` 因后端无分部件放置接口而排除；协议真源 `config.FROZEN_PROTOCOL`，测量范围 `measurement_scope.py`，启动策略 `startup.py`，门禁/指纹 `s0_evidence.py`；离线复算 `experiments/elastic_residency/analyze_s0_matrix.py` + `s0_matrix_metrics.py` + `s0_matrix_report.py`；启动器 `scripts/experiments/run_s0_pipeline.ps1` | `tests/elastic_residency/`（Windows 全量 500 通过、5 项 GPU 门控跳过；含 `test_s0_measurement_scope.py`、`test_s0_startup_policy.py`、`test_s0_protocol_freeze.py`、`test_analyze_s0_matrix.py`、`test_s0_tts_backend.py`、`test_s0_pipeline_launcher.ps1`）；测量范围分调度层（Gate B 判据）与设备层（时钟未验证前 pending）；真实批次：`s0_startup_20260917_v1` 与 `s0_formal_20260917_v1`（四配置 × 5 轮 = 20 次、19 通过，`K24:T4` r2 swap 失败计入分母）均为**旧后端 v1**；v2 首批 `s0_startup_20260917_v2`（加速后端）已完成：首句就绪 4.26 s、RTF 0.359、窗口内供给缺口 0、首播 proxy 4.28 s（预缓冲 26.65 s）；v2 正式比较 `s0_formal_20260920_223105_8510199`（`K24:T0` vs `K0:T0` 各 5 轮，10/10 通过、0 failure、来源干净）显示混合在主指标上全面一致占优（首句 4.384 vs 10.740 s、供给缺口 0 vs 6 段），但协变量未受控（输出 80/9 句 vs 65/8 句）＋设备层 pending，**仍不登记 Gate B Go**；归因已补两条（离线、不重跑）：按 token 归一后差距 2.22×→2.73×（未缩小），句数对齐到前 8 句后基线每句就绪都更慢且 5/5 一致（第 7 句 +34.76 s）；结论为**收缩主张**（收益未确证，非混合驻留被否定）：`K24:T3`/`K24:T4` 相对参考 `K24:T0` 未改善所报主指标，`K24:T0` 相对 `K0:T0` 的收益受输出差异影响尚未公平归因；本批四档都是部分驻留（`K24`=32 层上卡 24 层），不能反证完整共驻留可行；**暂停** planner 实现 |
| 多模态端点门（9B 主模型 / 4B 对照 + `mmproj`） | [实现路线 P1-B](../paper/thesis_experiment_guide/06_implementation_and_roadmap.md)、[当前门禁](../paper/thesis_experiment_guide/10_gates_and_current_status.md) | `experiments/elastic_residency/vision/` 子包；入口 `vision/vision_endpoint_smoke.py` | `tests/elastic_residency/test_vision_*.py`（静态/假后端 72 项通过）；真实端点门已通过（`v1_vision_endpoint_20260831_direct_nvml_v3`，4/4 单元）；`TEXT_ONLY` 与 `VISION_WARM_*` 证据必须分开，且这不是 4B 常驻画像或共置结论 |
| Phase-Aware | `.trae/skills/xiaoyou-mvp-core/SKILL.md` | `services/phase_aware/`, `cpp_scheduler/core/phase_strategy.*` | `tests/test_phase_aware_quick.py`, `experiments/phase_aware/` |
| GPU 计算份额（SM-A） | [CPU/SM 分配方法](../paper/thesis_experiment_guide/05_cpu_and_gpu_allocation.md)、[实现路线](../paper/thesis_experiment_guide/06_implementation_and_roadmap.md)、[当前门禁](../paper/thesis_experiment_guide/10_gates_and_current_status.md) | `experiments/elastic_residency/sm/` 子包；入口 `sm/sm_capability.py` | 原生 Linux 正式记录 `sm_a_20261013_native_linux_v6`（干净 `bb17b62`）：100/75/50/25% 四配置，`cuCtxGetExecAffinity` 实测有效 SM 36/26/18/8、七项 go_conditions 全真、verdict=GO；WSL 旧记录 `sm_a_20260830_direct_nvml_v1` 因 MPS binary 缺失为 NO_GO，仅作历史对照。SM-B 可按门禁启动，SM-C/D/E 待排 |
| 崩溃恢复 | `docs/PROJECT_CONTEXT.md` | `services/state_backplane.py`, `services/task_scheduler.py` | `experiments/recovery/`, `experiments/semantic_validation/` |
| C++ 调度器 | [cpp_scheduler/README.md](../cpp_scheduler/README.md) | `cpp_scheduler/core/`, `workers/`, `queue/`, `bindings/` | `cpp_scheduler/verify_bindings.py`, `tests/test_cpp_scheduler_minimal.py` |
| 跑实验 | [experiments/README.md](../experiments/README.md) | `experiments/run_*.py`, 各实验子目录 | `experiment_results/`, manifest |
| 论文重写与图表 | [paper/README.md](../paper/README.md)、[实验指南索引](../paper/thesis_experiment_guide/README.md)、[新版骨架](../paper/rewrite/OUTLINE.md) | `paper/thesis_experiment_guide/`, `paper/rewrite/`；方法公式 `paper/edgesched_phase_aware_formulation.tex`；架构图 `paper/system_architecture.tex`；`paper/THESIS_EXPERIMENT_GUIDE_DRAFT.md` 仅为兼容入口 | `paper/experiment_data/01_paper_main/`（当前尚未晋升新数据） |
| 运行环境（原生 Linux） | [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md#7-环境与同步边界) | `cpp_scheduler/build_wsl2.sh`（文件名保留历史命名） | 运行日志/manifest |
| git 与更新记录 | 直接在 `/home/leslie/mvp_core` 用 git 提交推送（2026-09-25 起，旧 `sync_to_wsl2_safe.ps1` 已废止）；详细记录见 [UPDATES.md](UPDATES.md) | `docs/UPDATES.md` | — |

## 入口速查

已验收的组合筛选入口：`scripts/experiments/run_small_9b_screen.ps1`（Windows 离线，无模型），见 [双任务/两条三任务分支与条件性四任务计划](SMALL_9B_NEXT_COMBINATIONS_20260907.md)。WSL 大版 9B 已按用户要求删除，小版及视觉文件保留。

视觉按需驻留的论文口径：指南 [01 工作流](../paper/thesis_experiment_guide/01_research_scope_and_workloads.md)、[02 状态](../paper/thesis_experiment_guide/02_state_and_runtime_modes.md)、[03 画像](../paper/thesis_experiment_guide/03_profiles_and_constraints.md)、[04 代价](../paper/thesis_experiment_guide/04_cost_and_plan_contract.md)。V 为 VL 前端组件，独立计加载/编码/释放；q_V 与 Diffusion 队列分开。当前回答后释放已实现，队列保温/提前释放仍待验证。

选定 small 9B 后重建完整曲线：`scripts/experiments/run_9b_small_curves.ps1`，依次 C2 90 + C3 150 + KV 48，手动运行、失败即停；见 [命令与口径](NINE_B_MODEL_COMPARISON.md)。

两份 9B 选型对比：`scripts/experiments/run_9b_comparison.ps1` → `compare_9b_models.py` / `compare_9b_support.py`。
默认两模型 × FULL/K0 × 3 轮，只测纯文本；运行时将 small 安全复制到 WSL 独立目录，见 [协议](NINE_B_MODEL_COMPARISON.md)。
用户批次 `compare_9b_20260907_023703` 已 12/12 并经离线原始记录复核；small FULL 显存峰值少 90 MiB、decode +2.05%，随后 small 视觉 Answer 已 1/1 通过，专用视觉入口默认切换为 Small。
small 视觉候选试测：现有视觉脚本加 `-ModelVariant Small -Stage Answer`，固定 WSL 独立副本及 SHA；文件头配对与用户真实 Answer 均通过，见 [试测说明](NINE_B_MODEL_COMPARISON.md)。

LLM_GPU/TTS_CPU 与 LLM_CPU/TTS_GPU 的旧结果定位、TTS 核数证据缺口及 GPU 兼容性/SM 性能分阶段口径，见 [放置复核](LLM_TTS_PLACEMENT_REVIEW_20260906.md)。

9B 文本/视觉手动入口：`scripts/experiments/run_vision_9b_smoke.ps1`，模型读取 WSL 原生目录
`/home/leslie/mvp_core/models/llm/`；两份文件已复制并通过 SHA-256 核验。
`-PrintOnly` 仅检查生成命令，实际实验由用户启动。
入口支持 `-Stage Answer`（1024-token 完整回答补测）和 `-Stage Profile`（128-token 视觉画像 5 轮）；
各阶段单独 run id，详见 [协议](VISION_9B_PROBE_20260906.md)。静态检查入口为 `tests/elastic_residency/test_vision_9b_launcher.ps1`。

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
| `experiments/tts/profile_qwen3_tts_cudagraph.py` | B2 TTS 画像（当前接入后端，T0） | 加速 CUDA Graph 路径整模型驻 GPU；20/50/100 字 × 5 轮 + 状态残留序列 + 主环境 ASR 回转核对；记录 RTF、波形就绪、加载/图准备成本、准备与执行阶段内存峰值、输出长度；入口 `scripts/experiments/run_tts_backend_compare.ps1 -Mode b2`，口径真源 `experiments/tts/b2_profile_contract.py`；旧后端 T0–T4 三批产物已于 2026-09-17 删除 |
| ~~`experiments/elastic_residency/tts_component_offload_matrix.py`~~ | 旧后端组件放置矩阵，**2026-09-17 已删除** | 原负责 20/50/100 字 × 5 轮的分部件放置（T0–T4）矩阵；产物先删、生成器随后删除（无环境可跑且加速后端不支持该维度）。冻结文本真源已移到 `experiments/tts/b2_profile_contract.py` 供新旧入口共用 |
| `experiments/elastic_residency/tts_gpu_snapshot.py` | TTS 同步 GPU 快照 | direct NVML 整卡/进程 unavailable 语义，并记录查询时间与新鲜度保守上界；禁止 CLI fallback |
| `experiments/elastic_residency/tts_runtime_evidence.py` | TTS 矩阵运行期证据 | 固定 deadline 采集 swap 原始时间线；校验同步 direct NVML 查询耗时上界，并把门禁写回 iteration/child/summary |
| `experiments/elastic_residency/llm_residency_curve.py` | C1/C2 LLM 驻留 profiler | 真实层数、精确 prompt/decode、阶段资源和 KV；C2 支持多上下文轮转、严格续跑、checkpoint 与 fail-fast；v3 起新增 `full_state_serialized_bytes`/`kv_sequence_state_bytes`/`kv_position_*`/`kv_gpu_buffer_mb` 等字段（见 `kv_measurement.py`），历史 C2/C3 数据不含这些字段且仍然有效 |
| `experiments/elastic_residency/kv_measurement.py` | KV 测量语义与采集共享模块 | 显式拆分 GPU 预分配 KV buffer / 已用 token 范围 / 单序列序列化状态 / `save_state()` 完整负载 / 整卡增量五个概念；`LlamaInitLogCapture` 解析 llama.cpp 初始化日志，缺字段保持 None 禁止填 0 |
| `experiments/elastic_residency/llm_kv_cache_calibration.py` | KV 容量与语义校准（独立入口） | `c3_kv_cache_calibration_20260830_direct_nvml_v1` 已完成 16 configs × 3 contexts × 1 run = 48/48 条；以 `n_ctx=context+128` 匹配 C2/C3 推理后状态，不做性能计时 |
| `experiments/elastic_residency/llm_tts_colocated.py` | LLM+TTS 真实共置显存测量 | direct NVML 整卡时间线、`B_idle/B_LLM/P_colocated`、正残差与扣除 idle 后的容量裁决；容量公式不重复叠加 idle 或 KV，每轮冻结非空 WAV 大小/SHA-256，正式组合支持 5 轮重复 |
| `experiments/elastic_residency/s0_pipeline/` | S0 无图像 LLM→TTS 端到端流水线（pilot） | 子包按 `config`/`events`/`clock`/`s0_prompt`/`token_identity`/`token_source`/`tts_executor`/`playback`/`playback_buffer`/`playback_stats`/`playback_clock`/`playback_starvation`/`playback_smoke`/`text_contract`/`output_contract`/`runner`/`s0_backends`/`s0_child`/`s0_parent` 拆分（全部 ≤300 行）；播放侧把生产者下标与设备下标分开，断流按独立段落合并并分启动/句间/句内/收尾/未知阶段，时钟映射必须通过「单位一致 + 残差 + 漂移 + 单调」验证才可用于实际播放时刻；`playback_smoke.py` 只验测量能力，判定分 `measurement_ok` 与 `continuity_ok` 两层：LLM 流式→固定分句→有界队列→逐句合成→连续播放；输入适配层只管格式安全（`text_contract`）与判据（`output_contract`），思考正文与空输出只判失败不删除，保存原文/处理后文本/句子映射；`s0_prompt` 用模型自带模板渲染并追加空 think 块关闭 thinking；事件区分生成与交付时间、时钟域与时钟映射，播放侧分离供给不足与设备 underrun；入口 `s0_parent.py`（四配置 pilot），产物进 `experiment_results/elastic_residency/s0_pipeline/`；pilot 只验链路与采集，不下性能结论 |
| `experiments/elastic_residency/colocation/` | V0 共跑减速诊断（2026-10-26） | `v0_parent.py` 逐单元起子进程：LLM 单独/TTS 驻留/同进程合成/分进程（核重叠或不重叠）× llama 线程数 × torch 线程数；产物 `experiment_results/elastic_residency/colocation_v0/`；结论见指南第 11 模块 3.1 |
| `experiments/elastic_residency/analyze_s0_slack.py` | S0 播放余量与共跑减速离线复算 | 读 S0 `events.jsonl`，按批次给出交付速度、合成后余量与突发任务盈亏平衡；可对照 C2/C3 隔离画像 |
| `scripts/experiments/edgesched_env.sh` | 原生 Linux 实验环境 | `source` 后得到 `$EDGESCHED_PYTHON`、CUDA 12 库路径与 TTS 权重路径（主 venv 为 `/home/leslie/Autarkia/.venv`） |
| `experiments/elastic_residency/colocated_audio_artifact.py` | 共置音频 artifact 门禁 | 独立负责输出 WAV 的非空检查、大小和 SHA-256；不引入 GPU 依赖，不代替 waveform 或听感验收 |
| `experiments/elastic_residency/colocated_capacity.py` | 共置容量纯计算口径 | `(P_colocated-B_idle)+G_safe`、物理峰值保护、正残差；供运行入口与离线复算共用 |
| `experiments/elastic_residency/analyze_colocated_results.py` | 正式 LLM+TTS 共置离线复算 | 硬验正式重复矩阵、真实后端/direct NVML/新鲜度/swap/provenance，复算 G_safe 与按 context 冻结 H_TTS；`--screen-schema small` 经 `small_screen_view.py` 适配 small 9B 离线筛选产物，候选判据为“已画像档位内”并输出与离线边界的差异；H_D 保持待 LLM+Forge 共置 |
| `experiments/elastic_residency/llm_forge_colocated.py` | LLM+Forge 真实共置入口 | 从 C3 screen 读取 H_D 候选和隔离画像；父进程预热真实 Forge，子进程加载精确 LLM/KV 状态并发起 b=1 请求；支持单候选 smoke、3 候选×5 轮正式矩阵、fail-fast 与严格 resume |
| `experiments/elastic_residency/llm_forge_child.py` | 单轮 LLM+Forge 测量 | 保存真实 PNG、`B_Forge/B_LLM/P_colocated`、KV、direct NVML 时间线、新鲜度、swap、affinity、资源归还和容量裁决 |
| `experiments/elastic_residency/llm_forge_plan.py` | H_D 候选/画像契约 | 解析 `CONTEXT:LLM`；正式模式只接受 screen 边界，显式探索模式仅接受 screen 已画像的更低驻留档，并绑定逐 context 的 M_L/M_D/G_safe/idle/GPU total |
| `experiments/elastic_residency/analyze_forge_colocated_results.py` | H_D 正式冻结分析 | 硬验 3 候选×5 轮真实矩阵，合并 TTS/Forge colocated 正残差并重新检查 H_TTS，最终冻结 G_safe/H_TTS/H_D |
| `experiments/elastic_residency/gpu_telemetry.py` | 论文 GPU 遥测 | 通过 ctypes 直接调用 NVML；正式路径禁止 CLI 子进程，显存末位查询保证时间戳语义，并记录设备级显存及 per-process unavailable 语义 |
| `experiments/elastic_residency/forge_b1_reprofile.py` | Forge b=1 同日复测（C 阶段收尾） | 依赖已启动的 Forge HTTP API；产出空闲高水位、常驻基线、逐轮峰值与 `m_d_incremental_estimate_mb`，供 H_D 冻结 |
| `experiments/elastic_residency/forge_runtime_evidence.py` | Forge 运行期有效性门禁 | 对所有参与容量计算的 idle/resident/measurement 窗口逐窗校验周期 GPU 样本年龄，并保留 Forge/WSL2 swap 增长容忍语义 |
| `experiments/elastic_residency/analysis_input_merge.py` | C2/C3 完整曲线输入门禁 | 按 context 合并粗测与细化结果，拒绝失败、context 不一致、重复档位和非 16 档输入 |
| `experiments/elastic_residency/analyze_results.py` | C 阶段 G_safe/H_TTS/H_D 筛选分析 | 纯标准库；合并 C2+C3 完整 16 档并读取 B2/Forge summary，输出 G_safe 分量、逐上下文 H 与 k±2/k±4 临界补测建议；只做容量筛选不做共置结论 |
| `experiments/elastic_residency/colocated_runtime_evidence.py` | LLM+TTS 共置运行期证据门禁 | 对 TTS 执行窗口的周期 GPU 样本、参与 `B_LLM/P_colocated` 的同步快照和全生命周期 swap 增长执行硬门 |
| `experiments/elastic_residency/colocated_plan.py` | LLM+TTS 显式候选矩阵 | 解析 `CONTEXT:LLM:TTS` 候选并按给定顺序生成重复轮次，避免筛选结果被扩成无关笛卡尔积；最紧候选可前置 fail-fast |
| `experiments/elastic_residency/sm/sm_capability.py` | SM-A GPU 计算份额能力探针 | 独立实验基础设施，不改生产调度器；原生 Linux 正式记录 `sm_a_20261013_native_linux_v6` 已 GO（四档有效 SM 36/26/18/8），SM-B 可按门禁启动；WSL 旧记录因 MPS binary 缺失为 NO_GO，仅作历史对照 |
| `experiments/elastic_residency/vision/vision_endpoint_smoke.py` | V1 多模态端点 smoke（K0/FULL × TEXT_ONLY/VISION_WARM） | 先过 artifact/mmproj 能力门禁，再逐单元子进程真实执行；视觉轮次用 libmtmd 对固定图片实际执行 projector，`--vision-eval-mode split_encode` 额外分离 projector 编码时间与视觉嵌入证据 |
| `experiments/elastic_residency/vision/vision_endpoint_gate.py` | artifact 与 mmproj 能力门禁 | 不加载模型；核验架构/层数/`projection_dim == embedding_length`/来源一致性/声明哈希与固定图片 |
| `experiments/elastic_residency/vision/gguf_reader.py` | GGUF 头部元数据只读解析 | 纯标准库；只解析 KV 与张量计数，数组折叠为类型与数量，缺字段保持 `None` |
| `paper/archive/pre_rewrite_20260830/manuscript/compile_en.bat` | 旧英文论文编译入口（已归档） | LaTeX 产物属于生成文件 |
| `cpp_scheduler/build_wsl2.sh` | WSL2 原生扩展构建 | 使用 `mvp_core-cu128`（软链→共享 3.13 venv）、CUDA 12.8、sm_120 |

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
- `mmproj` 文件存在或 artifact 门禁通过，不等于 projector 已在真实图片上执行；必须看视觉 token 数、`mtmd` 返回码/嵌入证据与 GPU/RSS 增量。
- `data/adapters/vl_adapter.py` 是独立 Qwen2-VL Transformers 路径，不能当作 Qwen3.5-4B GGUF + `mmproj` 统一执行链的证据。
- Windows 旧副本存在，不等于本机 Linux 工作树已更新；2026-09-25 起不再有自动同步流程。
- 父仓库 `.gitignore` 当前忽略整个 `/legacy/`；本目录修改默认不会出现在父仓库 Git 状态中。
- Python 文件超过 300 行即违反 `AGENTS.md` 的规模上限；应按职责拆模块，而不是继续在同一文件里追加。


## 辅助任务 CPU 核数画像入口

辅助任务 CPU 核数画像入口已实现：`experiments/elastic_residency/auxiliary_core_budget.py`，复用物理核租约与采样器，支持 JSON RAG 1/2 核、BGE/Whisper 1/2/4 核和 `R_sys=0/1/2/4` 隔离敏感性。模块职责、运行方法和 D3 正式并发验收边界见 [辅助核数画像](AUXILIARY_CORE_BUDGET.md)。


BGE 画像已按职责拆出 `experiments/elastic_residency/auxiliary_bge.py`：保留 `bge` 全量重编码，新增 `bge_cached` 候选预计算模式，准备成本与请求耗时分别记录，任务名分组避免混用。`scripts/experiments/run_auxiliary_cached.ps1` 供用户从任意目录手动启动 15 次 BGE 补测；具体协议与已完成的 40/40 结果见 [辅助核数画像](AUXILIARY_CORE_BUDGET.md)。真实实验由用户执行。
