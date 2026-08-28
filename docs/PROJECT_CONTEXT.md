# EdgeSched / MVP Core 项目上下文

**文档版本**：1.0  
**最后核对**：2026-08-29
**核对范围**：当前 Windows 工作副本中的源码、测试、实验入口、论文数据目录和 WSL2 脚本。

## 1. 项目定位

`legacy/mvp_core` 是 Xiaoyou Core 内部的调度研究原型，论文对外名为 **EdgeSched**。它研究资源受限单机上的多模态本地 AI 调度，重点不是主应用的对话、人设或主动关怀，而是：

- LLM 常驻与图像突发任务之间的 GPU/显存竞争；
- GPU→CPU 降级和 CPU→GPU 原子回迁；
- 准入控制、队列背压、取消与尾延迟；
- mmap 状态背板与崩溃后的任务状态闭合；
- Phase-Aware workflow、GPU lease、迁移代价决策和 CPU backfill；
- 可复现实验、原始迭代、聚合数据、图表和论文证据链。

该目录可以独立运行演示服务和实验，但配置优先复用父仓库；真实模型、CUDA、Forge 和 C++ 构建的权威环境在 WSL2。

## 2. 当前成熟度与证据边界

| 区域 | 当前状态 | 可以证明什么 | 不能自动证明什么 |
|---|---|---|---|
| Python 调度器 | 有实现和 Mock/单元验证 | 优先级、队列、取消、背压、状态流转 | 真实 GPU 上的性能收益 |
| C++ 调度器 | 有源码、pybind11 绑定和 WSL2 构建产物 | 当前绑定可导入，原生接口与调度机制可实现 | CUDA、真实 GGUF、迁移和性能一定可用 |
| 状态背板 | Python mmap 实现 + 恢复实验 | 固定槽位状态持久化和恢复语义 | 真实 PMEM/NVRAM 硬件性能 |
| 资源迁移 | Python/C++ 桥接与实验脚本 | 迁移协议、降级/回迁路径可测试 | 任意模型/驱动环境都能无缝迁移 |
| Phase-Aware | Python 和 C++ 两套策略实现 | 角色/亲和性/lease/cost/backfill 抽象 | 已成为所有主调度路径的默认生产策略 |
| 论文数据 | `01_paper_main` + 原始迭代 + manifest | 当前稿件使用的既有数据和生成产物 | 代码修改后数据仍然有效 |

历史文档曾使用“工业级”“零拷贝”“断电恢复”等强表述。除非当前代码、真实运行环境和实验数据共同支持，否则应降级为设计目标、软件模拟或历史设想。

## 3. 架构总览

```text
Client / experiment
        |
        v
FastAPI + WebSocket (main.py, presentation/websocket/handler.py)
        |
        +--> ChatService --> LLM / Memory / Image adapters
        |
        v
GlobalTaskScheduler (services/task_scheduler.py)
        +--> AdmissionController     成本与压力驱动的准入/路由
        +--> ResourceManager        显存监控、模型状态、释放与恢复
        +--> StateBackplane         mmap 任务状态槽与重启闭合
        +--> PythonWorker           CPU/普通 Python 执行
        +--> CPPSchedulerEngine ---- pybind11 ---- C++ ResourceIsolationScheduler
        |                                  +--> queue / workers / monitoring
        |                                  +--> llama.cpp worker / KV cache
        |
        +--> PhaseAwareScheduler
              +--> phase model
              +--> GPU lease
              +--> migration cost decision
              +--> CPU backfill
```

### 控制面与执行面

- Python 控制面负责请求生命周期、异步编排、准入、背压、取消、状态持久化和实验采样。
- C++ 执行面负责原生任务队列、worker、资源隔离、llama.cpp 推理和 Phase-Aware 原生策略。
- 两者不是互斥实现：`GlobalTaskScheduler` 根据配置选择 Python 回退、C++ 路由或图像队列路径。

## 4. 核心模块索引

### 服务与调度

| 文件/目录 | 职责 | 修改时同时检查 |
|---|---|---|
| `services/task_scheduler.py` | 全局门面（L1 Router）：生命周期、任务调度、C++ 路由、背板恢复；re-export 历史公开符号 | `presentation/websocket/handler.py`, 调度测试 |
| `services/task_models.py` | TaskInfo / TaskPriority / TaskStatus / TaskType 与 current_task_id 上下文 | `task_scheduler.py`, `python_worker.py` |
| `services/python_worker.py` | L2 回退执行器（asyncio 优先级队列 + ThreadPoolExecutor） | `task_scheduler.py` |
| `services/scheduler_image_pipeline.py` | ImagePipelineMixin：图像队列背压、准入快照、worker 循环、Forge→C++→SD→Synthetic 路由 | `task_scheduler.py`, `admission_controller.py` |
| `services/scheduler_llm_pipeline.py` | LLMPipelineMixin：LLM 取消事件、回退适配器、C++/Python LLM 路由 | `task_scheduler.py`, `cpp_scheduler_engine.py` |
| `services/admission_controller.py` | ADMIT_NOW / QUEUE / REJECT、动态任务画像、路由决策日志 | 图像背压、实验统计字段 |
| `services/resource_manager.py` | GPU/CPU/进程资源采样、模型优先级、offload/restore 回调 | `cpp_scheduler_engine.py`, NVIDIA-SMI 可用性 |
| `services/cpp_scheduler_engine.py` | C++ 引擎门面（单例）：生命周期、断路器状态机；导入路径不变 | `cpp_scheduler_wrapper.py`, 绑定 API, 配置项 |
| `services/cpp_engine_utils.py` | 共享守卫导入（wrapper/resource_manager/llama_cpp）+ OOM/CUDA 识别、nvidia-smi、llama.cpp 补丁 | 各 `cpp_engine_*.py` |
| `services/cpp_engine_migration.py` | MigrationMixin：LLM GPU↔CPU 非对称迁移、KV Cache 保存恢复、卸载 | `cpp_engine_llm.py`, 迁移实验 |
| `services/cpp_engine_llm.py` | LLMBackendMixin：Python/C++ LLM worker 装配、资源自适应加载、流式提交 | `cpp_engine_migration.py`, 绑定 API |
| `services/cpp_engine_image.py` | ImageBackendMixin：C++ 图像 worker 装配、Forge 优先图像提交、burst 指标 | `cpp_engine_llm.py`, 准入控制 |
| `services/cpp_scheduler_wrapper.py` | 搜索并加载 `scheduler_py` 扩展，兼容绑定差异 | 构建输出路径、导出符号 |
| `services/state_backplane.py` | `XYBP` mmap 布局、32 个任务槽、状态读写 | 恢复实验、临时数据路径 |
| `services/nvram_manager.py` | 早期/实验性 `XYNV` mmap 模拟 | 不要与当前任务背板混为一谈 |

### Phase-Aware

| 文件 | 职责 |
|---|---|
| `services/phase_aware/phase_model.py` | workflow、阶段属性、设备亲和性、LLM 三态 |
| `services/phase_aware/gpu_lease.py` | RESIDENT / BURST / SHARED GPU lease |
| `services/phase_aware/cost_model.py` | 迁移代价、burst 时长预测、迁移/等待/旁路决策 |
| `services/phase_aware/backfill_scheduler.py` | GPU 独占期间执行 CPU 亲和阶段 |
| `services/phase_aware/phase_scheduler.py` | SEQUENTIAL / V1 / V2 三种策略与执行轨迹 |
| `services/phase_aware/cpp_adapter.py` | Python 策略与 C++ 策略/迁移回调对接 |
| `cpp_scheduler/core/phase_strategy.*` | C++ 侧同构的阶段、代价、lease 与默认策略 |

### 接口与适配器

| 路径 | 职责 |
|---|---|
| `main.py` | DI 注册、FastAPI 生命周期、`/ws`、`/health` |
| `domain/interfaces/base_interfaces.py` | LLM/Memory/TTS/STT/Image/Vision 抽象 |
| `domain/services/chat_service.py` | LLM 流与记忆写入编排 |
| `presentation/websocket/handler.py` | 请求协议、图像排队、取消、安全发送、输出保存 |
| `data/adapters/gguf_llm_adapter.py` | llama-cpp-python GGUF 适配 |
| `data/adapters/forge_ui_adapter.py` | Forge API 图像适配 |
| `data/adapters/synthetic_image_adapter.py` | 无真实图像后端时的可控替代 |
| `services/tts/` | 独立句子级流式 TTS：断句、有界队列、取消、spawn Worker 与顺序检查 |
| `data/adapters/qwen3_tts_adapter.py` | Qwen3-TTS Base 真实适配器；模型仅存在于子进程 |

独立 TTS 的音色画像、真实运行方式和证据边界见 `docs/TTS_SUBSYSTEM.md`。当前尚未接入 Phase-Aware，既有 TTS 静态资源值仅是未校准占位。

## 5. 关键运行数据流

### LLM 请求

```text
WebSocket request
  -> WebSocketHandler
  -> ChatService
  -> LLMInterface
  -> GlobalTaskScheduler / CPPSchedulerEngine（按配置）
  -> token stream
  -> WebSocket response
```

请求取消通过 `request_id` 关联调度器取消事件；验证时必须检查取消后不再输出，而不只检查取消函数返回成功。

### 图像请求与背压

```text
WebSocket image request
  -> admission decision
  -> bounded asyncio queue
  -> image worker
  -> ResourceManager / LLM offload（需要时）
  -> Forge / SD / synthetic adapter
  -> result or structured rejection
```

队列容量和并发数来自调度配置。队列满时是受控拒绝，不应绕过队列直接启动新任务。

### 独立 TTS 请求

```text
token stream
  -> 跨 token 内部标签过滤
  -> 确定性完整句子
  -> 有界队列
  -> 单 spawn Qwen3-TTS Base Worker
  -> prompt 缓存复用
  -> 有序 WAV chunk
```

取消会在锁内移除待执行块；正在执行的块自然完成后作为迟到结果丢弃。当前是句子级流式，`non_streaming_mode=True`，不是模型内部的增量音频流；延迟指标统一命名为 `first_chunk_ready`，不声称第一音频采样延迟。2026-08-28 GPU 句长画像中 45/45 条 RTF 均大于 1，详见 `docs/TTS_GPU_SENTENCE_LENGTH_PROFILE_20260828.md`。

### 迁移

```text
GPU burst arrives
  -> cost/admission decision
  -> GPU->CPU: save KV/state -> release GPU -> CPU degraded service
  -> run non-preemptible burst
  -> CPU->GPU: preload new GPU instance -> atomic swap -> release old instance
```

“原子回迁”描述的是切换协议，不等于整个迁移过程零耗时。报告应分别记录迁移总耗时、用户可见 service gap、CPU 降级 TTFT 和 restore gap。

### 崩溃恢复

`StateBackplane` 保存固定槽位任务状态。启动时 `GlobalTaskScheduler` 读取活跃槽位：遗留 RUNNING 必须闭合为失败，PENDING 尝试重新入队。测试终点是所有任务最终闭合，不是仅能读取 mmap 文件。

## 6. 实验与论文证据链

### 目录语义

| 路径 | 语义 |
|---|---|
| `experiments/framework/` | 路径、配置、统计、随机种子、环境与真实/Mock 适配器 |
| `experiments/migration/` | EdgeSched 与对称/消融迁移策略 |
| `experiments/baselines/` | cold start、LRU thrashing、manual switching |
| `experiments/recovery/` | 崩溃恢复主实验 |
| `experiments/semantic_validation/` | 取消和背板恢复语义 |
| `experiments/admission/` | 准入策略对比 |
| `experiments/stress/` | 大规模与真实混合负载压力 |
| `experiments/workload/` | 真实交互 workload/trace |
| `experiments/phase_aware/` | Phase-Aware 三组实验 |
| `experiments/overhead/` | 实现开销、综合实验和消融套件 |
| `experiments/elastic_residency/` | 弹性驻留统一契约、物理核租约、资源与调度 jitter 采样及后续可行性实验 |
| `experiment_results/` | 工作区运行产物和中间结果 |
| `paper/experiment_data/01_paper_main/` | 当前正文聚合数据 |
| `paper/experiment_data/02_auxiliary/` | 辅助说明数据 |
| `paper/experiment_data/03_raw_iterations/` | 原始迭代证据 |
| `paper/previous/` | 历史稿件和历史产物，只读追溯 |

### 数据晋升流程

1. 在 WSL2 冻结代码、模型、环境变量和实验参数。
2. 先运行单次 smoke，确认真实后端没有回退到 Mock。
3. 按论文要求运行多次迭代，保留每次原始 JSON。
4. 生成 manifest，记录随机种子、环境与配置哈希。
5. 聚合到候选数据，检查缺失值、异常值和统计口径。
6. 只有人工确认后，才替换 `01_paper_main/` 的正文数据并重新生成图表/表格。
7. 对照 `paper/sections_en/06_evaluation.tex`，确认每个数字可追溯。

## 7. 环境与同步边界

| 用途 | 路径/环境 |
|---|---|
| Windows 编辑副本 | `D:\AI\xiaoyou-core\legacy\mvp_core` |
| WSL2 运行副本 | `/home/leslie/mvp_core` |
| Windows 访问 WSL2 | `\\wsl.localhost\Ubuntu-24.04-E\home\leslie\mvp_core` |
| 主实验环境 | `/home/leslie/.venvs/mvp_core-cu128` |
| Forge 环境 | `/home/leslie/.venvs/forge-cu128` |
| GGUF 模型 | `/home/leslie/mvp_core/models/llm` |
| 辅助模型 | `/home/leslie/mvp_core/models/auxiliary`（Faster-Whisper Small、BGE-small-zh-v1.5） |
| 图像模型 | `/home/leslie/mvp_core/models/Image` |
| Forge 服务 | `/home/leslie/mvp_core/stable-diffusion-webui-forge-main/` |

`scripts/sync_to_wsl2_safe.ps1` 是安全同步入口，只增量复制源码，不删除 WSL2 独有的模型、Forge 或构建产物。根目录模型排除必须写成 `/models/`，避免误排除 `domain/models/` 等源码包；Python 缓存和测试临时目录不参与同步。默认会提交并推送 Windows 独立仓库；存在其他未完成改动时使用 `-SkipGit`，避免把无关文件加入提交。真实运行前仍应核对目标文件时间戳或哈希。

### 7.1 当前 WSL2 运行快照

2026-08-28 在 `Ubuntu-24.04-E` 做过只读核验：

- `/home/leslie/.venvs/mvp_core-cu128` 能导入 `llama_cpp`；
- 配置 `cpp_scheduler/build/bin` 与 `cpp_scheduler/build/llama.cpp/src` 到 `LD_LIBRARY_PATH` 后，能导入 `/home/leslie/mvp_core/cpp_scheduler/build/scheduler_py.cpython-310-x86_64-linux-gnu.so`；
- Windows `venv_core` 中 benchmark 降级到 Mock，只表示 Windows 验证环境没有 `llama_cpp`，不能据此判断 WSL2 缺少依赖。
- `faster-whisper==1.2.1`、`ctranslate2==4.7.2`、`onnxruntime==1.23.2`、`sounddevice==0.5.5` 与 `libportaudio2` 已安装；Faster-Whisper CPU int8 真实接口 smoke 可运行。
- WSL2 已安装 ALSA→Pulse 桥接并能通过 WSLg Pulse 打开真实 PortAudio 输出；若 WSLg 音频端点失效，`experiments/elastic_residency/audio_playback_probe.py` 可回退到 Windows WASAPI，并显式记录重采样与传输路径。两条路径的 1 s 静音 smoke 均为 0 underflow，但正式结论仍须使用真实 TTS chunk 和并发负载。
- `blendmodes==2025`、`pillow==10.4.0` 与当前 NumPy/Gradio 形成兼容集合；WSL2 `pip check` 无破损依赖，Qwen-TTS、Gradio 与 Torch 导入通过。
- A1 最终 smoke `a1_smoke_20260829_verified_v7` 基于干净且已推送的 commit `ae68608` 通过全部 17 项硬门禁：GPU 首样本在时间线前就绪，41/41 点均可用且最大年龄 422.719 ms；资源 jitter p95/max 为 0.309/1.680 ms，事件循环为 2.165/3.081 ms；首个 CPU 样本被明确排除，swap 为 0，线程 affinity 正确且结束后 16 个物理核全部归还。旧 v6 已降级为调试记录。A1 仍不包含 TTS/LLM 正式负载或真实并发音频。
- B1 最终 smoke `b1_tts_components_20260829_verified_v2` 基于干净 commit `38f4d81` 使用真实 Qwen3-TTS/CUDA 后端，T0–T3 为 4/4 通过。实际组件设备、prompt device、`cuda:0→cpu` code 传输、分段时间、分阶段 PyTorch 显存峰值和非静音 WAV 均已验证。WSL 的进程显存接口返回 `[N/A]`，因此记录明确 unavailable，并以整卡相对空闲基线增量交叉估计；不把该估计伪装为精确进程显存。B1 是单轮冷态可行性证据，当前不能用于 Pareto 或论文性能结论。
- B2 `b2_tts_matrix_20260829_verified_v1` 基于干净 commit `ce024a0` 完成 60/60 条真实 TTS 迭代。T0/T3 为条件式 Pareto，T1/T2 被 T3 严格支配；T3 比 T0 少约 1304 MiB 最大整卡增量，但 TTFA p95 增加 26.1%。60 条 RTF 全部大于 1，所以当前后端不具备持续实时播放能力；T3 是否值得保留还要等 LLM 层驻留曲线。该数据没有晋升论文正文。
- C1 `c1_llm_residency_512_20260829_verified_v1` 基于干净 commit `234d2df`，用真实 32 层 Qwen3.5-9B GGUF 完成 K0/FULL 的 512-token 端点 smoke，两端均通过 19/19 项门禁并正确保存/恢复 640-token KV state。FULL 相对 K0 的 TTFT 为 0.326/10.659 s、decode 为 51.03/2.77 tok/s，GPU 增量估计为 5543/127 MiB，推理后 RSS 为 1.139/6.253 GiB。进程显存仍是整卡基线差值估计，KV 阶段存在单独 jitter 尖峰；本轮单次端点结果没有晋升论文正文，C2 仍须完成全部档位、上下文和重复轮次。

该快照只确认当前依赖和绑定加载链路。模型加载、CUDA offload、真实推理、迁移以及 Forge 仍须用对应验证单独证明。

## 8. 配置与环境变量

`config.py` 优先导入父仓库 `config.integrated_config.get_settings()`；独立部署导入失败时，才构造回退设置。高频变量：

| 变量 | 用途 |
|---|---|
| `XIAOYOU_LLM_MODEL_PATH` | 多个论文实验的真实 GGUF 路径 |
| `XIAOYOU_BENCH_LLM_PATH` | benchmark/演示回退的 GGUF 路径 |
| `XIAOYOU_BENCH_N_CTX` | 上下文长度 |
| `XIAOYOU_BENCH_N_GPU_LAYERS` | GPU offload 层数 |
| `XIAOYOU_USE_CPP_SCHEDULER` | 启用 C++ 调度器 |
| `XIAOYOU_USE_CPP_FOR_LLM` | LLM 走 C++ worker |
| `XIAOYOU_USE_CPP_FOR_IMAGE` | 图像走 C++ 调度路径 |
| `XIAOYOU_CPP_LLM_BACKEND` | C++ 引擎的 LLM backend |
| `XIAOYOU_FORGE_BASE_URL` | Forge API 地址 |
| `XIAOYOU_MVP_IMAGE_BACKEND` | 图像后端选择 |
| `XIAOYOU_ALLOW_CPP_LLM_WORKER` | 显式允许实验性 C++ LLM worker |

实验记录不得写入密钥；环境 manifest 只记录必要、非敏感的运行参数。

## 9. 验证分层

| 层级 | 示例 | 结论边界 |
|---|---|---|
| 文档/静态 | `tests/scripts/verify_project_context.py` | 索引存在、路径有效、入口一致 |
| Mock 单元 | `tests/test_chat_service.py`, `tests/test_phase_aware_quick.py` | 协议和调度语义 |
| 弹性驻留 A1/环境 | `tests/elastic_residency/` + WSL2 `smoke_a1.py` / `audio_playback_probe.py` | GPU/jitter/CPU 首样本/provenance 硬门禁、物理核租约/归还、实际线程 affinity 与跨宿主真实音频输出 |
| 弹性驻留 B1 | WSL2 `tts_component_offload.py` + `b1_tts_components_20260829_verified_v2` | T0–T3 组件发现、实际设备、code 传输、分段时间、显存证据和非静音单文本 WAV；不证明 Pareto |
| 弹性驻留 B2 | WSL2 `tts_component_offload_matrix.py` + `b2_tts_matrix_20260829_verified_v1` | 60 条 TTS 组件放置矩阵、分组统计与条件式 Pareto；不证明端到端无 underrun |
| 弹性驻留 C1 | WSL2 `llm_residency_curve.py` + `c1_llm_residency_512_20260829_verified_v1` | K0/FULL 真实端点、精确 token/gap、阶段资源与 KV 保存恢复；不代替 C2 多轮曲线 |
| 恢复/持久化 | `tests/verify_state_backplane_recovery.py` | mmap 读写和恢复闭合 |
| C++ 绑定 | `cpp_scheduler/verify_bindings.py`, `tests/test_cpp_scheduler_minimal.py` | 扩展导入与 API 契约 |
| 真实 LLM | `tests/test_llama_gpu.py`, migration experiments | 当前模型/驱动下的推理与迁移 |
| Forge 端到端 | `run_forge_experiments.py` | 图像服务、显存竞争和真实 workload |
| 论文复现 | 多次迭代 + manifest + 图表重建 | 可提交的统计证据 |

## 10. 已知维护风险

- 父仓库 `.gitignore` 含 `/legacy/`，当前目录全部被父仓库 Git 忽略；需要版本管理时必须先决定独立仓库或调整父仓库策略，不能误以为文档已被提交。
- `cpp_scheduler/build/` 约数百 MB，包含第三方源码和构建产物；搜索、统计和评审时必须排除。
- `data/` 同时含源码适配器和本地 mmap/日志数据，修改时要精确到文件，不能整目录清理。
- `TODO.md` 记录的是 2026-02 已完成事项与当时的重跑建议，不是当前唯一 backlog。
- `docs/` 下三份 NVRAM/C++ 评估文档包含历史设想和过强结论；当前任务不得直接引用为现状。
- README 曾长期漂移。新增或移动入口后，应先更新本上下文与验证脚本，再更新目录 README。

## 11. 文档维护约定

- 模块职责、入口或目录变化：更新 `PROJECT_INDEX.md` 和本文件第 3-4 节。
- 数据流、状态机或迁移协议变化：更新第 5 节和对应实验。
- 实验口径、正文数据目录或晋升流程变化：更新第 6 节、`experiments/README.md`、`paper/experiment_data/README.md`。
- 环境、模型或同步方法变化：更新第 7-8 节和局部 `AGENTS.md`。
- 每次文档更新运行 `tests/scripts/verify_project_context.py`，并对变更路径做最小可用测试。
