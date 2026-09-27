# EdgeSched / MVP Core 项目上下文

small 9B 离线筛选 `20260907_204516_653987` 已复算一致：T0/K20、T3/K30 仅为容量候选，长上下文余量不足 16 MiB。下一步用户运行 `scripts/experiments/run_small_9b_tts_compat.ps1`，C512/K18 配 T0/T3 各 1 轮；加载期另需验证，不冻结阈值，见 [验收与命令](SMALL_9B_NEXT_COMBINATIONS_20260907.md)。

small 9B 视觉生命周期五轮 `20260907_171635` 已 5/5、swap 0；加载/预热 0.481 s、编码 0.147 s、释放 0.011 s，见 [阶段画像](SMALL_9B_VISION_PROFILE_20260907.md)。这是独立进程画像，保温/提前释放策略仍待验证。

完整曲线批次 `20260907_143917` 已离线逐条核验，288/288、swap 0，见 [16 档曲线与 KV 验收](SMALL_9B_CURVES_20260907.md)。

> **2026-09-07 9B 版本选定**：后续主模型为 small `1a35ab89…` + F16 projector `e36da9b9…`。纯文本对比 12/12、small 视觉 Answer 1/1 通过，固定图形内容描述完整；专用视觉启动器默认 Small，Large 可显式复测。新版本 C2 90/90、C3 150/150、KV 48/48 已通过，4B 对照与旧哈希证据保留。单图不是完整质量评测或 TTS 共置结论。详见 [选型与验收](NINE_B_MODEL_COMPARISON.md)。

**文档版本**：1.0  
**最后核对**：2026-08-31
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
| 论文数据 | 新版 `01_paper_main` 当前未晋升数据；旧链条已归档 | 明确隔离旧稿与新证据，防止误引用 | `experiment_results/` 中的探索结果已经成为正文证据 |

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

### GPU 计算份额（SM-A，实验期基础设施）

这一组模块位于 `experiments/elastic_residency/sm/` 子包，只服务于实验，不接入任何在线调度路径。

| 文件 | 职责 |
|---|---|
| `sm/sm_capability.py` | SM-A 探针入口：基线 → MPS 可用性 → 逐 percentage 独立子进程 → 清理 → manifest/JSONL/summary/failures/events |
| `sm/sm_profiles.py` | 枚举、后端→能力映射、百分比/SM 校验原语、`estimate_requested_sm_count`（只生成请求值） |
| `sm/sm_share_profile.py` | `GpuShareProfile` 与 `unsupported_profile`（fail-closed） |
| `sm/sm_partition_rules.py` | `GpuPartitionCapability`、`GpuPartitionProbeResult` 与独占 SM 预算规则 |
| `sm/mps_protocol.py` | interactive 协议层：`CommandSpec`/`CommandResult`、stdin 编码、`subprocess_runner`、客户端环境、`quit` 成败判定 |
| `sm/mps_commands.py` | 常量、`MpsDaemonHealth`、`ControllerReadiness`、解析与 PID 文件读取 |
| `sm/mps_runtime.py` | control daemon 启动器：探测二进制、`-d` 启动、私有 pipe/log 目录 |
| `sm/mps_daemon.py` | daemon 句柄：controller 就绪判定、客户端环境、清理与退出证据 |
| `sm/mps_queries.py` | `get_server_list` / `get_client_list` / `ps` / `start_server` 查询；持久记录 server PID 与进程身份 |
| `sm/mps_shutdown.py` | 关闭阶段证据：controller 退出后按进程身份核对 server，`reached_quiescent` 判定与超时 |
| `sm/mps_artifacts.py` | daemon 日志归档、残迹检查与证据快照 |
| `sm/sm_client_probe.py` / `sm/sm_client_workload.py` | 独立进程客户端：env 先于 CUDA 初始化、context 有效 SM 数、短 workload 同步 |
| `sm/cuda_exec_affinity.py` | `cuCtxGetExecAffinity` 与驱动版本只读查询 |
| `sm/green_context_types.py` / `sm/green_context_probe.py` | Green Context 符号表/结构体自检与只读探测；默认结论为 `UNSUPPORTED` |
| `sm/sm_probe_env.py` / `sm/sm_client_process.py` / `sm/sm_client_runner.py` / `sm/sm_probe_records.py` / `sm/sm_probe_verdict.py` | 环境采集、子进程执行、客户端编排、记录构造与 Go/No-Go 判定 |

### 多模态端点门（4B + `mmproj`，实验期基础设施）

这一组模块位于 `experiments/elastic_residency/vision/`，只服务于实验，不接入任何在线调度路径。

| 文件 | 职责 |
|---|---|
| `gguf_reader.py` | GGUF 头部 KV 只读解析（纯标准库，不加载模型、不占显存） |
| `vision_artifacts.py` | 主 GGUF / mmproj 画像、哈希与对应关系核验 |
| `fixed_image.py` | 确定性固定测试图片；`pixel_payload_sha256` 是内容身份，`file_sha256` 只作记录 |
| `vision_endpoint_gate.py` | 真实测量前的 artifact + mmproj 能力硬门 |
| `mtmd_runtime.py` | libmtmd projector warm、bitmap 解码、tokenize 与释放 |
| `mtmd_eval.py` | chunk 级真实执行（`split_encode` / `chunk_single`）、状态同步与固定长度解码 |
| `vision_endpoint_phases.py` | 文本端点与视觉端点的真实执行阶段 |
| `vision_evidence.py` | 证据语义、驻留状态、prompt 构造与视觉门禁原语 |
| `vision_endpoint_checks.py` / `vision_endpoint_contracts.py` | 单轮硬门禁装配；run_config 与 iteration 记录装配 |
| `vision_endpoint_cli.py` / `vision_endpoint_child.py` / `vision_endpoint_report.py` / `vision_endpoint_smoke.py` | 命令行契约、单轮测量子进程、单元级聚合与端点 smoke 编排 |

语义边界：

- `TEXT_ONLY`：projector 未加载，prompt 为 C1/C2 口径的固定 token 填充；
- `TEXT_ONLY_CHAT`：projector 未加载，但使用与视觉轮次相同的对话 prompt，作为同 prompt 对照；
- `VISION_WARM_GPU` / `VISION_WARM_CPU`：projector 已 warm 且对固定图片实际执行，区别只在 projector 所在设备；
- `visual_token_total` 来自 `mtmd_input_chunk_get_n_tokens`，`visual_position_total` 是上下文实际推进的位置数；启用 M-RoPE 时两者不互相替代；
- `split_encode` 才有 `projector_encode_s` 与 `visual_embedding_available` 的直接证据，`chunk_single` 只有返回码与位置推进的间接证据；
- projector 显存/RSS 只能用 warm 前后的 direct NVML 整卡差值与进程 RSS 差值描述，禁止把 `mmproj` 文件大小当作运行期占用；
- 视觉 prompt 与 `TEXT_ONLY` 填充 prompt 语义不同，TTFT/prefill 不可跨状态直接比较。

语义边界：

- `UNMANAGED`：普通 CUDA 共享，没有可验证份额。
- `MPS_CAP`：`CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` 只是客户端执行资源上限，不是独占 SM，不参与 “SM 数之和 <= 物理 SM” 的准入判断。
- `MPS_STATIC` / `GREEN_CONTEXT`：只有真实验证成功后才按空间分区处理；当前没有可用的静态分区后端。
- `UNSUPPORTED`：后端不存在、初始化失败或实际份额无法验证，禁止静默退回普通 CUDA。
- `effective_sm_count` 只接受客户端 CUDA context 的实测结果；比例推算值只能写入 `requested_sm_count`。
- 显存限制与 SM 限制分别建模，互不代表。

真实 NVIDIA API 契约（已按驱动头文件/文档核对，违反会导致结论失真）：

- `cuCtxGetExecAffinity(CUexecAffinityParam*, CUexecAffinityType)`：第二个参数是整数枚举
  （`CU_EXEC_AFFINITY_TYPE_SM_COUNT = 1`），不是元素个数指针；`argtypes` 用 `ctypes.c_int`。
  `CUexecAffinityParam` 的 `type`/`numSms` 偏移由 C ABI 保证（0 / 4），联合体整体大小未核对，
  需在结果中保留 `layout.union_total_size_verified = false`，不得据此宣称设备能力已验证。
- `nvidia-cuda-mps-control` 的 `get_server_list` / `get_client_list` / `ps` / `quit` /
  `start_server` 是 interactive control commands，必须写入 **stdin**；argv 只包含可执行文件路径。
  全程禁用 `shell=True`，`CommandResult` 分别记录 `argv`、`stdin_command`、`returncode`、输出与超时。
- `-d` 只拉起 control daemon；MPS server 由首个 CUDA client 按需启动。因此就绪判据是私有 pipe
  目录中 `nvidia-cuda-mps-control.pid`（及其中 PID 存活），**不能**在启动阶段等待 server 出现。
  顺序为：启动 controller → 等 PID 文件 → 启动客户端并保持 context → 查询 server/client →
  客户端退出 → stdin 发送 `quit` → 验证 controller/server 消失 → 清理私有目录。
- 清理必须 fail-closed：`quit` 只有在 `returncode == 0` 且未超时且无 OS 错误时才算成功，
  `returncode=None` 一律视为失败；等待 controller 退出超时会抛 `MpsRuntimeError`，不会静默
  置为成功；controller 未确认退出时**保留**私有 pipe/log 目录与 PID 供人工处理，只归档日志。
- server 与 controller 观察状态必须**持久记录**在句柄上（`observed_server_identities` /
  `controller_snapshot`，pid → starttime/executable），不能从单次 post-quit 证据推断。
  **controller 退出之后不再调用 `get_server_list`**（前端可能重新拉起 control daemon），
  controller 与 server 都改用退出前记录的进程身份核对是否仍存活。
- `reached_quiescent=True` 必须同时满足：`controller_status` 为 `gone`/`reused`、所有 server 为
  `gone`/`reused`、没有 unverifiable controller/server、没有查询错误。进程身份无法确认时
  fail-closed 并保留私有目录与结构化原因；PID 被其他进程复用时（starttime/executable 不同）
  判为 `reused`，不误判为残留。
- `get_server_list` 超时 / OS 错误 / `returncode=None` / 非零返回码在 controller 仍在时写入
  `last_server_query_error` 与 `post_quit_evidence.server_query_error`，并令
  `reached_quiescent=False`；查询失败绝不能折叠成空列表。
- 物理 SM 与 context 可见 SM 必须分开：不受控 CUDA 基线给出 `physical_total_sms`；
  MPS 客户端的 `torch.cuda.get_device_properties(0).multi_processor_count` 只是
  `context_visible_sm_count`（可能被 active-thread percentage 限制），
  所有 `requested_sm_count` 都用基线物理总数计算，MPS 客户端属性不得覆盖物理总量。
- Green Context 的 stream 符号是 `cuGreenCtxStreamCreate`（不存在 `cuStreamCreateWithGreenCtx`）；
  `CUdevResource` 布局未核对，设备资源查询只能记为 `device_sm_count_unverified` 并保持
  `UNSUPPORTED / layout_unverified`，不得崩溃也不得宣称支持。

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
| `experiments/elastic_residency/` | 弹性驻留统一契约、物理核租约、资源与调度 jitter 采样、TTS swap/NVML 运行期证据及后续可行性实验 |
| `experiments/elastic_residency/sm/` | GPU 计算份额（SM-A）子包：数据契约、MPS 生命周期、客户端探针、探针编排与 Green Context 只读探测 |
| `experiments/elastic_residency/vision/` | 多模态端点门子包：GGUF 只读解析、artifact 对应关系、固定图片、libmtmd 真实执行、视觉证据语义与端点 smoke |
| `experiment_results/` | 工作区运行产物和中间结果 |
| `paper/experiment_data/01_paper_main/` | 新版正文聚合数据入口；当前尚未晋升数据 |
| `paper/experiment_data/02_auxiliary/` | 新版辅助说明数据入口 |
| `paper/experiment_data/03_raw_iterations/` | 新版原始迭代证据入口 |
| `paper/thesis_experiment_guide/` | 按任务拆分的研究设计、实验协议、证据边界与当前门禁真源；新会话先读目录索引和第 10 模块 |
| `paper/rewrite/` | 按新版总指南重写的正文工作区 |
| `paper/archive/pre_rewrite_20260830/` | 重写前主稿、图表、工具和数据完整快照 |
| `paper/archive/legacy_materials_before_rewrite/` | 更早的历史稿件和产物，只读追溯 |

弹性驻留工作产物采用“当前批次留在阶段目录、历史批次进入 `_archive/{superseded,stopped,debug_smoke}`”的布局。2026-08-30 的完整分类、移动后路径和禁止混用口径见 [弹性驻留运行结果目录](ELASTIC_RESIDENCY_RESULT_CATALOG_20260830.md)。归档只用于本地追溯，不改变 `paper/experiment_data/` 的人工晋升规则。

### 数据晋升流程

1. 在 WSL2 冻结代码、模型、环境变量和实验参数。
2. 先运行单次 smoke，确认真实后端没有回退到 Mock。
3. 按论文要求运行多次迭代，保留每次原始 JSON。
4. 生成 manifest，记录随机种子、环境与配置哈希。
5. 聚合到候选数据，检查缺失值、异常值和统计口径。
6. 只有人工确认后，才向新版 `01_paper_main/` 晋升数据并重新生成图表/表格。
7. 对照 `paper/rewrite/OUTLINE.md` 和后续新版 Evaluation 源文件，确认每个数字可追溯。

## 7. 环境与同步边界

用户已授权并完成删除 WSL `Qwen3.5-9B-q4_k_m.gguf`（large `423fb3e5…`，约 5.56 GiB）；small 独立副本和 mmproj 保留。
历史 Large 入口需恢复原哈希文件才能复测，不能把路径存在性当作当前事实。
后续离线入口 `run_small_9b_screen.ps1` / `screen_small_9b.py` 联合整理文本、FULL 视觉及 Diffusion GPU 候选，保留 16G 主机门；组合路线见 [计划](SMALL_9B_NEXT_COMBINATIONS_20260907.md)。

论文明确将视觉编码/投影组件 V 纳入通用阶段与驻留状态，V+L 构成 VL 模型，不再视为 L 永久附加的显存项。
q_V、r_V 和加载/释放代价见指南 01–04；S0/S1/S2 的现有图像队列指 Diffusion，视觉请求不自动触发 LLM 逐出。
当前实验是每轮加载 V、回答后释放；跨请求保温与编码后提前释放尚未接入调度器。此轮只修订论文模型和证据边界，未改运行代码。

small 9B 完整画像手动入口 `scripts/experiments/run_9b_small_curves.ps1`：固定独立模型路径/SHA，保存共享源码快照，依次运行 C2/C3/KV（90/150/48）；只重建纯文本画像，失败即停，实际实验由用户启动。

9B 换版本前的手动选型入口为 `scripts/experiments/run_9b_comparison.ps1`。
由 `compare_9b_models.py` 复用原驻留测量器，`compare_9b_support.py` 管理安全复制与汇总；
新 small 副本放在 `models/llm/comparison_1a35ab89/model.gguf`，不覆盖 large 或 4B。
两模型均在 WSL 原生盘做 FULL/K0 纯文本比较，完整协议与哈希见 [对比说明](NINE_B_MODEL_COMPARISON.md)。
用户完成 `compare_9b_20260907_023703`：12/12、swap 0、原始记录离线复核一致。
small FULL 显存峰值少 90 MiB、RSS 峰值少 331.45 MiB、decode +2.05%；仅为三轮选型诊断，small 视觉 Answer 随后已通过，专用视觉入口默认 Small。
视觉手动入口支持 `-ModelVariant Small -Stage Answer`，读取已准备的独立副本并校验固定 SHA。
单侧 basename 缺失时按分隔符归一化的名称与非空精确基础来源核验，结构/哈希门禁保留；small 真实文件头和用户实际视觉 Answer 均通过（20260907_142945）。

9B 原生目录复跑 `v1_vision_9b_probe_20260906_204035` 已 2/2 通过；用户随后完成 Answer 补测（1/1 技术通过，漏黑边）
和 Profile 五轮画像（5/5 全通过）。当前先离线筛容量与阶段切换候选，不重复运行基础画像。
两个阶段生成预算与 n_ctx 不同，分别保存，不能混为同一性能曲线；详见 [协议](VISION_9B_PROBE_20260906.md)。

| 用途 | 路径/环境 |
|---|---|
| **主开发副本（唯一权威）** | `/home/leslie/mvp_core`，本机原生 Linux（Ubuntu-24.04-E，非 WSL2）；编辑、运行、git 提交/推送均在此处，`.git` 在工作树内（origin = `https://github.com/hakituo/EdgeSched`） |
| Windows 旧副本（已停用） | `D:\AI\xiaoyou-core\legacy\mvp_core`：2026-09-25 前的权威仓库，`.git` 已复制到 Linux 并校验；此后不再参与提交，`sync_to_wsl2_safe.ps1` 流程废止，保留仅作历史备份 |
| 行尾/权限 | 仓库根 `.gitattributes` 统一文本 LF（2026-09-25 全量归一）；`core.fileMode=false`、`core.symlinks=true` |
| 主实验环境 | `/home/leslie/.venvs/mvp_core-cu128`（稳定路径，**2026-09-25 起为软链**，实际指向 `/home/leslie/xiaoyou-core/.venv`：uv 管理 / Python 3.13.15 / **torch 2.11.0+cu128 / torchvision 0.26.0+cu128（2026-09-27 由 cu130 迁移，见 `docs/UPDATES.md` 2026-09-27 条目）** / transformers 4.57.3 / llama_cpp 0.3.35(CUDA 12.8 源码编译) / faster-qwen3-tts 0.2.6 / qwen-tts 0.1.1 / sounddevice / faster-whisper / onnxruntime。mvp 侧补装 diffusers 0.37.1、qwen-vl-utils、loguru、safetensors 0.8、pybind11。numpy 保持 1.26.4 不动。**CUDA 12 运行时不依赖系统 toolkit**：`nvidia-*-cu12` pip 包提供 `libcudart.so.12` / `libcublas.so.12` 等，`.../site-packages/nvidia/*/lib` 已写入 `/etc/ld.so.conf.d/xiaoyou-nvidia-cu12.conf` 并 `ldconfig`，`llama_cpp` 与 `cpp_scheduler` 的 `.so` 均可直接加载） |
| 遗留环境 | ① 2026-09-17 删除的 Python 3.10.19 / torch 2.10 / transformers 4.57.3 旧主环境；② 2026-09-17–24 软链目标 `/home/leslie/qwen3tts-cugraph-probe/faster-qwen3-tts/.venv`（Python 3.12 / torch 2.11+cu128 / transformers 5.15.1 的加速隔离环境）在系统迁移后已不存在（现为空的 3.10 venv）。旧后端脚本（`--tts-backend upstream_dynamic`、`run_small_9b_tts_compat/t4_isolate`、`services/tts/qwen3_tts_worker.py`）重跑前仍须按 TTS 文档准备环境；但 `cuda_graph_static` 加速后端**已验证可直接在共享 3.13 venv 运行**（faster-qwen3-tts 0.2.6 + transformers 4.57.3，稳态 RTF 0.3136，与历史 5.x 隔离环境 0.303–0.317 一致，见 7.1 与 TTS 文档；不要升级 transformers 5.x） |
| Forge 环境 | `/home/leslie/.venvs/forge-cu128`（**2026-09-27 重装系统后再建**：`uv python install 3.10` + `uv venv --python 3.10 --seed` 创建，预装 torch 2.10.0+cu128 / torchvision 0.25.0+cu128（RTX 5070 不能用 launch.py 默认的 2.3.1+cu121），`setuptools` 已钉 69.5.1 以规避 84 版移除 `pkg_resources` 导致的老 CLIP 构建失败；其余依赖仍由 Forge `launch.py` 首次启动时按 `requirements_versions.txt` 安装，扩展装完后须把 numpy 钉回 1.26.2，CLIP 需免隔离预装，坑点详见 `docs/UPDATES.md` 2026-09-25 与 2026-09-27 条目。**Forge 本体扩展尚未安装、真实出图 smoke 未跑**） |
| GGUF 模型 | `/home/leslie/mvp_core/models/llm` 下 4 个文件均为软链，真实权重在 970 EVO `/media/leslie/970EVO/xiaoyou-models/llm`（2026-09-25 sha256 四份逐一比对一致后替换；已实测 llama.cpp 经软链 mmap+生成正常）。**2026-09-27 起该盘被改作 Ventoy，软链当前断裂；权重本体保留，待迁回后恢复**；代码路径不变，可用性依赖 970 EVO 挂载 |
| 当前 9B + 视觉副本 | 上述原生目录中的 `Qwen3.5-9B-q4_k_m.gguf` / `Qwen3.5-9B-f16.mmproj`；2026-09-06 从 D: 复制并核验两侧 SHA-256，见 [复核记录](VISION_9B_PROBE_20260906.md) |
| TTS 模型 | `/home/leslie/xiaoyou-core/models/Qwen3-TTS-12Hz-0.6B-Base`（2026-09-25 起全部实验脚本默认指向；与 mvp_core 下旧副本 `diff -rq` 一致；可用 `XIAOYOU_BENCH_TTS_PATH` 覆盖） |
| 辅助模型 | `/home/leslie/mvp_core/models/auxiliary`（Faster-Whisper Small、BGE-small-zh-v1.5） |
| 图像模型 | `/home/leslie/mvp_core/models/Image`；970 EVO 额外权重（`/media/leslie/970EVO/xiaoyou-models/Image` 的 krea2/z-image checkpoint、4 个 VAE、openposeXL controlnet）已软链进 Forge `models/`。**2026-09-27 起 970 EVO 未接入，上述软链处于断裂状态**，迁回后恢复 |
| Forge 服务 | `/home/leslie/mvp_core/stable-diffusion-webui-forge-main/` |

**2026-09-25 起不再有 Windows→Linux 同步流程**：主开发与 git 操作都在 `/home/leslie/mvp_core` 原生 Linux 完成，直接 `git add`（显式路径）/ `git commit -m "一句话说明"` / `git push origin main`。`scripts/sync_to_wsl2_safe.ps1` 已废止，禁止再用于提交；其 `/models/` 排除语义只作为 `.gitignore` 的历史背景（根目录模型排除写成 `/models/`，避免误伤 `domain/models/`）。提交信息要求不变：一句话写清改动、禁止 auto/更新/同步 等无信息文案、细节写 `docs/UPDATES.md`；存在其他会话未提交改动时只暂存本任务文件。

### 7.1 当前 Linux 运行快照

2026-09-27（**用户重装系统后重建环境；本段为当前有效状态**）：

- 系统重装为 Ubuntu 24.04.5（内核 7.0.0-31-generic），`/home` 由备份恢复；原根分区 `nvme1n1p2`（ext4）现为 NTFS，当前根为 `nvme1n1p3`（ext4，278 G）。
- NVIDIA 驱动重装 `nvidia-driver-595-open`（595.91.07）：dkms 编译通过、`nvidia-smi` 正常、`/dev/nvidia*` 齐全；GPU 仍为 RTX 5070 Laptop（8151 MiB，sm_120）。
- 主 venv 由 **cu130 迁移到 cu128**：torch `2.12.0+cu130` → **`2.11.0+cu128`**、torchvision `0.27.0+cu130` → `0.26.0+cu128`、torchaudio `2.11.0+cu128`，cu13 依赖包全部移除；`torch.cuda.is_available()=True` 并实测矩阵运算通过。迁移动因：cu128 分支没有 torch 2.12，而本机 C++ 产物与 `llama-cpp-python` 都按 CUDA 12 编译，统一到 12.x 后无需安装系统 CUDA toolkit。
- **系统不再需要 CUDA toolkit**：CUDA 12 运行时全部来自 venv 内 `nvidia-*-cu12` pip 包（`libcudart.so.12`、`libcublas.so.12`、`libcudnn.so.9` 等），其 `.../site-packages/nvidia/*/lib` 已写入 `/etc/ld.so.conf.d/xiaoyou-nvidia-cu12.conf` 并 `ldconfig`；此后 `llama_cpp`（0.3.35，`GPU offload=True`）与 `cpp_scheduler` 的 `scheduler_py` / `libggml-cuda.so` 均无需 `LD_LIBRARY_PATH` 即可加载，`ldd` 无 `not found`。
- Forge 独立环境 `/home/leslie/.venvs/forge-cu128` 用 uv 托管的 Python 3.10.21 重建（旧 venv 因指向已不存在的系统 `/usr/bin/python3.10` 而失效），预装 torch 2.10.0+cu128 / torchvision 0.25.0+cu128、`setuptools` 钉 69.5.1；Forge 本体扩展待 `launch.py` 首次启动安装，真实出图 smoke 未跑。
- 970 EVO（`/media/leslie/970EVO`）已改作 Ventoy 用途，`models/llm/` 四个 GGUF 软链与 Forge 额外权重软链当前断裂；权重本体保留，迁回后恢复，本次不重跑依赖它们的实验。

2026-09-25（系统迁移后重建环境，**历史快照，其 cu130 环境已被上条取代**）：

- 主环境改为直接复用父项目 venv `/home/leslie/xiaoyou-core/.venv`（uv，Python 3.13.15，torch 2.12.0+cu130，驱动 595.91.07 / CUDA 13.2），`/home/leslie/.venvs/mvp_core-cu128` 重建为指向它的软链；`torch.cuda.is_available()` 正常识别 RTX 5070 Laptop（8 GiB，sm_120）。
- 在共享 venv 中用 `/usr/local/cuda-12.8` 源码编译安装 `llama-cpp-python==0.3.35`（`GGML_CUDA=on`、`GGML_CUDA_ARCHITECTURES=120`），`llama_supports_gpu_offload()=True` 并枚举出本卡；另补 diffusers 0.37.1、qwen-vl-utils 0.0.14、loguru、safetensors 0.8.0、torchvision 0.27.0+cu130、pybind11。未升级 numpy（保持 1.26.4），未安装代码零导入的 blendmodes/peft。
- Forge 独立环境 `/home/leslie/.venvs/forge-cu128` 用 Python 3.10.21 重建并预装 cu128 torch（RTX 5070 不能用 launch.py 默认的 2.3.1+cu121）。
- `cpp_scheduler/build_wsl2.sh` 改为 3.13 共享 venv + pybind11 cmake 路径 + `CMAKE_CUDA_ARCHITECTURES=120`；系统补装 `libuv1-dev` 后全量构建通过，`scheduler_py.cpython-313-x86_64-linux-gnu.so` 可导入。
- 加速 TTS 后端无需 transformers 5.x：共享 venv 内 faster-qwen3-tts 0.2.6（依赖声明 `transformers>=4.57`）在 4.57.3 下实测两图捕获成功、经 mvp 集成入口合成有效音频，稳态 RTF 0.3136 与历史 5.x 隔离环境（0.303–0.317）一致；S0 v2 `cuda_graph_static` 在现环境可直接重跑。刻意不升级 transformers（qwen-tts 0.1.1 钉死 4.57.3，且是 xiaoyou-core 主项目环境）。
- TTS 权重默认路径切到 `/home/leslie/xiaoyou-core/models/Qwen3-TTS-12Hz-0.6B-Base`（8 个实验入口 + TTS 文档；旧 mvp_core 副本内容一致保留）；模型存储收敛到外挂盘：970 EVO 上的 krea2/z-image checkpoint、VAE、controlnet 共 7 个权重软链进 Forge `models/`；`models/llm/` 下 4 个 GGUF/mmproj（约 9.3 G）sha256 校验一致后也替换为指向 970 EVO 的软链（llama.cpp 经软链 mmap+生成实测正常）。上述均依赖 970 EVO 挂载。
- **git 主仓库在 Linux 重建**：先从 Windows 副本复制 `.git`（NTFS stat 导致 status 长期虚报 200+ 假 M），最终按用户意见改为从 GitHub `git clone --no-checkout` 取干净 `.git`（HEAD bcb45b0 与本地一致）并 `git reset` 重建 index，幻影消失；新增 `.gitattributes` 并把已跟踪文本全量归一 LF。最终未提交差异：27 改 / 6 删 / 148 新增（迁移期遗留，去留另行决定）；`core.fileMode=false`、`core.symlinks=true`、`core.ignorecase=false`。`experiment_results` 同时从 D 盘软链改为本地实体目录（1.7 G，复制后与 HEAD 一致）。WSL2 双副本与 `sync_to_wsl2_safe.ps1` 自动提交流程废止。

2026-08-28 在 `Ubuntu-24.04-E` 做过只读核验：

- `/home/leslie/.venvs/mvp_core-cu128` 能导入 `llama_cpp`；
- 配置 `cpp_scheduler/build/bin` 与 `cpp_scheduler/build/llama.cpp/src` 到 `LD_LIBRARY_PATH` 后，能导入 `/home/leslie/mvp_core/cpp_scheduler/build/scheduler_py.cpython-310-x86_64-linux-gnu.so`；
- Windows `venv_core` 中 benchmark 降级到 Mock，只表示 Windows 验证环境没有 `llama_cpp`，不能据此判断 WSL2 缺少依赖。
- `faster-whisper==1.2.1`、`ctranslate2==4.7.2`、`onnxruntime==1.23.2`、`sounddevice==0.5.5` 与 `libportaudio2` 已安装；Faster-Whisper CPU int8 真实接口 smoke 可运行。
- WSL2 已安装 ALSA→Pulse 桥接并能通过 WSLg Pulse 打开真实 PortAudio 输出；若 WSLg 音频端点失效，`experiments/elastic_residency/audio_playback_probe.py` 可回退到 Windows WASAPI，并显式记录重采样与传输路径。两条路径的 1 s 静音 smoke 均为 0 underflow，但正式结论仍须使用真实 TTS chunk 和并发负载。
- `blendmodes==2025`、`pillow==10.4.0` 与当前 NumPy/Gradio 形成兼容集合；WSL2 `pip check` 无破损依赖，Qwen-TTS、Gradio 与 Torch 导入通过。
- A1 最终 smoke `a1_smoke_20260829_verified_v7` 基于干净且已推送的 commit `ae68608` 通过全部 17 项硬门禁：GPU 首样本在时间线前就绪，41/41 点均可用且最大年龄 422.719 ms；资源 jitter p95/max 为 0.309/1.680 ms，事件循环为 2.165/3.081 ms；首个 CPU 样本被明确排除，swap 为 0，线程 affinity 正确且结束后 16 个物理核全部归还。旧 v6 已降级为调试记录。A1 仍不包含 TTS/LLM 正式负载或真实并发音频。
- B1 最终 smoke `b1_tts_components_20260829_verified_v2` 基于干净 commit `38f4d81` 使用真实 Qwen3-TTS/CUDA 后端，T0–T3 为 4/4 通过。实际组件设备、prompt device、`cuda:0→cpu` code 传输、分段时间、分阶段 PyTorch 显存峰值和非静音 WAV 均已验证。WSL 的进程显存接口返回 `[N/A]`，因此记录明确 unavailable，并以整卡相对空闲基线增量交叉估计；不把该估计伪装为精确进程显存。B1 是单轮冷态可行性证据，当前不能用于 Pareto 或论文性能结论。
- ~~B2 `b2_tts_matrix_20260829_verified_v1` 基于干净 commit `ce024a0` 完成 60/60 条真实 TTS 迭代。~~ **该批产物已于 2026-09-17 删除**，B2 不再引用其任何数值；旧后端 T0/T3 条件式 Pareto 结论一并退役（B2 现指当前接入后端加速整模型 GPU 的 T0 句长画像，见 `experiments/README.md` 第 9 节）。
- C1 `c1_llm_residency_512_20260829_verified_v1` 基于干净 commit `234d2df`，用真实 32 层 Qwen3.5-9B GGUF 完成 K0/FULL 的 512-token 端点 smoke，两端均通过 19/19 项门禁并正确保存/恢复 640-token KV state。FULL 相对 K0 的 TTFT 为 0.326/10.659 s、decode 为 51.03/2.77 tok/s，GPU 增量估计为 5543/127 MiB，推理后 RSS 为 1.139/6.253 GiB。进程显存仍是整卡基线差值估计，KV 阶段存在单独 jitter 尖峰；本轮单次端点结果没有晋升论文正文，C2 仍须完成全部档位、上下文和重复轮次。
- 2026-08-29 复核发现旧弹性驻留 GPU 时间线由 `nvidia-smi` CLI 子进程轮询，长矩阵中探针耗时会接近或超过采样周期。正式后端已改为 ctypes direct NVML；C2 direct v2（90 条）、C3 direct v2（150 条）和 B2 direct v2（30 条）已形成同后端当前证据，旧曲线只保留为工程基线。2026-08-31 Forge direct NVML v1 也已通过逐窗新鲜度、swap 增长和核心归还门禁，`M_D=5484 MiB`；当前分析入口强制合并 C2+C3 完整 16 档后再筛选候选。TTS 的 PyTorch allocator 峰值仍是独立的框架内证据。
- ~~T0/T3 direct NVML v2 `b2_tts_pareto_matrix_20260831_direct_nvml_v2` 完成 30/30、0 failure/fallback。~~ **该批产物（连同 v1）已于 2026-09-17 删除**，其 T3 相对 T0 节省 1275.938 MiB、TTFA/RTF p95 分别增加 28.94%/25.56% 等数值不再作为依据；T3 属于当前加速后端不支持的组件放置维度，需先确认后端支持再重测。
- direct NVML C2 首轮在 4096-token/K24 暴露 512 ms 驱动查询长尾；旧固定网格会在查询完成后额外等待，把样本间隔放大到 756 ms。GPU sampler 现改为越过下一 tick 后立即补采，显存查询放在辅助 NVML 指标之后，并记录 `gpu_deadline_recovery_count`；750 ms 新鲜度硬门保持不变。修复前的 64 条不得续跑或与新曲线合并。
- 2026-08-30 KV 口径补强已完成正式校准：`c3_kv_cache_calibration_20260830_direct_nvml_v1` 基于干净 commit `820ee0f` 完成 48/48 条、0 failure，模型哈希与 C2/C3 相同。`kv_state_bytes` 是含 logits、embedding 与 KV memory 的完整序列化迁移负载，不是纯 GPU KV 显存；`n_ctx/KV tokens=640/2176/4224` 时，KV buffer 总量为 `24/72/136 MiB`，完整序列化状态为 `70.263/118.292/182.331 MiB`。历史 C2/C3 的 240 条 direct NVML 曲线无需重跑；论文旧稿的 1–2 GiB 数值已证伪并删除。
- 2026-08-31 Qwen3.5-9B LLM+TTS 正式共置 `llm_tts_colocated_candidates_20260831_direct_nvml_v1` 基于干净 commit `d1009158ed03a906933106b500d91d34a1d3ad3c` 完成 6 候选×5 轮，30/30、0 failure/fallback，真实后端、direct NVML、GPU 新鲜度、swap 与 provenance 全部通过。运行版把绝对 `P_colocated` 再加完整 `G_safe`，对 15 条 FULL+T3 记录重复计入 idle；原始测量仍有效，离线 `c3_h_tts_colocated_20260831_v1.json` 按 `(P_colocated-B_idle)+G_safe` 复算后六组均可行，正残差 max=0，冻结 9B `G_safe=501.282 MiB` 与 H_TTS（T0: 512/K24、2048/K20、4096/K20；T3: 全部 FULL）。随后 K6/K4+Forge 均触发 swap 门，9B `H_D` 不冻结；这些阈值不得复用为 4B+视觉结论。
- 2026-08-31 已实现独立的 LLM+Forge 共置实验链。Qwen3.5-9B 的 512/K6 smoke 与探索性 512/K4 smoke 均只有 swap 门失败，增长分别约 290.44 MiB 与 2070.32 MiB；两者的真实后端、PNG、direct NVML、新鲜度、KV、GPU 容量和 provenance 有效。降低 GPU 层数使主机换页显著恶化，因此不再继续 K2，也不运行 9B 正式 15 条矩阵；这些 artifact 保留为模型特定负面证据，`H_D` 未冻结。
- 历史选型（已由 2026-09-06 的 9B+视觉决定取代）：2026-08-31 主实验 LLM 决定迁移到 Qwen3.5-4B-Q4_K_M + 对应 BF16/F16 `mmproj`，用统一模型真实执行文本与图片理解；9B 保留为压力/NO_GO 对照。旧 `VLAdapter` 是独立 Qwen2-VL Transformers 路径，不能代表新链路；C2/C3、KV、LLM+TTS、LLM+Forge 入口仍是纯文本 `Llama(model_path=...)`。
- 2026-08-31 WSL2 只读核验：`llama-cpp-python==0.3.35` 随包提供 `libmtmd.so` 与 `llama_cpp/mtmd_cpp.py` 绑定，`llama_chat_format` 已含通用 `MTMDChatHandler`，因此当前环境**具备**该 Qwen3.5 多模态执行能力，无需新增依赖。
- 2026-08-31 4B artifact 门禁已通过（只读、未加载语言模型）：主 GGUF `general.architecture=qwen35`、`block_count=32`、`context_length=262144`、`embedding_length=2560`、`file_type=15`、426 张量、SHA-256 `00fe7986…ef11a4`、2740937888 B；`mmproj-BF16.gguf` 为 `general.architecture=clip`、`general.type=mmproj`、`clip.projector_type=qwen3vl_merger`、`clip.has_vision_encoder=true`、24 层视觉块、`image_size=768`、`patch_size=16`、`spatial_merge_size=2`、`projection_dim=2560`、298 张量、SHA-256 `302b92d5…21f3a6`、675569344 B。`projection_dim == embedding_length` 与 `general.basename`/`base_model.0.*` 一致构成对应关系证据；`mtmd_get_cap_from_file` 报告 `inp_vision=true`、`inp_audio=false`，media marker 为 `<__media__>`。
- 2026-08-31 已实现 `experiments/elastic_residency/vision/` 多模态端点门链路：artifact 门禁、固定 448×448 确定性图片（`pixel_payload_sha256=a1514034…43cb00`，`spec_sha256=d4c6fc1c…a0efb8`，按 patch/merge 期望 196 个视觉 token）、libmtmd projector warm/执行/释放、`TEXT_ONLY`/`TEXT_ONLY_CHAT`/`VISION_WARM_GPU`/`VISION_WARM_CPU` 四状态与 K0/FULL 端点 smoke 编排。Windows 侧静态与假后端测试 72 项通过，父进程门禁失败路径已在 WSL2 用合成 artifact 实跑验证。
- 2026-08-31 **多模态端点门真实通过**：`run_id=v1_vision_endpoint_20260831_direct_nvml_v3`、commit `c9a75da`，四单元（`K0/FULL × TEXT_ONLY/VISION_WARM_GPU`，512+128）全部 `passed=true`、0 失败、swap 0、direct NVML 且无 GPU CLI fallback。projector 在 GPU 上真实执行并输出 `[2560, 196, 1]` 嵌入；视觉 token 196；M-RoPE 下 196 个视觉 token 塌缩为 14 个序列位置，KV 位置 167/167 精确；模型文字输出正确描述固定图片。projector warm 0.28–0.44 s、free 0.008–0.011 s，warm GPU 增量 710–870 MiB、RSS +55.9 MB。FULL decode 86–90 tok/s，K0 decode 6.66–6.91 tok/s。三项真实运行暴露的绑定问题已修：`mtmd_decode_use_non_causal` 是 per-chunk 查询、回调位必须传 CFUNCTYPE 空实例、warmup 按最大像素预留 buffer 需明确记录。
- **证据边界（2026-09-04 更新）**：4B TEXT_ONLY C2 90/90、C3 150/150 与 KV 48/48 已通过；完整 16 档+B2 筛出 T0/T3 均可 FULL。C4096/FULL+T0 smoke 已 1/1 技术通过（峰值 6406.699 MiB、swap 0、正残差 0），但尚无正式重复共置矩阵，不能冻结 `G_safe/H_TTS`。新增 `colocated_audio_artifact.py` 独立负责 WAV 非空、大小及 SHA-256，后续每轮 child/JSONL/summary/CSV 均记录；旧 smoke 不改写。VISION_WARM 多轮画像、图像后端与 `H_D` 仍 pending。projector warm 的 GPU 增量实测有约 ±80 MiB 波动（CUDA 碎片 + 整卡其他进程），且包含 warmup 按 1472×1472 预留的 223.30 MiB，不是本轮 448×448 图片的编码成本；后续引用必须多轮取统计并说明口径。
- **证据边界（2026-09-06 更新）**：4B 文本共置正式矩阵已冻结。`llm_tts_colocated_4b_text_only_20260904_direct_nvml_v1`（commit `dc3974bf`，dirty=false）6 候选×5 轮 30/30、0 failure/fallback、swap 增长 0、窗口内新鲜度 218–324 ms、30 个 WAV 大小/SHA-256 全通过；离线复算 `c3_h_4b_tts_colocated_20260906_v1.json` 正残差 max=0，G_safe=501.282 MiB 校准后维持预设值并冻结为 4B 文本口径，H_TTS 在 512/2048/4096×T0/T3 全部 `frozen_real_colocated`。同日 4B VISION_WARM 多轮画像 `v1_vision_endpoint_4b_warm_profile_20260906_direct_nvml_v1`（FULL×VISION_WARM_GPU×5 轮）5/5 通过，projector warm GPU 增量 mean 761.2 / min 710 / max 838 MiB（含 warmup 预留 223.30 MiB，非 448×448 编码成本）、release −888 MiB、RSS +约 55.8 MB；不冻结任何 H。
- **证据边界（2026-09-06 图像侧收案）**：4B `H_D` 按 16G 论文环境以负面边界证据收案。Forge 同日复测 `forge_b1_reprofile_20260906_direct_nvml_v1`（`M_D=5273.21 MiB`、idle 高水位 655.61）注入 `c3_h_4b_tts_direct_nvml_20260906_v1.json` 后，FULL 因 GPU 容量不可行，H_D 候选为 512/K16、2048/K14、4096/K14；正式矩阵 `llm_forge_colocated_4b_20260906_direct_nvml_v1`（commit `a39778f`）首个 K16@C512 唯一失败门为 swap 增长 405.15 MiB，失败即停（1/15）。归因诊断 `llm_forge_colocated_4b_20260906_mem20g_diag_v1` 在 `.wslconfig memory=20GB` 下同代码、同候选、同参数 3/3 通过（swap 0.0–0.2 MiB、GPU 峰值 8122/8151 MiB），证明失败根因为 16G 主机内存容量而非 K 档固有换页或 WSL2 特有行为；该诊断不进论文正文（论文环境为 16G），`.wslconfig` 须恢复默认。9B（K6/K4）与 4B（K16）的同模式失败构成跨尺度负面证据：8GB GPU + 16GB 主机下 LLM+Diffusion 无任何可共置档位，图像期应采用迁移/逐出而非共驻留。

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
| 弹性驻留 B2 | `scripts/experiments/run_tts_backend_compare.ps1 -Mode b2` → `experiments/tts/profile_qwen3_tts_cudagraph.py` + `b2_profile_contract.py` + `tts_phase_memory.py`；核对 `check_tts_output_correctness.py` | B2 现指当前接入后端（加速 CUDA Graph 整模型 GPU，放置 T0）的句长画像：20/50/100 字 × 5 轮 + 状态残留序列 + 主环境 ASR 回转正确性；记录 RTF、波形就绪、加载与图准备成本、准备/执行阶段内存峰值、输出长度。旧后端 T0–T4 三批产物已于 2026-09-17 删除，其数值不再作为依据 |
| Forge b=1 校准 | WSL2 `forge_b1_reprofile.py` + `forge_runtime_evidence.py` | 真实 Forge HTTP b=1、direct NVML 周期采样、idle/resident/每轮 measurement 逐窗 750 ms 样本年龄硬门、swap 增长与核心归还；2026-08-31 正式批次 5/5 通过，`M_D=5484 MiB` |
| 弹性驻留 C1 | WSL2 `llm_residency_curve.py` + `c1_llm_residency_512_20260829_verified_v1` | K0/FULL 真实端点、精确 token/gap、阶段资源与 KV 保存恢复；不代替 C2 多轮曲线 |
| KV 语义/容量 | `tests/elastic_residency/test_kv_measurement.py` + WSL2 `llm_kv_cache_calibration.py` | 字段语义、原生 position/`llm.n_tokens`、日志解析与缺字段失败；`c3_kv_cache_calibration_20260830_direct_nvml_v1` 已完成 48/48 条正式容量矩阵，不作为性能计时证据 |
| GPU 计算份额 SM-A | `tests/elastic_residency/test_sm_profiles.py`、`test_sm_capability.py`、`test_sm_probe_records.py`、`test_mps_commands.py`、`test_mps_runtime.py`、`test_mps_daemon*.py`、`test_cuda_exec_affinity.py`、`test_green_context_probe.py`、`test_sm_wsl_gates.py` + WSL2 `sm_capability.py` | Windows 静态测试与 WSL 门控均已执行。正式记录 `sm_a_20260830_direct_nvml_v1` 基于干净 `ca7a7ff`，真实 CUDA 基线、36 SM、direct NVML、0 failure/fallback/cleanup failure，但 `nvidia-cuda-mps-control` 不存在，WSL2 路径暂定 NO_GO；暂停 SM-B/SM-C，等待原生 Linux 重验，不修改生产调度器 |
| LLM+TTS 共置 | `test_llm_kv_colocated.py`、`test_analyze_colocated_results.py` + WSL2 `llm_tts_colocated.py` | `B_idle/B_LLM/P_colocated`、正残差和扣除 idle 后的容量裁决；9B 正式 30/30 已通过并冻结该模型的 `G_safe/H_TTS`，未晋升正文；4B+视觉须重跑并重新冻结 |
| LLM+Forge 共置 | `test_llm_forge_plan.py`、`test_analyze_forge_colocated_results.py` + WSL2 `llm_forge_colocated.py` | 9B 的 K6/K4 smoke 分别因约 290.44/2070.32 MiB swap 增长失败，正式矩阵停止；入口与冻结分析待扩展到4B+真实 `mmproj` 后重新筛选 |
| 多模态端点门 V1 | `test_vision_artifacts.py`、`test_vision_mtmd_runtime.py`、`test_vision_evidence.py`、`test_vision_endpoint_phases.py` + WSL2 `vision/vision_endpoint_smoke.py` | 静态层证明 GGUF 解析、artifact 对应关系、固定图片内容哈希、libmtmd 调用契约与失败路径、状态一致性门禁与聚合口径；真实层需要 WSL2 GPU 运行才能证明 projector 真实执行、视觉 token 数、输出文本、GPU/RSS 增量与 swap。假后端通过**不**代表多模态可用 |
| 恢复/持久化 | `tests/verify_state_backplane_recovery.py` | mmap 读写和恢复闭合 |
| C++ 绑定 | `cpp_scheduler/verify_bindings.py`, `tests/test_cpp_scheduler_minimal.py` | 扩展导入与 API 契约 |
| 真实 LLM | `tests/test_llama_gpu.py`, migration experiments | 当前模型/驱动下的推理与迁移 |
| Forge 端到端 | `run_forge_experiments.py` | 图像服务、显存竞争和真实 workload |
| 论文复现 | 多次迭代 + manifest + 图表重建 | 可提交的统计证据 |

## 10. 已知维护风险

- 父仓库 `.gitignore` 含 `/legacy/`，当前目录全部被父仓库 Git 忽略；需要版本管理时必须先决定独立仓库或调整父仓库策略，不能误以为文档已被提交。
- `cpp_scheduler/build/` 约数百 MB，包含第三方源码和构建产物；搜索、统计和评审时必须排除。
- `data/` 同时含源码适配器和本地 mmap/日志数据，修改时要精确到文件，不能整目录清理。
- 重写前的旧需求说明已移至 `docs/archive/pre_rewrite_20260830/EDGE_SCHED_REQUIREMENTS_OLD.md`，不能代替当前论文总指南和执行计划。
- `docs/` 下三份 NVRAM/C++ 评估文档包含历史设想和过强结论；当前任务不得直接引用为现状。
- README 曾长期漂移。新增或移动入口后，应先更新本上下文与验证脚本，再更新目录 README。

## 11. 文档维护约定

- 模块职责、入口或目录变化：更新 `PROJECT_INDEX.md` 和本文件第 3-4 节。
- 论文实验路线变化：更新 `paper/thesis_experiment_guide/` 对应主题模块；短期状态只更新第 10 模块与结果目录，兼容入口不复制正文。
- 数据流、状态机或迁移协议变化：更新第 5 节和对应实验。
- 实验口径、正文数据目录或晋升流程变化：更新第 6 节、`experiments/README.md`、`paper/experiment_data/README.md`。
- 环境、模型或同步方法变化：更新第 7-8 节和局部 `AGENTS.md`。
- 每次文档更新运行 `tests/scripts/verify_project_context.py`，并对变更路径做最小可用测试。


## 辅助任务 CPU 核数画像入口

辅助任务 CPU 核数画像入口已实现：`experiments/elastic_residency/auxiliary_core_budget.py`，复用物理核租约与采样器，支持 JSON RAG 1/2 核、BGE/Whisper 1/2/4 核和 `R_sys=0/1/2/4` 隔离敏感性。模块职责、运行方法和 D3 正式并发验收边界见 [辅助核数画像](AUXILIARY_CORE_BUDGET.md)。


BGE 画像已按职责拆出 `experiments/elastic_residency/auxiliary_bge.py`：保留 `bge` 全量重编码，新增 `bge_cached` 候选预计算模式，准备成本与请求耗时分别记录，任务名分组避免混用。`scripts/experiments/run_auxiliary_cached.ps1` 供用户从任意目录手动启动 15 次 BGE 补测；具体协议与已完成的 40/40 结果见 [辅助核数画像](AUXILIARY_CORE_BUDGET.md)。真实实验由用户执行。
