# EdgeSched 弹性驻留可行性实验执行计划（2026-08-28）

## 1. 本轮目标与边界

本计划只覆盖进入弹性驻留开发前必须完成的四组真实可行性实验：

1. 辅助任务物理核预算与 `R_sys` 正式验证；
2. Qwen3-TTS T0–T3 组件级部分卸载；
3. Qwen3.5-9B LLM 驻留曲线与 `H_TTS` / `H_D` 临界档位；
4. 无图像时 LLM→TTS 的真实重叠放置。

在这些实验完成并出现真实 Pareto 改进前，不修改 Phase-Aware 核心调度器、不扩展 C++ 驻留状态，也不把结果晋升到论文正文数据。

所有正式运行均在 WSL2 `/home/leslie/mvp_core` 和 `/home/leslie/.venvs/mvp_core-cu128` 中进行。结果先写入 `experiment_results/elastic_residency/`，只有人工复核后才考虑进入 `paper/experiment_data/`。

## 2. 已核验的当前条件

### 2.1 硬件与运行环境

| 项目 | 当前核验值 | 计划中的用法 |
|---|---:|---|
| GPU | RTX 5070 Laptop，8151 MiB | 所有显存约束使用卡总量、进程总量和 PyTorch allocated/reserved 三套读数 |
| WSL2 内存 | 约 15 GiB，另有 4 GiB swap | 记录 RSS、系统可用内存和 swap；发生 swap 的轮次判为不可用 |
| CPU | 16 个物理核、32 个逻辑 CPU | 物理核代表为 `0,2,4,...,30`，不能把 SMT 线程当作额外物理核 |
| LLM | Qwen3.5-9B-Q4_K_M，GGUF 约 5.23 GiB | GGUF 元数据 `qwen35.block_count=32` |
| TTS | Qwen3-TTS-12Hz-0.6B-Base | 真实 worker 已存在；当前为句子级非增量音频生成 |

因此 LLM 首轮粗测点固定为：

```text
N = 32
k = 0, 8, 16, 24, 32, FULL
```

`FULL` 使用 llama.cpp 的完整 offload 语义，必须与 `k=32` 分开测，因为 embedding/output 等非 block 张量可能使两者显存不同。

### 2.2 辅助模型

模型已复制到 WSL2，关键权重哈希与 Windows 原文件一致：

| 任务 | WSL2 路径 | 格式与规模 | 当前状态 |
|---|---|---|---|
| 轻量 ASR | `/home/leslie/mvp_core/models/auxiliary/faster-whisper-small` | Whisper Small，CTranslate2 FP16，约 486 MB | 权重与固定依赖可用；已完成 CPU int8 单核真实接口 smoke |
| BERT embedding | `/home/leslie/mvp_core/models/auxiliary/bge-small-zh-v1.5` | 4 层 BERT，512 维，ONNX 权重约 96 MB | 可由现有 ONNX Runtime 直接加载 |

BGE ONNX 已完成一次单线程可行性 smoke：3 条文本输出形状为 `3×512`，模型加载约 0.23 s，单批推理约 0.025 s。该数字只证明接口可用，不是正式性能结果。

Faster-Whisper Small 已在 `mvp_core-cu128` 使用 1 个物理核、CPU int8、beam size 1 对 3.42 s 中文参考音频完成一次真实接口 smoke：加载约 1.71 s，转写约 4.92 s，RTF 约 1.44。该单轮结果只证明模型与依赖链可用，并提示 1 核大概率不能满足正式 `RTF p95 <= 0.8` 门槛；仍须按 1/2/4 核和冻结音频集正式测量。

BGE 是 bi-encoder，不是 cross-encoder reranker。计划中的“rerank”定义为：先对候选文档生成/读取 BGE 向量，再按 query cosine 重排；报告中命名为 `dense_embedding_rerank`，不得写成 cross-encoder rerank。

### 2.3 TTS 组件规模与现有性能

从 safetensors 元数据得到：

| 组件 | 权重规模（约） | 说明 |
|---|---:|---|
| Talker 主体与 code predictor | 1.73 GiB | TTS 生成主体，T0–T3 都保留在 GPU |
| Speech tokenizer / codec | 650 MiB | T1–T3 的主要可卸载对象 |
| Speaker encoder | 17 MiB | T2/T3 额外释放量很小 |

现有 GPU 句长画像中，短/中/长文本的进程内 peak allocated 约为 2.25/2.41/2.76 GiB，45/45 条结果的 RTF 都大于 1。由此得到两个前置判断：

- codec 卸载理论上可能腾出约 3–4 个“按 GGUF 文件均分”的 LLM block 空间，但最终必须使用实测 `M_L(k+1)-M_L(k)` 裁决，不能用均分估算作结论；
- 当前 TTS 后端无法提供持续实时音频。后续若仍然 `RTF_p95 > 1`，无论是否通过有限回复的预缓冲避免 underrun，都不得声称稳态无断流。

### 2.4 兼容修复与剩余阻塞项

1. 音频环境兼容已完成：安装 `libasound2-plugins/pulseaudio-utils/alsa-utils` 后，WSL2 PortAudio 能通过 WSLg Pulse 枚举真实输出；同时 `audio_playback_probe.py` 提供 Windows WASAPI 回退和 24 kHz→设备采样率显式转换。1 s PCM16/24 kHz 单声道静音在原生 Pulse 与强制 Windows WASAPI 两条路径均为 0 underflow。该 smoke 只证明播放链路可用，正式“无 underrun”仍须使用真实 TTS chunk 和并发负载验证。
2. Python 依赖冲突已修复：`blendmodes==2025` 与 `numpy==2.2.6` 兼容，`pillow==10.4.0` 同时满足 blendmodes 与现有 `gradio==4.40.0`；WSL2 `pip check` 为 `No broken requirements found`，Qwen-TTS、Gradio、Torch 均可导入。
3. 现有 TTS 高层接口把 talker 生成和 codec decode 合在一次调用中，尚不能分别记录组件传输与 decode 时间；这是 B1 的当前工作项。
4. 现有 LLM adapter 不输出准确 prefill/decode/token-gap/KV 指标，也不支持任意 `n_gpu_layers` 的可复现实验编排。
5. 当前 Phase-Aware 中的 TTS/LLM 静态显存值是占位值，不得用于本轮决策。

## 3. 冻结的实验口径

### 3.1 通用运行规则

- 正式配置至少 5 次独立测量；每个配置独立预热，预热不计入统计。
- 运行顺序采用正序/反序交替，记录温度、功耗和时钟，降低热漂移影响。
- 固定模型哈希、代码状态、随机种子、CPU affinity、WSL 内存限制、驱动和后端类型。
- 每次运行保存原始 JSONL；汇总 JSON/CSV 必须能由原始记录重新生成。
- 任何 fallback、OOM、swap、音频设备缺失或 affinity 未生效都写成结构化失败，不静默跳过。
- p95 性能结论至少需要 30 个有效 chunk/event；只有 5 个独立轮次的字段报告全部原值、mean/max，不把插值 p95 当强证据。

### 3.2 资源采样

统一采样器以 50 ms 周期记录：

- direct NVML GPU card used/total/utilization/power/temperature；
- direct NVML 目标进程 GPU memory（WSL/WDDM 不可用时记录 `NVML_VALUE_NOT_AVAILABLE`，不得伪造）；
- PyTorch allocated/reserved/max allocated（适用时）；
- 进程 RSS、系统 available memory、swap；
- 进程 CPU、每个逻辑 CPU 利用率、实际 affinity；
- 事件循环/调度探针的 wake-up jitter；
- 音频队列深度、可播放时长和 underrun callback（真实设备可用后）。

资源曲线保存为 `timeline.jsonl`，业务事件使用同一 monotonic clock 写入 `events.jsonl`，不能只保留一个峰值。

### 3.3 数据驱动的显存安全余量

不预设固定 `G_safe`。每类配置先从重复运行中计算：

```text
G_safe = max(
  idle GPU baseline 的高水位,
  reserved - allocated 的最大 allocator slack,
  单模型峰值相加与真实共置峰值之间的最大正残差,
  重复运行的峰值漂移范围
)
```

`H_TTS` 和 `H_D` 只用于筛选候选；最终可行性必须由目标组合真实同时加载并运行证明，不能只依赖显存加法。

### 3.4 默认延迟门槛（执行前冻结）

原需求没有给辅助任务的具体 SLA。建议先使用以下可解释门槛；若后续更改，必须新建 run id，不能覆盖旧结果：

| 任务 | 建议门槛 | 工作负载 |
|---|---|---|
| JSON RAG | p95 ≤ 50 ms | 1 个 query，在冻结的 1000 条 JSON 文档中检索 top-32 |
| BGE embedding/rerank | p95 ≤ 150 ms | 1 个 query + 32 个候选，max length 128，输出 top-8 |
| Faster-Whisper Small | p95 RTF ≤ 0.8 | 固定 5–10 s 中文音频，CPU int8，beam size 1 |

正式 `R_sys` 验证还要求：主流水线 decode tokens/s、TTFA 和完整回复时间相对同放置无辅助任务基线的损失都不超过 5%，并且调度 jitter 不出现新的明显尖峰。明显尖峰同时报告绝对 p99/max 和相对基线倍数，不只用主观判断。

## 4. 实现与执行顺序

### 阶段 A：实验基础设施

新增独立目录，不先接入生产调度器：

```text
experiments/elastic_residency/
  contracts.py                 统一配置、事件和结果契约
  cpu_topology.py              物理核/SMT 分组与互斥 affinity 分配
  resource_sampler.py          GPU/DDR/CPU/jitter 时间线
  auxiliary_core_budget.py     JSON RAG、BGE、ASR 核心曲线
  tts_component_offload.py     T0–T3
  llm_residency_curve.py       粗测、临界补测和 KV/切换
  kv_measurement.py            KV buffer/序列状态/完整状态语义与采集
  llm_kv_cache_calibration.py  16 档独立 KV 容量校准
  llm_tts_colocated.py         LLM+TTS 真实共置峰值与残差
  no_image_placement.py        真正流式 LLM→TTS 重叠
  analyze_results.py           Pareto、阈值和汇总生成
tests/elastic_residency/
```

验收：Windows 静态/Mock 测试通过；WSL2 preflight 能确认真实模型、真实后端、affinity、采样器和时钟一致性。任何真实脚本默认禁止 fallback。

**A1 历史记录（2026-08-28）**：`a1_smoke_20260828_verified_v6` 在 WSL2 真实 OS/NVIDIA 探针下采到了正常原始数据，但旧版 `passed` 没有强制 GPU 成功和 jitter 门槛，dirty 源码也只有状态项数量，因此仅保留为基础设施调试记录，不再作为当前 A1 门禁证据。该批次现位于 `_archive/superseded/a1/`。

**A1 当前硬门槛（2026-08-29）**：`passed` 必须同时满足首个成功 GPU 样本先于资源时间线、时间线无 GPU unavailable 点、GPU 数据最大年龄不超过 500 ms、资源采样 jitter p95/max 不超过 5/10 ms、事件循环 jitter p95/max 不超过 5/10 ms、无 swap、工作/采样/GPU 线程 affinity 与租约一致、全部物理核归还，并且源码 provenance 完整。首个 CPU 利用率样本保留在原始时间线但标记 `cpu_sample_valid=false`，所有聚合明确排除它。正式运行优先使用 `dirty=false + status_entry_count=0` 的干净 commit；若必须使用 dirty 源码，则必须用 `--source-git-snapshot` 将可还原快照复制进运行目录并校验 SHA-256。门槛更新后的 WSL2 重跑结果必须使用新的不可覆盖 run id。

**A1 最终记录（2026-08-29）**：`a1_smoke_20260829_verified_v7` 基于已推送的干净 commit `ae686083970d68c2b27c62810b144a566cdbfc22` 在 WSL2 通过全部 17 项硬门禁。工作 affinity 为 `[0,2]`，资源与 GPU 采样线程为 `[28,29,30,31]`；41 个时间线点全部含可用 GPU 数据，最大数据年龄 422.719 ms；资源 jitter p95/max 为 0.309/1.680 ms，事件循环 jitter p95/max 为 2.165/3.081 ms。首个 CPU 点标记无效，聚合使用后续 40 点，进程 CPU mean/max 为 5.502%/20.2%；swap 为 0，结束后 16 个物理核全部归还且无残留租约。Windows/WSL2 均为 19 项弹性驻留测试通过，Ruff 通过。该结果只通过实验基础设施门，不是 TTS、LLM、并发音频或论文性能证据。

### 阶段 B：TTS T0–T3 组件卸载

先做模块发现 smoke，确认实际模块设备与张量传输，再跑正式矩阵。这里“Tokenizer/Codec”解释为 speech tokenizer/codec；文本 tokenizer 本身没有 GPU 推理模块，始终在 CPU。

| 配置 | Talker | Speech codec | Speaker encoder / prompt |
|---|---|---|---|
| T0 | GPU | GPU | GPU 构建并缓存 |
| T1 | GPU | CPU | GPU 构建并缓存 |
| T2 | GPU | CPU | CPU 构建并缓存 |
| T3 | GPU | CPU | GPU 构建一次并缓存 prompt，随后释放 speaker encoder |

实现使用实验侧 wrapper/hook，不修改已安装的 `qwen_tts` 包。需要将高层调用拆成：文本 tokenize、talker generate、code tensor transfer、codec decode、序列化/写盘，并逐段计时。

#### B1 开跑前的强制边界

1. 每个 T0–T3 配置使用独立子进程，防止上一配置的 CUDA allocator、模型对象或缓存污染下一配置；子进程必须回报 talker、speech codec、speaker encoder、文本 tokenizer 的实际参数/缓冲区设备，不能只回报目标配置。
2. `passed` 必须要求计时前已有成功 direct NVML 探针；短时显存峰值以每个测量阶段前重置的 PyTorch `max_memory_allocated/max_memory_reserved` 为主，direct NVML 整卡显存作为交叉验证。
3. CPU 利用率探针初始化值必须丢弃并在 manifest 中记录；GPU 首样本等待不计入 talker、传输或 codec 时间。
4. 计时严格拆为 text tokenize、talker generate、code tensor transfer、codec decode；同步点放在每段边界，T1–T3 必须实际观察到 code 从 GPU 到 CPU 的设备变化。
5. T3 的语义是 prompt 构建后不再保留 speaker encoder，而不是仅把名字标成“已卸载”；cached prompt tensor 的实际设备和 encoder 不存在状态都要写入 inventory。
6. B1 正式 smoke 只允许干净 commit，或附带可还原源码快照及 SHA-256 的 dirty 状态。模型目录和参考音频分别记录树哈希/文件哈希。
7. 单文本 B1 只证明 T0–T3 可加载、设备映射正确和分段指标可采集；不用于 Pareto、延迟或论文结论。只有它通过后才进入 B2 的 `4 × 3 × 5` 正式矩阵。

**B1 完成记录（2026-08-29）**：`b1_tts_components_20260829_verified_v2` 基于干净 commit `38f4d81dca1ff4020c4b245426c49fc124341d65` 在 WSL2 真实 Qwen3-TTS/CUDA 后端通过，T0–T3 为 4/4、每配置 13 项门禁均通过，失败文件为空。talker 始终为 `cuda:0`；T0 codec/speaker 为 GPU，T1 codec 为 CPU 且 speaker 为 GPU，T2 codec/speaker 均为 CPU，T3 codec 为 CPU 且 speaker encoder 在 prompt 缓存后实际移除。T1–T3 的 audio codes 均观测到 `cuda:0→cpu`，四份 WAV 均为有限非静音波形。WSL 驱动对 compute process 显存返回 `[N/A]`，产物没有伪造为 0，而是记录 unavailable 原因、整卡 ready 值、相对 314 MiB 空闲基线的独占增量估计和每阶段 PyTorch allocated/reserved 峰值。

单次冷态 smoke 中，T0/T1/T2/T3 的 ready 整卡增量估计分别为 2362/1914/1902/1906 MiB，TTFA 分别为 5.939/7.075/6.496/7.043 s，RTF 分别为 1.954/2.327/2.538/2.317。该数字只用于检查量纲与发现明显错误：每个配置只有一轮、没有独立预热，T2 的 CPU speaker 还改变了输出时长，因此不能据此判断 Pareto、质量等价性或“节省显存值得卸载”。这些裁决留给 B2 的冻结文本矩阵和多轮统计。

**B2 完成记录（2026-08-29）**：`b2_tts_matrix_20260829_verified_v1` 基于干净 commit `ce024a051e2e878600eb53611b1ea1ee72ddf7fa` 完成 60/60 条真实迭代，12 个配置/句长分组各 5 条，0 fallback、0 failure。Pareto 候选为 T0/T3；T1/T2 被 T3 在最大整卡增量、TTFA p95 和 RTF p95 上严格支配。T3 相对 T0 的最大整卡增量少 1304 MiB，但 TTFA p95 增加 26.1%，RTF p95 增加 9.4%，最终是否保留要由 C 阶段验证这部分显存能否增加有效 LLM 层。全部 60 条的 RTF 均大于 1，当前后端无法支持持续实时播放，因此组件放置不能直接满足无 underrun 目标。完整分析见 `docs/TTS_COMPONENT_OFFLOAD_MATRIX_20260829.md`。

正式矩阵：4 配置 × 20/50/100 字 × 5 轮，另加每配置独立预热。记录 TTFA（当前语义为完整句 WAV ready）、合成时间、RTF、allocated/reserved/进程总显存、主机内存、CPU 利用率和组件传输时间。

裁决顺序：

1. 删除任何被另一配置同时在显存、TTFA、合成时间和 RTF 上支配的配置；
2. 检查节省显存是否至少覆盖一个实测相邻 LLM 驻留档位的增量和安全余量；
3. 只有能增加 LLM 层数或形成实际共置配置的 TTS 方案进入阶段 E；
4. 若 T1–T3 都不能增加可驻留层数，则停止 TTS 部分卸载开发，保留 T0 和全 CPU 回退。

### 阶段 C：LLM 驻留曲线

固定 `c_L=14` 个物理核，粗测：

```text
k = 0, 8, 16, 24, 32, FULL
context = 512, 2048, 4096 tokens
generation = 128 tokens
runs = 5
```

每次请求使用精确 token 数构造 prompt，清空 KV 后运行。记录：

- 进程/整卡显存、RSS/available/swap；
- TTFT；
- llama.cpp prompt-eval tokens/s 和 decode tokens/s；
- 每个输出 token 时间戳以及 gap p50/p95/max；
- 模型加载时间；
- KV save、KV restore、state bytes；
- `FULL→k` 和 `k→FULL` 的目标实例创建、状态恢复和原子替换时间。

使用阶段 B 的 Pareto TTS 峰值和同日 Forge batch=1 峰值计算：

```text
H_TTS = max k: M_L(k) + M_T + G_safe <= V_GPU
H_D   = max k: M_L(k) + M_D + G_safe <= V_GPU
```

已有 Forge batch=1 结果的最大整卡占用约 5904 MiB，只能作为预估；阶段 C 必须同日复测 Forge，避免把旧驱动/旧进程基线混入 `H_D`。

在两个临界点分别补测 `k±2`、`k±4`，裁剪到 `[0,32]`、去重，并再次真实共置验证。若 `H_TTS` 或 `H_D` 在不同上下文下变化，保留按 context 分组的档位，不强行合并成一个常数。

**C1 完成记录（2026-08-29）**：`c1_llm_residency_512_20260829_verified_v1` 基于干净 commit `234d2df686998fbb82f51068f6b55ee886cf0205`，使用实际 32 层 Qwen3.5-9B GGUF 完成 K0/FULL、512-token、固定生成 128-token 的真实端点 smoke。两条记录均通过 19/19 项门禁，0 failure、0 fallback，KV state 均保存并恢复到 640 tokens，核心和租约完全归还。K0/FULL 的 TTFT 为 10.659/0.326 s，prefill 为 48.05/1593.58 tok/s，decode 为 2.77/51.03 tok/s，GPU 占用增量估计为 127/5543 MiB，推理后 RSS 为 6.253/1.139 GiB。WSL 进程显存精确值仍不可用，增量只作为估计。KV save/restore 存在独立短时 jitter 尖峰，已经分阶段保留，不能声称系统全程无尖峰。完整分析见 `docs/LLM_RESIDENCY_C1_20260829.md`。该单轮端点 smoke 不用于正式曲线、`H_TTS/H_D` 或论文性能结论。

#### C2 正式运行边界

1. 一个 run id 冻结 `6 配置 × 3 上下文 × 5 轮=90` 条记录；每个上下文内按轮次轮转配置顺序，避免始终由同一配置承担冷热顺序偏差。
2. 每个子轮完成后立即追加 iteration、timeline 和 event。意外中断只允许在运行配置、源码 commit、Python/平台和模型 manifest 完全相同时用 `--resume` 跳过已完成键；矩阵外或重复键直接拒绝。
3. 子进程超时、返回失败或测量有效性门失败默认 fail-fast；最终 `summary.passed=false` 时入口返回非零。`failures.jsonl` 非空的 run 永远不能成为通过记录，修复后必须换新 run id，不能靠续跑洗掉失败历史。
4. C1 的 jitter 是基础设施硬门；C2 中同一阈值是被测性能目标，正式矩阵使用 `--jitter-gate-mode observe`。超标轮次必须保留为 `performance_target_passed=false`，进入系统尖峰/Go-No-Go 结论，但不能阻止其余驻留档位的数据收集。GPU 探针可用性、新鲜度、affinity、token/KV 完整性、swap 和核心归还仍是硬门。KV save/restore 尖峰按阶段单列，不混入推理目标，但必须进入切换成本讨论。
5. `by_context_config` 才是正式统计分组；跨上下文的 `by_config` 只用于总览，不用于比较 TTFT/prefill。

**C2 v1 停止记录（2026-08-29）**：`c2_llm_residency_curve_20260829_verified_v1` 在第 22/90 条（512-token、K0、第 4 轮）按旧语义 fail-fast。该轮 18 项其他检查通过，jitter p95 为 0.866 ms，但出现一次 39.490 ms 最大迟到，超过 30 ms 目标。v1 证明性能目标确有尾部失败，也暴露了“性能不达标会阻止收集 No-Go 证据”的口径错误；它只作为停止/调试记录，不续跑、不作为完整 C2 曲线。后续新 run id 保留 30 ms 目标但使用 `observe`，同时报告有效性 `passed` 与性能目标 `performance_target_passed`。该批次现位于 `_archive/stopped/c2/`。

**C2 v2 停止记录（2026-08-29）**：`c2_llm_residency_curve_20260829_verified_v2` 完成 512-token 全部 30 条后，在第 31/90 条（2048-token、K0、首轮）因 GPU 数据最大年龄 856.023 ms 超过 750 ms 有效性门而停止。探针始终可用，根因是旧 GPU sampler 在一次 `nvidia-smi` 探针耗时 624.415 ms 后又固定等待 250 ms，使实际采样间隔叠加到 874.730 ms。修复将 GPU sampler 改为固定 deadline：慢探针只跳过已经错过的 tick，不再在完成后额外等待整个周期；750 ms 门槛不放宽。v2 只作为停止/调试记录，不续跑、不作为完整 C2 曲线。该批次现位于 `_archive/stopped/c2/`。

**C2 direct NVML v1 停止记录（2026-08-29）**：`c2_llm_residency_curve_20260829_direct_nvml_v1` 在第 64/90 条（4096-token、K24、首轮）遇到一次 512.231 ms 的 direct NVML 长尾；旧固定网格推进又等待到下一个 250 ms tick，把有效样本间隔放大到 756.209 ms，推理阶段最大样本年龄 751.457 ms，因而按 750 ms 硬门正确停止。主资源线程同阶段最大 jitter 仅 0.775 ms，排除 CPU sampler 饥饿；运行时后端为 `direct_nvml` 且 `gpu_cli_subprocess_used=false`。修复改为越过下一 tick 后立即补采，并把显存查询放到辅助 NVML 查询之后以保证时间戳贴近显存读数。该 64 条运行冻结为停止/调试证据，不得续跑或与修复后曲线合并，现位于 `_archive/stopped/c2/`。

**C2 旧 CLI 完成记录（2026-08-29）**：`c2_llm_residency_curve_20260829_verified_v3` 基于干净 commit `63e75baecd8895fc7fa1c6fc74cb351cb9c1db02` 完成 `6 配置 × 3 上下文 × 5 轮 = 90/90` 条真实 llama.cpp/Qwen3.5-9B 迭代，`failures.jsonl` 为空，0 fallback；每条记录的 17 项有效性门全部通过，`performance_target_passed` 为 90/90（30 ms jitter 目标在 observe 模式下全部满足）。所有 `gpu_delta_mb` 为相对空闲整卡的增量估计（WSL 驱动对 compute process 显存返回 N/A）。该批次已被 direct NVML v2 替代，现位于 `_archive/superseded/c2/`。

整卡增量估计（MiB）：

| k | 512 | 2048 | 4096 |
|---|---:|---:|---:|
| K0 | 127 | 127 | 127 |
| K8 | 2365 | 2417 | 2431 |
| K16 | 3373 | 3425 | 3455 |
| K24 | 4383 | 4439 | 4485 |
| K32 | 5407 | 5455 | 5519 |
| FULL | 5543 | 5593 | 5659 |

性能总览（`by_config` 跨上下文 mean；TTFT 按 512/2048/4096 分列）：

| k | decode tok/s | prefill tok/s | token gap p95 (ms) | TTFT 512/2048/4096 (s) | RSS (MiB) |
|---|---:|---:|---:|---|---:|
| K0 | 3.54 | 47.1 | 335 | 14.0 / 42.4 / 74.8 | 6439 |
| K8 | 4.49 | 561.0 | 263 | 2.2 / 3.5 / 6.4 | 4345 |
| K16 | 6.30 | 783.3 | 188 | 0.8 / 2.6 / 4.8 | 3328 |
| K24 | 10.52 | 1180.7 | 114 | 0.5 / 1.8 / 3.3 | 2324 |
| K32 | 36.24 | 2322.1 | 32 | 0.3 / 0.9 / 1.8 | 1334 |
| FULL | 55.52 | 2690.8 | 20 | 0.25 / 0.81 / 1.58 | 1182 |

关键观察：

1. `K32` 与 `FULL` 的整卡增量只差约 140 MiB，但 decode 从 36.2 升至 55.5 tok/s（约 +53%）；非 block 张量（embedding/output 等）留在 CPU 是 k=32 的主要瓶颈，证实了"FULL 必须与 k=32 分开测"的预设。
2. decode 吞吐在 K24→K32 之间出现跳变（10.5 → 36.2 tok/s），而 K0→K24 段平缓（3.5 → 10.5）；驻留层数的边际收益在 8151 MiB 卡上高度非线性，中间档位（K8–K24）的交互价值有限。
3. KV save/restore mean 约 0.66/0.99 s，各配置基本一致；配置切换成本主体在实例创建与状态恢复。
4. K0 的 TTFT 随上下文从 14 s 增至 74.8 s，全 CPU 降级在 2048+ 上下文不可交互。

C 阶段收尾前仍必须完成：Forge b=1 同日复测、按 3.3 节校准 `G_safe`、临界补测 `k±2`/`k±4`，以及与 B2 Pareto TTS 峰值的真实共置验证。基于旧 Forge 峰值 5904 MiB 与 B2 数字的初步筛选（只用于排程，不是结论）：T0（3284 MiB）下候选约 K24，T3（1980 MiB）下 K32/FULL 可能可行，`H_D` 落在 K0–K8 之间；这些筛选值在 Forge 复测与 `G_safe` 校准后必须全部重算。

**C 阶段收尾工具（2026-08-29）**：新增 `experiments/elastic_residency/forge_b1_reprofile.py` 与 `experiments/elastic_residency/analyze_results.py`，配套 13 项纯逻辑测试（`tests/elastic_residency/test_forge_b1_reprofile.py`、`test_analyze_results.py`，Windows 侧 pytest 与 Ruff 均通过）。

- `forge_b1_reprofile.py`：Forge b=1 同日复测。通过 Forge HTTP API（服务需先在 `forge-cu128` 启动）执行 warmup + 5 轮 b=1 txt2img（口径与 `batch_scaling_experiment.py` 一致），产出 `global_idle_high_water_mb`、`forge_resident_baseline_mb`（冷却窗口口径）、逐轮峰值与 `m_d_incremental_estimate_mb = peak_max - idle_high_water`，写入统一证据契约产物目录。`forge_runtime_evidence.py` 对参与上述差值的 idle/resident/每轮 measurement 窗口逐窗执行 750 ms GPU 样本年龄硬门；超限或无有效年龄样本会写 failure 并使整批失败。
- `analyze_results.py`：按 3.3 节公式计算 `G_safe`（idle 高水位、`reserved-allocated` allocator slack、重复运行漂移、colocated 正残差四个分量，后者在真实共置验证前保持 pending）与逐上下文 `H_TTS`/`H_D`（mean 基准推荐 + max 基准保守双口径），并生成 `k±2`/`k±4` 临界补测建议。

**C3 初步筛选（2026-08-29，legacy Forge 口径）**：`experiment_results/elastic_residency/c3_analysis/c3_h_screen_20260829_legacy_forge.json` 使用 C2 v3 + B2 v1 + 旧 Forge 峰值 5904 MiB（idle 314 MiB）计算：`G_safe = 501.3 MiB`（由 B2 T0 的 allocator slack 主导，colocated 残差 pending）；三个上下文结论一致：`H_TTS(T0) = 16`（K24 差约 17 MiB 不可行）、`H_TTS(T3) = 32` 且 FULL 可行、`H_D = 0`（K0–K8 临界）。该筛选值不得用于冻结，Forge 复测后必须重算；T0 下 K24 的临界贴边也说明补测 `k={20,22}` 有实际价值。

**Forge 复测 v1 停止记录（2026-08-29）**：`forge_b1_reprofile_20260829_verified_v1` 采集数据全部正常（idle 高水位 185 MiB、Forge 常驻基线 409 MiB、5 轮峰值 max/mean 5669/5668.6 MiB、`M_D` 增量估计 5484 MiB），但 `swap_unused` 门禁失败：时间线中 WSL2/Forge 带来 780 KB 级系统性 swap 噪声，而 v1 沿用了 A1/C2 的 `swap==0` 口径。A1/C2 当时无 Forge 服务常驻，该口径对 Forge 复测不可达。修复将 Forge 复测的 swap 门禁改为"增长容忍"（`swap_growth_within_tolerance`，默认 4 MiB，实测噪声远低于此值；真实换页压力为数百 MB 量级仍会被拒绝），并在 summary 中保留 start/max/growth 全量数值。v1 保留为口径调试记录；口径修复后使用新 run id 重跑，其数据不得直接作为 `M_D` 证据。该批次现位于 `_archive/stopped/forge/`。

**Forge 复测 v2 完成记录（2026-08-29）**：`forge_b1_reprofile_20260829_verified_v2` 基于干净 commit `50d2362adf00fa86fb2c28a9c1b1261af15fc563` 通过全部 10 项门禁（0 failure、swap 增长 0 字节、16 物理核归还、采样/GPU 线程 affinity `[28,29,30,31]`）。空闲整卡高水位 415 MiB（v1 已触发 Forge 模型加载后的稳态），Forge 常驻基线 419 MiB；5 轮 b=1 txt2img 延迟 mean/p95 为 4.844/4.932 s（低于 2026-08-28 旧口径的 5.44 s），逐轮峰值稳定在 5667 MiB（每轮 96–100 个 GPU 样本）。**`M_D(b=1) = 5252 MiB`**（peak − idle 高水位口径），比旧估算 5590 MiB 低约 338 MiB。该值与 C2/B2 的 `max_card_delta_mb` 同为相对空闲整卡的增量口径。

**C3 复测后正式筛选（2026-08-29）**：`experiment_results/elastic_residency/c3_analysis/c3_h_reprofiled_20260829.json`（`analyze_results.py`，C2 v3 + B2 v1 + Forge v2）：

- `G_safe = 501.3 MiB`（仍由 B2 T0 的 allocator slack 主导；idle 高水位 415、重复运行漂移 0；colocated 残差 pending）。
- `H_TTS(T0) = 16`（三上下文一致；K24 距预算差约 17–119 MiB）。
- `H_TTS(T3) = 32` 且 FULL 可行（三上下文一致）。
- **`H_D` 按上下文分化**：512 下为 8（K8@512=2365 ≤ 预算 2397.7），2048/4096 下为 0（K8=2417/2431 超预算）。按 3.3 节规则保留按 context 分组的档位，不合并成常数。
- 补测建议（脚本输出）：`k = {2, 4, 6, 10, 12, 14, 18, 20, 28, 30}`。

这些仍是容量筛选值：colocated 残差分量未进入 `G_safe`，最终可行性必须由目标组合（H_TTS 档位 + TTS 配置、H_D 档位 + Forge）真实同时加载并运行证明。

### 阶段 D：辅助任务动态核心预算

初始固定 `R_sys=2` 个物理核。辅助任务在独立进程中运行，结束后进程退出并释放 affinity，主流水线不得永久少核。

测试矩阵：

```text
JSON RAG:                 1, 2 cores
BGE dense embedding:     1, 2, 4 cores
Faster-Whisper Small:    1, 2, 4 cores
system/audio/scheduler:  fixed 2 cores
```

JSON RAG 使用冻结的 1000 条 JSONL corpus 和固定 query 集；BGE 使用 ONNX Runtime 的 `intra_op_num_threads` 加物理核 affinity；ASR 使用同一批 5–10 s 中文 WAV、CPU int8、beam size 1。每类选择满足冻结 SLA 的最小核心数；若最大预算仍不满足，结果标记为 infeasible，不自动扩大矩阵。

随后正式测试：

```text
R_sys = 0, 1, 2, 4
```

每个 `R_sys` 分别运行 JSON RAG、BGE、ASR 与一次固定主流水线，并采集真实音频 underrun、调度 jitter、辅助任务完成时间和主流水线损失。辅助任务结束后核验 affinity 和可用核心集合恢复。

依赖状态：`faster-whisper==1.2.1`、`ctranslate2==4.7.2`、`onnxruntime==1.23.2`、`sounddevice==0.5.5`、`blendmodes==2025`、`pillow==10.4.0` 和所需系统音频库均已安装，`pip check` 通过。播放探针优先使用 WSLg Pulse，失效时可回退到 Windows WASAPI；正式 underrun 验收仍须使用真实 TTS chunk 和并发主流水线，而不是 1 s 静音 smoke。

### 阶段 E：无图像静态放置与真实重叠

只测试阶段 B/C 证明可加载的配置：

1. LLM FULL GPU + TTS CPU；
2. LLM CPU + TTS GPU；
3. LLM `H_TTS`/临界候选 + TTS T0；
4. LLM `H_TTS`/临界候选 + TTS Pareto 混合配置；
5. 容量预检通过时，LLM FULL GPU + TTS T0。

`LLM CPU + TTS CPU` 只保留为故障回退，不进入性能矩阵。

实验必须使用真实 token stream：确定性分句器产生第一完整句后立即提交 TTS 子进程；LLM 继续生成后续句子，记录两阶段同时处于 running 的交集时长。若交集为零，该轮判为“未实现真实重叠”，不能计入结果。

短/中/长回复各至少 5 轮，并累计至少 30 个音频 chunk。记录第一 token、第一完整句、TTFA、每个 chunk ready/play 时间、underrun、完整回复时间、LLM tokens/s、TTS RTF、显存/内存/CPU 峰值。

最终保留同时满足以下条件的 Pareto 配置：

- 无 OOM、swap、chunk 丢失/重复/乱序；
- 真实播放验证中 underrun 为 0；
- 调度 jitter 没有超出冻结门槛；
- 辅助任务满足各自 SLA；
- 主流水线相对同放置无辅助任务基线损失不超过 5%。

如果所有配置都因 `RTF_p95 > 1` 无法持续播放，实验仍完整保存，但结论必须是“TTS 后端阻塞当前无断流目标”，下一阶段先优化/替换 TTS 后端，而不是继续实现联合调度器。

## 5. 结果目录与完成判据

```text
experiment_results/elastic_residency/<stage>/<run_id>/
  run_config.json
  manifest.json
  iterations.jsonl
  events.jsonl
  timeline.jsonl
  summary.json
  summary.csv
  failures.jsonl

experiment_results/elastic_residency/_archive/
  superseded/<stage>/<run_id>/
  stopped/<stage>/<run_id>/
  debug_smoke/<stage>/<run_id>/
```

当前批次和全部归档映射见 `docs/ELASTIC_RESIDENCY_RESULT_CATALOG_20260830.md`。分析脚本不得自动扫描 `_archive`。

每个阶段完成必须同时满足：

- 代码、静态/Mock 测试和真实 smoke 均通过；
- manifest 能证明模型哈希、真实后端、CPU affinity 和 Git dirty 状态；
- 原始记录数量与矩阵期望一致，失败轮次有原因；
- 汇总可从原始记录重建；
- 文档只引用当前 run id，不混用历史数据；
- 没有人工确认前不修改 `01_paper_main/`。

## 6. 建议的逐步交付节奏

**当前状态（2026-08-31）**：C2+C3 direct NVML 已完成 16 档（共 240 条）性能曲线，KV 校准 48/48、T0/T3 direct NVML v2 30/30 与 Forge direct NVML v1 5/5 均已完成技术核验。Forge 新批次 `M_D=5484 MiB`；当前用完整 16 档、B2 direct v2 与 Forge direct v1 重新筛选共置候选。SM-A/B/C 不在本计划 A–E 阶段内，其暂定 NO_GO/暂停状态见 `paper/thesis_experiment_guide/10_gates_and_current_status.md`。正式运行前必须从当时的干净 Windows HEAD 读取完整 commit，不在长期文档中硬编码临时 HEAD 或短期测试计数。

- [x] **A1**：GPU/jitter/CPU 首样本/provenance 硬门槛已通过。证据：`a1_smoke_20260829_verified_v7`；v6 仅作历史调试记录。
- [x] **B1**：T0–T3 真实单文本 smoke 通过。证据：`b1_tts_components_20260829_verified_v2`。
- [x] **B2**：旧 60 条矩阵筛出 T0/T3；当前 direct NVML v2 `b2_tts_pareto_matrix_20260831_direct_nvml_v2` 基于干净 `0261e4da` 完成 30/30 技术核验，0 failure/fallback，运行期 swap 与同步查询新鲜度证据完整，尚未晋升论文正文数据。
- [x] **C1**：K0/FULL 的 512-token 真实端点 smoke 已通过。证据：`c1_llm_residency_512_20260829_verified_v1`。
- [x] **C2 direct NVML 重跑**：`c2_llm_residency_curve_20260829_direct_nvml_v2` 已完成 90/90 条并通过，干净 commit 为 `17c40d3495cb01237bdf2e7429f2635362e908cf`；旧 CLI 运行只保留为工程基线。
- [x] **C3 工具**：Forge b=1 复测脚本、`G_safe`/`H_TTS`/`H_D` 筛选分析与临界补测建议已实现并通过静态测试；legacy 口径初步筛选见 `experiment_results/elastic_residency/c3_analysis/c3_h_screen_20260829_legacy_forge.json`。
- [x] **C3 Forge 旧后端复测**：`forge_b1_reprofile_20260829_verified_v2` 通过当时门禁，`M_D = 5252 MiB`；`c3_h_reprofiled_20260829.json` 只保留为候选筛选（`H_TTS(T0)=16`、`H_TTS(T3)=32+FULL`、`H_D` 为 512:8 / 2048:0 / 4096:0），不得当作 direct NVML 冻结值。
- [x] **C3 临界补测**：`c3_llm_critical_refinement_20260829_direct_nvml_v2` 已完成 150/150 条并通过；与 C2 direct v2 合并后覆盖 16 个 LLM 驻留档位。
- [x] **C3 KV 正式校准**：`c3_kv_cache_calibration_20260830_direct_nvml_v1` 已完成 48/48 条并通过，0 failure；`n_ctx/KV tokens=640/2176/4224` 对应 KV buffer 总量 `24/72/136 MiB`，完整序列化状态 `70.263/118.292/182.331 MiB`。
- [ ] **当前 C3 收尾**（下一项为离线完整 16 档候选筛选）：
  1. [x] `forge_b1_reprofile_20260831_direct_nvml_v1`：真实 Forge b=1、warmup 1 次、正式 5 轮全部通过；direct NVML、逐窗 750 ms 样本年龄、swap 增长与核心归还均通过，`M_D=5484 MiB`；
  2. 基于完整 16 档 LLM 曲线、B2 direct v2 与 Forge direct v1 重新筛选最小共置候选；分析入口必须同时读取 C2/C3，并拒绝每个 context 非 16 档的输入；
  3. 先跑 LLM+TTS 单组合 smoke，再对保留的 context/LLM/T0-T3 组合各跑 5 轮，得到 colocated 正残差；
  4. 重算 `G_safe`，冻结按 context 分组的 `H_TTS/H_D`，然后进入 D 阶段。

**C3 临界补测 v2 停止记录（2026-08-29）**：`c3_llm_critical_refinement_20260829_verified_v2` 完成 94/150 条后，在 C2048/K20/r5 因推理期 GPU 样本最大年龄 772.875 ms 超过 750 ms 有效性硬门而停止；其余有效性检查及该轮 jitter 性能目标均通过。94 条中所有记录的最大样本年龄均超过 500 ms，10 条超过 700 ms，第 91 条已达 749.570 ms。根因是周期 sampler 每次先查询整卡指标、再查询 WSL 驱动无法数值化返回的 compute-process 显存，后者不贡献正式曲线证据却显著拉长探针周期。修复仅在 LLM 周期 sampler 中关闭无效的进程显存查询；阶段性显存快照与整卡基线差值估算保持不变，750 ms 门槛不放宽。v2 保留为停止记录，不续跑、不用于完整曲线，现位于 `_archive/stopped/c3/`。

**direct NVML 口径修订（2026-08-29）**：对 v2 的第一次修复只减少了 CLI 查询次数，仍未消除子进程采样对论文实验的扰动，因此 `e446e33` 下的 card-only smoke 也只保留为调试记录。正式 GPU 遥测改为 ctypes direct NVML，CLI 不允许静默 fallback；本机空闲校准中 direct NVML 完整设备指标查询约为毫秒级，且 manifest/summary/acceptance 显式锁定后端。由于测量后端变化，C2 粗测、C3 细化及用于安全余量的 B2/Forge 整卡指标均须在新后端下重跑或校准，旧结果不得与新结果拼接冻结阈值。

**C2 direct NVML 完成记录（2026-08-29）**：`c2_llm_residency_curve_20260829_direct_nvml_v2` 基于干净 commit `17c40d3495cb01237bdf2e7429f2635362e908cf` 完成 90/90 条，`summary.passed=true`、0 failure。全部轮次使用 `direct_nvml` 且 `gpu_cli_subprocess_used=false`；推理期最大 GPU 样本年龄为 685.582 ms，最大单次探针耗时为 504.778 ms，deadline recovery 合计 3 次，均未突破 750 ms 有效性门。所有轮次无 swap、固定生成 128 tokens，并完成 KV save/restore。该 run 取代旧 CLI C2 作为当前完整粗测曲线。

**C3 direct NVML 完成记录（2026-08-29）**：`c3_llm_critical_refinement_20260829_direct_nvml_v2` 使用同一干净 commit 完成 `10 配置 × 3 上下文 × 5 轮 = 150/150` 条，`summary.passed=true`、0 failure。全部轮次使用 `direct_nvml` 且未启动 GPU CLI；推理期最大 GPU 样本年龄为 630.711 ms，最大单次探针耗时为 425.154 ms，deadline recovery 合计 11 次。该 run 与 C2 direct v2 共同构成当前 16 档完整驻留曲线；旧的 94 条停止记录不参与合并。

**KV/共置测量实现记录（2026-08-30）**：新增 `kv_measurement.py`、`llm_kv_cache_calibration.py` 与 `llm_tts_colocated.py`。历史 `kv_state_bytes` 明确为 `save_state()` 完整序列化状态，不再表述为纯 GPU KV 显存；独立 KV 校准从 llama.cpp 初始化日志采集 GPU/CPU/CUDA_Host KV/compute buffer，并分别查询单序列状态、原生 position 边界和 `llm.n_tokens`。校准和共置入口都以 `n_ctx = context + generation`（正式默认 128）重建与 C2/C3 相同的推理后状态。共置入口在同一进程实测 `B_LLM`、TTS 窗口整卡峰值 `P_colocated`、PyTorch 分阶段峰值与正残差，正式矩阵支持每组合 5 轮。Windows 静态测试 94/94 通过；`kv_calibration_smoke_20260830_verified_v1` 在干净 commit `2dca3651d94978523e7cddd24826099321bd760b` 下保存了 1/1 条通过的 direct NVML K0 artifact（C64+8，KV token=72，0 failure），另有 K2 临时 smoke 识别到 CUDA0 KV/compute buffer；最小 K0+T3 共置临时 smoke 也通过 direct NVML、生成 1.2 s 非静音 WAV。这些 smoke 只证明真实链路可运行，不代替 48 条 KV 正式产物或共置正式矩阵。KV smoke 现位于 `_archive/debug_smoke/kv_calibration/`。

**KV 正式校准完成记录（2026-08-30）**：`c3_kv_cache_calibration_20260830_direct_nvml_v1` 基于干净 commit `820ee0fbcbbf671a791d31786bf171bf2a241d1a` 完成 `16 配置 × 3 上下文 × 1 轮 = 48/48` 条，`summary.passed=true`、0 failure、0 swap、0 GPU unavailable，模型哈希与 C2/C3 相同。`n_ctx/KV tokens=640/2176/4224` 时，预分配 KV buffer 总量为 `24/72/136 MiB`；FULL 的 GPU KV 部分同为 `24/72/136 MiB`；完整 `save_state()` 状态为 `70.263/118.292/182.331 MiB`。该 run 证明 KV 容量和完整序列化状态远小于旧稿误写的 1--2 GiB，但不单独冻结共置阈值。
- [ ] **D1（未到正式矩阵）**：物理核拓扑、affinity 与租约基础设施已完成；BGE 与 Faster-Whisper 仅单轮接口 smoke；正式入口 `auxiliary_core_budget.py` 尚未实现，因此 JSON RAG/BGE/ASR 的 1/2/4 核曲线与 `R_sys={0,1,2,4}` 均未跑；LLM/TTS 的 `{2,4,6,8,10,12,14,16}` 核扩展曲线也未跑。需先实现 `auxiliary_core_budget.py` 与 CPU 核数画像入口，再进入 D2/D3 正式矩阵。
- [ ] **D2**：接入已验证的 Faster-Whisper，并在音频路由就绪后接入真实 underrun 探针。
- [ ] **D3**：运行 `R_sys={0,1,2,4}` 正式验证。
- [ ] **E1**：实现真实 LLM→TTS 重叠实验并做单轮 smoke。
- [ ] **E2**：只对前序阶段保留下来的配置跑完整矩阵，形成最终 Pareto/停止结论。

每一步完成后先检查原始证据和停止条件，再进入下一步；不会一次性修改生产调度器。
