# 研究定位、工作流与任务抽象

> 导航：[总索引](README.md) · 兼容入口：[THESIS_EXPERIMENT_GUIDE_DRAFT.md](../THESIS_EXPERIMENT_GUIDE_DRAFT.md)

> **版本**：v5.0
> **最后更新**：2026-08-30
> **状态**：研究设计与实验路线草案。本文档描述目标、口径和验收条件，不代表所有机制已经实现或已有真实实验支持。

## 0. 使用原则

本指南用于统一 EdgeSched 的研究问题、实现顺序、实验口径和论文证据链。当前实现、测试与真实运行结果的优先级高于本文档中的设计设想。

开始相关工作前，应依次核对：

1. `docs/PROJECT_INDEX.md`
2. `docs/PROJECT_CONTEXT.md`
3. `experiments/README.md`
4. `paper/experiment_data/README.md`
5. `docs/ELASTIC_RESIDENCY_EXPERIMENT_EXECUTION_PLAN_20260828.md`
6. 对应源码、测试、原始 JSON 与运行 manifest

必须遵守以下边界：

- Mock 结果只能验证调度语义和统计链路，不能证明真实模型性能。
- `experiment_results/` 是探索性运行产物；只有人工核对后，结果才可晋升到 `paper/experiment_data/01_paper_main/`。
- 换用新模型、修改调度策略或改变运行环境后，旧性能数字默认失效。
- 本文档不再把历史设计报告、旧稿或 `paper/previous/` 当成当前实现真源。
- 论文只陈述已经由真实运行和可追溯证据支持的能力。

---

## 1. 当前研究定位

### 1.1 目标场景

EdgeSched 面向单用户、单机、资源受限的本地多模态 AI 助手。目标不是多租户云服务，也不是分布式集群调度，而是在一台消费级设备上协调长期交互服务与短时异构任务。

当前计划使用的主要工作负载为：

| 工作负载 | 计划模型/后端 | 运行语义 |
|---|---|---|
| LLM + Vision | Qwen3.5-4B-Q4_K_M + 对应 BF16/F16 `mmproj`，GGUF + llama.cpp/libmtmd | 统一文本与真实图片理解；语言层可重新配置，视觉 projector 可作为 session-warm 组件驻留或释放 |
| TTS | Qwen3-TTS-12Hz-0.6B-Base | 下游、句子级流式、CPU/GPU 均可运行 |
| Image | SDXL Turbo + Forge | GPU 必需、非硬抢占、一次性或短队列任务 |
| ASR | Faster-Whisper Small，CTranslate2 FP16 权重 | CPU 辅助阶段，不作为主要显存竞争者 |
| RAG embedding | BGE-small-zh-v1.5，ONNX，512 维 | CPU 辅助阶段；dense embedding/rerank，不冒充 cross-encoder |

除新选定的 4B 主 GGUF 与 `mmproj` 外，既有后端模型已在当前 WSL2 副本完成路径和关键权重核验；新模型仍待下载和哈希冻结。所有正式实验仍必须把量化方式、运行参数和哈希写入 manifest，不能只引用本指南中的快照。

**2026-08-31 模型迁移决策**：原 Qwen3.5-9B-Q4_K_M 的 direct NVML 曲线、KV、LLM+TTS 和 LLM+Forge smoke 保留为大模型压力与负面边界证据；主实验改用 Qwen3.5-4B-Q4_K_M 加真实执行的 `mmproj`。旧 9B 性能数字默认不再代表主实验模型。

**2026-08-31 artifact 与环境冻结（已核验）**：主模型为 `Qwen3.5-4B-Q4_K_M.gguf`（`general.architecture=qwen35`、`block_count=32`、`context_length=262144`、`embedding_length=2560`、`file_type=15`、426 张量、2740937888 B、SHA-256 `00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4`，Unsloth 量化）；视觉 projector 为 `mmproj-BF16.gguf`（`general.architecture=clip`、`general.type=mmproj`、`clip.projector_type=qwen3vl_merger`、`clip.has_vision_encoder=true`、24 层视觉块、`image_size=768`、`patch_size=16`、`spatial_merge_size=2`、`projection_dim=2560`、298 张量、675569344 B、SHA-256 `302b92d565080b9cc0281186979ae75a7429ec23d14f6f7607a035539b21f3a6`）。执行后端为 `llama-cpp-python==0.3.35` 随包的 `libmtmd`，media marker 为 `<__media__>`。这些文件体积仍只是 artifact 尺寸，**不能**代替运行期显存/RSS；真实的 \(M_V\) 必须来自 projector warm 前后的整卡显存差值与进程 RSS 差值。当前真实 GPU 视觉 smoke 尚未运行，因此 4B 没有任何性能或显存结论。

### 1.2 权威实验环境

- GPU：NVIDIA GeForce RTX 5070 Laptop GPU，8151 MiB 可见显存
- CPU：AMD Ryzen 9 8940HX
- 主机内存：32GB DDR5
- 正式实验约束：WSL2 实际可用内存上限约 16GB；必须在 manifest 中记录当次内存上限与 swap 状态
- WSL2 运行副本：`/home/leslie/mvp_core`
- 调度与实验环境：`/home/leslie/.venvs/mvp_core-cu128`
- Forge 环境：`/home/leslie/.venvs/forge-cu128`

论文不得只写“32GB RAM”而隐藏 WSL2 的有效内存约束。宿主机容量与实验进程实际可用容量需要同时报告。

### 1.3 新版核心问题

原稿主要研究：

> 当图像任务到达时，将整个 LLM 从 GPU 迁到 CPU；图像完成后再恢复 LLM。

该机制仍然有价值，但如果系统只能在“LLM 占 GPU”和“Diffusion 占 GPU”之间二选一，资源模型仍然过于二元。

新版研究问题是：

> 在显存、GPU 计算份额、CPU 核心和系统内存同时受限时，如何根据多模态流水线状态、任务时长和转换代价，联合决定模型驻留程度、设备放置、CPU 核心预算、GPU SM 预算与图像执行方式，从而保持文本、语音和图像交互的可用性？

建议将核心贡献描述为：

> **面向模型无法全部驻留于加速器的本地多模态流水线，进行代价感知的弹性驻留与 CPU/GPU 联合资源调度。**

“非对称迁移”保留为弹性驻留状态转换中的关键机制，而不是整篇论文唯一的创新点。

论文不得把“消费级设备”本身写成创新。5090 等大显存设备同样属于消费级产品，真正需要强调的是以下资源区间：

\[
\sum_i M_i^{FULL}>V_{GPU},
\]

即所有活跃模型无法以完整高性能配置同时驻留于 GPU，CPU/DDR 因而从辅助运行环境变成一等计算与驻留资源。与只在完整共驻留模型之间分配 GPU 时间或 SM 的系统相比，EdgeSched 还必须决定哪些权重、组件和状态能够留在 GPU、哪些退回 CPU/DDR，以及转换成本是否值得支付。

新增 GPU SM 分配不单独作为创新点。已有工作已经研究 MPS、Green Contexts、时空 GPU 共享以及 LLM/语音阶段共运行；EdgeSched 的差异必须落在：

1. 8GB 级硬显存约束下，模型不能全部完整共驻留；
2. 独立 LLM、TTS 与 Diffusion 后端之间存在真实跨设备驻留与状态转换；
3. CPU 物理核、GPU SM、显存驻留和转换方式由同一个计划联合决定；
4. 目标是单用户交互中的 TTFA、连续播放和第一张图片时间，而不是只优化服务器吞吐。

---

## 2. 工作流与任务语义

### 2.1 语音工作流

```text
ASR / RAG
    -> streaming LLM
    -> deterministic sentence splitter
    -> bounded TTS chunk queue
    -> TTS inference
    -> ordered audio playback
```

形式化为：

\[
ASR/RAG \rightarrow LLM_{\mathrm{stream}}
\rightarrow TTSChunkQueue \rightarrow TTS \rightarrow Playback.
\]

要求：

- LLM 以 token 流产生文本。
- 断句器在稳定边界产生 TTS chunk。
- TTS 模型常驻，不能每个 chunk 重新加载。
- 一个 TTS chunk 内部暂不抢占，chunk 之间允许重新调度。
- 音频必须按 chunk 序号输出，不允许丢失、重复或乱序。
- 取消后，排队 chunk 立即清除；正在执行的 chunk 可以自然完成，但结果必须丢弃。
- 实验中关闭随机逗号断句，保证相同文本得到相同切分。

### 2.2 视觉理解工作流

视觉理解请求为：

\[
ImageInput \rightarrow VisionProjector_{\mathrm{mmproj}}
\rightarrow LLM_{\mathrm{stream}}.
\]

Qwen3.5 的 `mmproj` 必须加载对应模型的真实文件，并对固定图片执行实际编码；不能只把 projector 文件大小作为静态显存填充。正式画像至少区分文本常驻、projector GPU warm、projector CPU/未加载、真实图片编码峰值和图片 token 进入 LLM 后的 KV 状态。连续视觉对话可以保留 projector warm；视觉队列清空后是否释放由后续 queue horizon 与转换成本决定。

图片输入集必须固定分辨率、内容、SHA-256、prompt 和图像 token 统计。识图结果为空、静默 fallback 或 projector 未实际执行的轮次无效。

当前实现把这些要求落成四个互斥状态：`TEXT_ONLY`（projector 未加载，固定 token 填充 prompt）、`TEXT_ONLY_CHAT`（projector 未加载，与视觉轮次同 prompt 但无图片）、`VISION_WARM_GPU` 与 `VISION_WARM_CPU`。固定输入为内置确定性 448×448 PNG，内容身份用像素哈希 `pixel_payload_sha256=a1514034f07cd8362119d5fd124948d1b0970515d83fdba535c5b3a7b243cb00` 固定（与 PNG 编码器版本无关），spec 哈希为 `d4c6fc1c7f5cb4420a14470cd775fc9f90b6fe2c0644ab62f2913b0d49a0efb8`；按 `patch=16`、`merge=2` 期望 196 个视觉 token，真实值以 libmtmd 统计为准并与该期望交叉核对。视觉 token 数与上下文位置推进分别记录，启用 M-RoPE 时两者不互相替代。

### 2.3 图像生成工作流

普通单图请求为：

\[
LLM \rightarrow Diffusion(b=1).
\]

普通请求要求立即进入资源准备和执行流程，不为了收集未来请求而增加主动等待窗口。迁移、模型加载和显存释放时间仍计入第一张图片延迟。

### 2.4 多图语义

必须区分两类情况：

1. **同一 prompt 的多候选图**：可以比较 \(b\in\{1,2,4\}\)。
2. **不同 prompt 的独立请求**：当前 Forge 路径保持有界队列和串行执行，不能根据同一 prompt 的 `batch_size` 实验声称可以任意合批。

规则：

- 不等待尚未到达的图像请求。
- b=8 不进入默认候选集。
- 第一张普通图片使用 b=1。
- 图片队列非空时不在每两个任务之间反复恢复 LLM。
- 只有同一请求明确要求多候选图时，才允许使用 b=2 或 b=4。

现有 `experiment_results/batch_scaling/` 结果是探索性特性数据，当前只证明同一 prompt 下的 batch 扩展趋势；正式实验需要对每个 batch shape 独立预热并完善 manifest。

### 2.5 轻量任务

ASR、RAG embedding 和文档解析用于证明调度器能够容纳 CPU 辅助阶段，但它们不是论文的主要显存竞争对象。

OCR 若由多模态 LLM 完成，不必额外引入独立大模型。不得为了凑任务数量而人为增加不符合应用流程的模型。

### 2.6 通用任务抽象：设备亲和性、驻留能力与 GPU 计算份额能力

EdgeSched 不按 LLM、TTS 或 Diffusion 等模型名称硬编码路由规则。每个阶段首先声明三个正交维度；第三个维度描述阶段及其后端能否接受可验证的 GPU 计算份额约束，而不是根据一次平均 GPU 利用率猜测所需算力。

#### 设备亲和性

设备亲和性描述阶段能够或倾向于在哪里执行：

\[
d_i\in\{
CPU\_ONLY,
CPU\_PREFERRED,
GPU\_PREFERRED,
GPU\_REQUIRED
\}.
\]

- `CPU_ONLY`：没有可用的 GPU 执行路径；
- `CPU_PREFERRED`：CPU 路径已足够，使用 GPU 的收益不足以覆盖占用与转换成本；
- `GPU_PREFERRED`：CPU 和 GPU 均可执行，但 GPU 通常具有明显性能优势；
- `GPU_REQUIRED`：该阶段的有效执行路径要求 GPU，必要时可以排队或要求其他阶段释放显存。

#### 驻留能力

驻留能力描述模型状态能否跨 CPU/DDR 与 GPU 拆分，不能再压缩成一个 `migratable` 布尔值：

\[
\rho_i\in\{
FIXED,
WHOLE\_MIGRATABLE,
PARTITIONABLE
\}.
\]

- `FIXED`：运行期只允许一个固定驻留方案；
- `WHOLE_MIGRATABLE`：可以在完整 CPU 与完整 GPU 实例之间切换，但不能拆分模型；
- `PARTITIONABLE`：允许经过验证的子结构驻留在 GPU，其余部分驻留在主机 DDR 或由 CPU 执行。

对于 `PARTITIONABLE` 阶段，还需要声明拆分粒度：

\[
g_i\in\{LAYER,COMPONENT,STATE\}.
\]

- `LAYER`：按连续模型层或计算块划分；
- `COMPONENT`：按 talker、tokenizer、encoder、decoder 等相对独立组件划分；
- `STATE`：仅将 KV Cache、条件编码或其他运行状态放置到不同内存层级。

设备亲和性与驻留能力不能互相替代。例如，一个阶段可以同时是 `GPU_REQUIRED + PARTITIONABLE`：主要计算仍必须在 GPU 上，但部分权重或组件可以驻留在 DDR。另一个阶段可以是 `GPU_PREFERRED + WHOLE_MIGRATABLE`：CPU 路径可用，但实现只支持整体切换。

#### GPU 计算份额能力

GPU 计算份额能力属于“阶段 + 后端”的联合能力，而不是模型名称的固定属性：

\[
\chi_i\in\{
UNMANAGED,
CAP\_LIMITABLE,
SM\_PARTITIONABLE
\}.
\]

- `UNMANAGED`：能够在 GPU 上运行，但当前后端不能施加或验证独立计算份额；
- `CAP_LIMITABLE`：可以通过 CUDA MPS 等机制限制客户端可使用的执行资源上限，但该上限不等同于独占保留；
- `SM_PARTITIONABLE`：可以通过经过验证的 Green Context、静态 MPS 分区或等价机制，在离散 SM 子集中提交该阶段的全部 GPU kernel。

GPU 计算份额统一记录为实际有效 SM 数：

\[
s_i\in\mathcal S_i=\{S_i^0,S_i^1,\ldots,S_i^n\}.
\]

百分比只用于请求配置和展示。正式分析必须记录 GPU 的物理 SM 总数、后端要求的最小分区粒度、请求比例、舍入后的有效 SM 数以及客户端实际观测值。平均 GPU utilization 为 20\% 不等于该阶段只需要 20\% 的 SM；它可能来自突发 kernel、CPU 等待、显存带宽瓶颈或小 batch 并行度不足。

显存驻留与 SM 份额必须分开描述。限制 SM 不会自动限制显存，也不会把超出的权重分页到 CPU；限制显存也不会保证某个阶段获得固定 GPU 算力。因此，一个 GPU 配置至少写成：

\[
G_i=\langle r_i,s_i,m_i,backend_i\rangle,
\]

其中 \(r_i\) 是驻留档位，\(s_i\) 是有效 SM 预算，\(m_i\) 是显存上限或预测峰值，\(backend_i\) 是 `NONE`、`MPS` 或 `GREEN_CONTEXT` 等实际执行后端。

每个阶段最终提供有限、离散且经过 profiling 的驻留集合：

\[
r_i\in\mathcal R_i=\{R_i^0,R_i^1,\ldots,R_i^m\}.
\]

声明 `PARTITIONABLE` 不等于允许任意百分比切分。只有真实后端能够构造、能够正确运行并已有资源画像的配置，才进入 \(\mathcal R_i\) 或 \(\mathcal S_i\)。当前可用于说明抽象的实例包括：LLM 的层级/状态拆分、组件化 TTS 的组件拆分，以及只允许完整 GPU 运行的图像阶段；这些只是任务画像实例，不是调度器中的模型名称分支。

---
