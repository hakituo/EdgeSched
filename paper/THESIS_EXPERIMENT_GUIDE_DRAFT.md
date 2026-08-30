# EdgeSched 论文与实验指南（弹性驻留与异构资源协同版）

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

- --

## 1. 当前研究定位

### 1.1 目标场景

EdgeSched 面向单用户、单机、资源受限的本地多模态 AI 助手。目标不是多租户云服务，也不是分布式集群调度，而是在一台消费级设备上协调长期交互服务与短时异构任务。

当前计划使用的主要工作负载为：

| 工作负载 | 计划模型/后端 | 运行语义 |
|---|---|---|
| LLM | Qwen3.5-9B-Q4_K_M，GGUF + llama.cpp | 常驻、流式、可在 CPU/GPU 间重新配置 |
| TTS | Qwen3-TTS-12Hz-0.6B-Base | 下游、句子级流式、CPU/GPU 均可运行 |
| Image | SDXL Turbo + Forge | GPU 必需、非硬抢占、一次性或短队列任务 |
| ASR | Faster-Whisper Small，CTranslate2 FP16 权重 | CPU 辅助阶段，不作为主要显存竞争者 |
| RAG embedding | BGE-small-zh-v1.5，ONNX，512 维 | CPU 辅助阶段；dense embedding/rerank，不冒充 cross-encoder |

上述模型文件已在当前 WSL2 副本完成路径和关键权重核验，但量化方式、运行参数和哈希仍必须写入每次正式实验 manifest，不能只引用本指南中的快照。

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

- --

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

### 2.2 图像工作流

普通单图请求为：

\[
LLM \rightarrow Diffusion(b=1).
\]

普通请求要求立即进入资源准备和执行流程，不为了收集未来请求而增加主动等待窗口。迁移、模型加载和显存释放时间仍计入第一张图片延迟。

### 2.3 多图语义

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

### 2.4 轻量任务

ASR、RAG embedding 和文档解析用于证明调度器能够容纳 CPU 辅助阶段，但它们不是论文的主要显存竞争对象。

OCR 若由多模态 LLM 完成，不必额外引入独立大模型。不得为了凑任务数量而人为增加不符合应用流程的模型。

### 2.5 通用任务抽象：设备亲和性、驻留能力与 GPU 计算份额能力

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

- --

## 3. 系统状态与决策动作

### 3.1 调度状态

在调度事件发生时，系统维护状态：

\[
x_t = \langle
q_I, q_T, K, \{r_i\}_{i\in\mathcal A_t}, M_G, S_G, U_C, v, h
\rangle,
\]

其中：

- \(q_I\)：图像队列及当前图像任务状态；
- \(q_T\)：待合成 TTS chunk 数量；
- \(K\)：当前上下文和 KV Cache 规模；
- \(\mathcal A_t\)：当前活跃或候选执行的阶段集合；
- \(r_i\)：阶段 \(i\) 当前采用的离散驻留配置；当前工作负载中包括 \(r_L\) 和 \(r_T\)；
- \(M_G\)：可用显存及近期峰值；
- \(S_G\)：GPU 物理 SM 总量、各阶段当前有效 SM 预算、分区后端及近期 GPU 利用率/带宽压力；
- \(U_C\)：CPU 利用率、核心拓扑和可用核心预算；
- \(v\)：是否处于连续语音会话；
- \(h\)：近期任务时长、到达间隔与转换历史。

调度不需要在每个 token 后重新求解。决策点限定为：

- 新图像请求到达；
- 新 TTS chunk 入队；
- 图像任务完成；
- TTS 队列清空；
- GPU 分区 worker 创建、退出或分区能力失效；
- LLM 回复或语音会话结束；
- 模型重新配置完成或资源压力越界。

### 3.2 调度动作

通用联合动作表示为：

\[
a_t=\left(\{(r_i,c_i,s_i,b_i)\}_{i\in\mathcal A_t},\mu\right),
\]

其中：

- \(r_i\)：阶段 \(i\) 的设备和驻留档位；
- \(c_i\)：阶段 \(i\) 的 CPU 核心预算；
- \(s_i\)：阶段 \(i\) 的有效 GPU SM 预算；不在 GPU 上执行时取 0，后端不能受控时标记为 `UNMANAGED` 而不是伪造为满卡；
- \(b_i\)：阶段 \(i\) 的可选 batch 或并发参数；不适用时取 1；
- \(\mu\)：保持、紧急迁出、原子切换、恢复或重新配置。

对当前 LLM、TTS、Diffusion 工作负载，该动作可具体展开为：

\[
a_t=(r_L,r_T,c_L,c_T,s_L,s_T,s_D,b_D,\mu).
\]

该展开只是一种 workload 实例，不是调度接口的固定上限。

### 3.3 候选驻留档位

候选集合由驻留能力生成：

- `FIXED` 只产生一个可选驻留配置；
- `WHOLE_MIGRATABLE` 可以产生完整 CPU 和完整 GPU 配置；
- `PARTITIONABLE` 可以在完整配置之间加入经过真实验证的层级、组件级或状态级配置。

LLM 第一版使用离散档位：

\[
r_L\in\{CPU,H_1,H_2,H_3,FULL\_GPU\}.
\]

其中 \(H_k\) 对应经过真实 profiling 验证的 `n_gpu_layers` 配置，而不是平均切成固定百分比。

TTS 的基础配置为：

\[
r_T\in\{CPU,GPU\}.
\]

如果真实后端验证组件拆分可行，则扩展为：

\[
r_T\in\{CPU,H_{COMP},GPU\},
\]

其中 \(H_{COMP}\) 表示显式定义的组件放置方案，例如将主要生成组件放在 GPU、将可独立执行或可缓存的辅助组件放在 CPU/DDR。不得使用不透明的 `device_map=auto` 结果直接充当稳定驻留档位；每个组件放置必须写入 manifest 并单独 profiling。

MVP 不实现 CUDA 通用分页，也不宣称原地逐页迁移。第一版只支持后端明确暴露的层级、组件级或状态级拆分。当前 llama.cpp 的 `n_gpu_layers` 是模型加载配置，因此 LLM 弹性驻留应定义为：

> 创建目标驻留配置的新实例，迁移 KV 状态，并通过紧急替换或原子热替换完成配置切换。

- --

## 4. 三类运行状态

### 4.1 S0：没有图像任务

需要评估的候选配置：

| 配置 | LLM | TTS |
|---|---|---|
| S0-A | 全 GPU | CPU |
| S0-B | CPU | GPU |
| S0-C(k) | 混合驻留 \(H_k\) | GPU |
| S0-D | 全 GPU | GPU，仅在真实显存画像允许时测试 |

选择依据不是单独模型速度，而是完整 LLM→TTS 流水线。

第一段语音时间：

\[
TTFA = T_{transition}
+T_{LLM}^{first\ chunk}
+T_{TTS}^{first\ chunk}.
\]

进入稳态后，流水线节拍近似为：

\[
T_{pipe}\approx
\max(T_{LLM}^{chunk},T_{TTS}^{chunk}).
\]

如果把 LLM 放到 CPU 后导致第一句产生过慢，即使 TTS 在 GPU 上很快，也不一定改善 TTFA。因此不能预先认定 S0-B 或 S0-C 一定优于 S0-A。

对于同时在 GPU 上执行的 LLM 与 TTS，驻留可行后还必须测试离散 SM 分配：

\[
(s_L,s_T)\in\mathcal S_{LT},\qquad s_L+s_T\le S_{GPU}.
\]

第一轮不穷举任意百分比，先测试不受控共享以及由单模型曲线推导出的少量候选，例如 `75/25`、`50/50` 和临界点邻域。选择目标不是平均利用率最大，而是最小化第一段音频时间并保证后续 chunk 能在播放截止时间前准备完成。

### 4.2 S1：一个图像任务运行

需要评估的候选配置：

| 配置 | Diffusion | LLM | TTS |
|---|---|---|---|
| S1-A | GPU | CPU | CPU |
| S1-B(k) | GPU | 混合驻留 \(H_k\) | CPU |
| S1-C | GPU | CPU | GPU |
| S1-D(k) | GPU | 混合驻留 \(H_k\) | GPU |

S1-C 与 S1-D 只有在显存安全且 Diffusion slowdown 可接受时才进入调度候选。显存放得下不等于 GPU 计算共存一定有收益，必须测量实际干扰。

S1 必须区分两类共享：

1. **显存共驻留但计算串行**：保留部分 LLM/TTS GPU 状态以减少恢复成本，Diffusion 执行时不提交它们的 kernel；
2. **显存共驻留且计算并发**：为 TTS 或部分 LLM 分配有限 SM，Diffusion 使用剩余 SM，并真实测量显存带宽竞争与 slowdown。

建议优先测试 `LLM_CPU + TTS_T3_GPU + Diffusion_GPU`。先求满足语音约束的最小 \(s_T\)，再将剩余 SM 交给 Diffusion。只有该二阶段共享形成 Pareto 改进后，才测试 LLM、TTS、Diffusion 三者同时具有 GPU 计算份额的配置。

### 4.3 S2：图像队列非空

- 保持当前可行的降级或混合驻留状态。
- 不在相邻图像任务之间恢复到 S0。
- 不同 prompt 串行执行。
- 同一请求的多候选图可以根据画像选择 b=1、2、4。
- 图片队列非空期间保持已验证的 SM 配置，除非语音 SLO、显存安全或后端健康状态越界；不因短时 utilization 波动反复重建分区。
- 队列清空后重新选择 S0 的目标配置，而不是无条件恢复 LLM 全 GPU。

- --

## 5. 配置相关任务画像

旧模型用一个静态显存值和一个 CPU/GPU 比率描述任务，无法支撑联合放置。新版首先定义静态能力描述：

\[
\Phi_i=\langle
d_i,\rho_i,g_i,\chi_i,role_i,preempt_i,queue_i,deps_i
\rangle,
\]

其中 \(d_i\) 是设备亲和性，\(\rho_i\) 是驻留能力，\(g_i\) 是拆分粒度，\(\chi_i\) 是 GPU 计算份额能力，其余字段描述阶段角色、可抢占性、可排队性和工作流依赖。

随后为每个允许的驻留配置建立动态性能画像：

\[
\mathcal P_i(r_i,s_i,z)=\langle
\hat M_i(r_i,s_i,z),\hat T_i(r_i,s_i,z),\hat C_i(r_i,s_i,z),\hat E_i(r_i,s_i,z)
\rangle,
\]

其中 \(r_i\in\mathcal R_i\)，\(s_i\in\mathcal S_i\)，\(z\) 包括核心预算、输入规模和运行环境。静态能力决定哪些配置可以被考虑，动态画像决定这些配置在当前设备上是否值得选择。SM 画像必须同时包含隔离运行和目标共运行组合；单模型曲线不能直接预测显存带宽竞争下的共运行性能。

### 5.1 LLM 画像

输入特征：

- `n_gpu_layers` / 驻留档位；
- 有效 SM 数或不受控共享标记；
- CPU 物理核心预算与线程数；
- 上下文长度和 KV Cache 大小；
- prefill token 数；
- 输出 token 数。

记录：

- TTFT；
- prefill tokens/s；
- decode tokens/s；
- token gap；
- 显存、内存峰值；
- KV 保存、恢复和实例切换时间；
- 不同驻留档位之间的转换时间。

### 5.2 TTS 画像

输入特征：

- CPU、GPU 或经过验证的组件级驻留配置；
- 有效 SM 数、MPS/Green Context 后端或不受控共享标记；
- 组件放置映射以及哪些中间表示已缓存；
- CPU 核心预算；
- 文本字符数和分句长度；
- 目标语言和固定声音配置。

记录：

- 队列等待时间；
- 首音频 chunk 延迟；
- 单 chunk 合成时间；
- 音频时长；
- 实时率 \(RTF=T_{synthesis}/T_{audio}\)；
- 峰值显存和内存；
- 设备加载或切换时间；
- talker、code tensor transfer、codec decode 与 prompt 构建的分段时间。

当前 Qwen3-TTS 路径使用 `non_streaming_mode=True`。因此现有指标 `first_chunk_ready` 表示完整句子 WAV 在主进程可用，不是模型内部第一音频采样延迟；论文在真正增量音频后端完成前不得把它简称为模型 TTFA。

### 5.3 Diffusion 画像

输入特征：

- 模型；
- 分辨率；
- steps、sampler、CFG；
- batch size；
- 有效 SM 数或不受控共享标记；
- 是否有 LLM/TTS GPU 工作共存。

记录：

- 模型冷加载和稳态生成时间；
- 第一张图片完成时间；
- 峰值显存及时间线；
- 共存时的 slowdown；
- 与 TTS/LLM 共运行时的有效 SM 分配、GPU utilization、显存带宽压力代理与尾延迟；
- 失败和 OOM 原因。

画像首先来自离线 profiling，随后可按配置使用 EMA 修正。不同配置不得共用同一个全局平均时长。

- --

## 6. 可行性约束与优化目标

### 6.1 显存约束

\[
\sum_{i\in\mathcal A_t}\hat M_i(r_i,s_i,z_i)+G_{safe}
\le V_{GPU}.
\]

在当前三模型实例中可展开为：

\[
M_L(r_L,K)+M_T(r_T)+M_D(b)+G_{safe}
\le V_{GPU}.
\]

安全余量 \(G_{safe}\) 必须根据 idle GPU 高水位、`reserved-allocated` allocator slack、单模型峰值相加与真实共置峰值之间的正残差，以及重复运行峰值漂移共同校准，不能只写死为一个未经验证的比例。容量公式只用于筛选，最终可行性必须由目标组合真实同时加载并运行证明。

### 6.2 CPU 约束

\[
\sum_{i\in\mathcal A_t}c_i+R\le C_{physical},
\]

其中 \(R\) 是设备驱动、音频播放、调度器、WSL2 和操作系统的保留预算。在当前主要 CPU 竞争者只有 LLM 和 TTS 时，该式退化为 \(c_L+c_T+R\le C_{physical}\)。

当前校准使用以下明确起点：

\[
R_{sys}=2\text{ 个物理核},\qquad C_{work}=16-2=14.
\]

被保留物理核的 SMT sibling 同时排除在 LLM/TTS affinity 外。该数值是待验证的起始配置，不是预先写死的论文结论；正式校准比较：

\[
R_{sys}\in\{0,1,2,4\}.
\]

选择同时满足以下条件的最小 \(R_{sys}\)：

- 音频播放无 underrun；
- 系统和调度线程无资源饥饿导致的明显延迟尖峰；
- 辅助阶段满足执行前冻结的绝对 SLA，并同时报告相对独立运行的 slowdown；
- LLM→TTS 主流水线相对“相同放置、无辅助任务”的隔离基线性能损失不超过 5%。

ASR、RAG 和 BERT 不永久占用这两个系统核心。它们作为短时阶段按需获得并在结束后归还核心。初始画像候选为：

| 辅助阶段 | CPU 核心候选 |
|---|---|
| JSON RAG 查询、过滤和排序 | 1、2 个物理核 |
| BERT embedding / reranking | 1、2、4 个物理核 |
| 轻量 ASR | 1、2、4 个物理核 |

每种辅助阶段选择满足自身延迟约束的最小核心数。多个辅助任务同时到达时优先排队或按工作流依赖执行，不要求它们同时占满独立核心预算。

当前建议在正式实现前冻结以下默认 SLA：JSON RAG top-32 p95 不超过 50 ms；BGE 对 1 个 query 与 32 个候选做 dense rerank 的 p95 不超过 150 ms；Faster-Whisper Small 对固定 5–10 秒中文音频的 p95 RTF 不超过 0.8。修改 SLA 必须使用新 run id，不能覆盖旧结果。

SMT 逻辑线程不能直接当作额外物理核心。正式实验必须记录物理核心、逻辑 CPU、亲和掩码和 CCD 拓扑。

### 6.3 GPU 计算份额与共运行约束

对能够形成显式空间分区的后端，GPU 计算份额满足：

\[
\sum_{i\in\mathcal G_t}s_i\le S_{GPU},
\]

其中 \(\mathcal G_t\) 是当前提交 GPU kernel 的阶段集合，\(S_{GPU}\) 是设备物理 SM 总数。所有 \(s_i\) 必须属于设备和后端真实支持的离散集合，并记录请求比例与实际有效 SM 数。对于仅支持 MPS active-thread cap 的配置，该式只表示客户端上限，不得表述为严格独占的 SM 保留。

SM 数量约束不能代替性能可行性。共享配置还必须满足：

- TTS 的 \(RTF^{p95}\)、TTFA 和真实播放 underrun 门槛；
- LLM 的 TTFT、decode throughput 和 token-gap 门槛；
- Diffusion 在 5\%、10\%、20\% slowdown 边界下的可行性；
- 无 OOM、无静默 fallback，且实际显存峰值通过第 6.1 节门槛；
- 所有 GPU kernel 确实通过目标 MPS client 或 Green Context/stream 提交，不能有未记录的 default-stream 逃逸。

将配置分为两层验证：

1. **容量可行**：显存公式和真实共置峰值通过；
2. **执行可行**：并发运行后仍满足交互约束并形成 Pareto 点。

论文需要明确给出一个关键结论：`VRAM-compatible co-residency` 不必然意味着 `execution-compatible co-scheduling`。GPU utilization 的单个平均值也不能替代 SM 曲线与共运行画像。

### 6.4 分层目标

不使用难以解释的单一主观加权和。采用分层约束与 Pareto 比较。

第一层：正确性与安全

- 不发生 OOM；
- KV 和对话状态在切换后保持一致；
- TTS chunk 不丢失、不重复、不乱序；
- 有界队列、拒绝和取消语义正确；
- 真实后端失败不能静默回退成 Mock 后继续记录性能。

第二层：交互约束

- 普通单图请求不主动等待聚批；
- 稳态语音无播放断流；
- 目标 \(RTF_{TTS}^{p95}\le0.8\)，同时报告 \(RTF<1\) 的实际边界；
- 运行中的 Diffusion 不硬中断；
- 未知画像或显存不安全时使用保守回退配置。

第三层：性能优化

1. 最小化语音 TTFA；
2. 最小化第一张图片完成时间；
3. 最大化 LLM decode tokens/s；
4. 降低完整语音回复完成时间；
5. 减少无收益的模型转换和驻留抖动。

Diffusion slowdown 不预先锁死为唯一阈值。正式报告应展示 5%、10%、20% slowdown 约束下的可行配置或 Pareto 前沿。

### 6.5 保守回退

画像缺失、预测显存不足或共享 GPU 配置未经验证时，回退到：

\[
Diffusion_{GPU}+LLM_{CPU}+TTS_{CPU}.
\]

回退路径本身也必须经过真实测试。

- --

## 7. 代价模型

### 7.1 预测对象

代价模型预测已经到达的任务和已知流水线阶段，不为了聚批去等待未来请求。

需要维护：

\[
\hat T_i(r_i,c_i,s_i,z_i),
\]

以及当前工作负载的具体预测器：

\[
\hat T_D(model,resolution,steps,b,s_D,co\_runner),
\]

\[
\hat T_L(r_L,c_L,s_L,K,input,output),
\]

\[
\hat T_T(r_T,c_T,s_T,text\_length,co\_runner),
\]

\[
\hat T_{transition}^{(i)}(r_{src},r_{dst},state_i).
\]

### 7.2 转换代价

\[
C_{transition}=
T_{load}+T_{KVsave}+T_{KVrestore}+T_{swap}+P_{service}.
\]

代价必须与源配置、目标配置和 KV 大小关联，不再只有“全 GPU→全 CPU”的一个常数。

### 7.3 驻留迟滞

为避免反复加载和释放，需要加入迟滞：

- 图像队列非空时不恢复到 S0；
- 刚完成配置转换时，在最小驻留窗口内不因微小预测变化再次切换；
- SM 配置只在新会话、首个 TTS chunk、图像到达/完成、队列清空或后端异常等安全事件点重选，不在每个 token 或短时 utilization 波动后重建；
- 连续语音会话可将一次转换成本摊销到多轮回复；
- 队列清空或会话状态改变时重新评估目标配置；
- 任何迟滞不能给普通单图请求增加主动聚批等待。

### 7.4 规划结果

准入控制不应只返回 CPU/GPU 和一个 `requires_offload` 布尔值，而应返回：

```text
PlacementPlan
  phase_placements[]
    phase_id
    target_device
    residency_profile
    cpu_core_budget
    gpu_compute_backend
    requested_sm_share
    effective_sm_count
    gpu_memory_limit
    batch_or_concurrency
  transitions[]
  predicted_vram
  predicted_latency
  predicted_audio_rtf
  predicted_diffusion_slowdown
  fallback_plan
  decision_reason
```

- --

## 8. CPU 核心分配方法

核心分配不能从 7+7 或 8+6 等对称组合开始猜。

CPU 分核只在两个或以上 CPU-heavy 阶段真实重叠时展开。没有图像任务时，不把 `LLM_CPU + TTS_CPU` 作为主要性能候选：若只有一个重型 CPU 阶段，则直接令

\[
c_i=C_{physical}-R_{sys}.
\]

因此，大规模 LLM/TTS 分核矩阵主要用于 Diffusion 占据 GPU 后的 S1 状态。S0 中只有当最终候选同时包含活跃的 LLM CPU 分区和 TTS CPU 分区时，才进行小范围局部细化。

### 8.1 单模型扩展曲线

先分别测试 LLM 与 TTS：

\[
c\in\{2,4,6,8,10,12,14,16\}.
\]

得到：

- LLM 的 \(Throughput_L(c,r_L)\)；
- TTS 的 \(RTF_T(c,r_T,length)\)；
- 每增加两个物理核心的边际收益；
- 出现带宽、缓存或功耗饱和的位置。

### 8.2 推导 TTS 最低核心预算

\[
c_T^{min}=\min\{c:
RTF_T^{p95}(c)\le0.8
\land Gap_{audio}=0\}.
\]

并发实验只需要从以下邻域开始：

\[
c_T\in\{c_T^{min}-2,c_T^{min},c_T^{min}+2\},
\]

\[
c_L=C_{physical}-R-c_T.
\]

超出合法范围的候选自动删除。7+7 只作为对称基线，而不是默认策略。

### 8.3 拓扑消融

只对最优的两个分配方案比较：

- 线程数限制但不绑核；
- 分离绑核；
- CCD 感知隔离；
- 开启 SMT 后的逻辑线程配置；
- 不保留系统核心；
- 操作系统默认调度。

这样才能区分收益来自减少线程、核心隔离、缓存局部性还是 SMT。

### 8.4 GPU SM 分配与 LLM→TTS 流水线平衡

SM 分配的目标不是复现任务管理器中的平均 GPU utilization，也不是预先固定 TTS 使用 20\%。应首先求满足语音交互门槛的最小 GPU 计算预算：

\[
s_T^{min}=\min\{s:
RTF_T^{p95}(r_T,s)\le0.8
\land Gap_{audio}=0
\}.
\]

若当前后端在完整 GPU 上仍无法达到 \(RTF^{p95}\le0.8\)，则同时报告 \(RTF<1\) 的可持续边界和有限回复通过预缓冲实现零 underrun 时所需的额外 TTFA。不得因为 CPU LLM 产生文本较慢就把 TTS 的持续实时性判为通过。

LLM 与 TTS 共运行时，资源平衡以端到端流水线为准：

\[
TTFA=T_{transition}+T_{LLM}^{first\ sentence}(r_L,c_L,s_L)
+T_{TTS}^{first\ chunk}(r_T,c_T,s_T),
\]

并对每个后续 chunk 检查：

\[
t_{j+1}^{ready}\le t_j^{playback\_end}.
\]

第一版 SM 管理只实现离散静态 profile，不实现 kernel 执行中的任意连续缩放：

```text
GpuShareProfile
  backend: NONE | MPS | GREEN_CONTEXT
  requested_percentage
  effective_sm_count
  gpu_memory_limit_mb
  phase_id
```

最低实现要求：

1. 启动前记录 GPU UUID、CUDA/driver、物理 SM 总数、计算能力、后端和最小分区粒度；
2. MPS 环境变量必须在客户端创建任何 CUDA context 前设置；每个静态配置使用独立子进程；
3. 客户端必须回报实际有效 SM 数，不能只写目标百分比；
4. TTS 与 Forge 退出前执行 CUDA synchronize，异常退出写入结构化失败；
5. GPU 显存限制和 SM 限制分别记录，任何一方都不能被描述成另一方；
6. 真实后端不支持或分区未生效时明确失败，不静默退回普通共享；
7. 第一版不修改生产规划器，只在 `experiments/elastic_residency/` 建立独立能力探针和实验 wrapper。

能力探针至少输出：

```text
GpuPartitionCapability
  backend_available
  total_physical_sms
  minimum_partition_sms
  requested_percentage
  effective_sm_count
  dynamic_reconfiguration_supported
  cuda_context_created_after_configuration
```

推荐的第一轮候选为 `100/75/50/25%`，随后只在 TTS 性能临界点附近补测。正式论文使用实际 SM 数分组，百分比仅用于解释。对于 LLM→TTS，先测不受控共享，再测试少量 \((s_L,s_T)\) 候选；对于 TTS→Diffusion 共运行，先固定 LLM 在 CPU，测试 TTS 的最小可行 \(s_T\)，其余 SM 分配给 Diffusion。

第一版 Go 条件：至少存在一个受控 SM 配置，相对 CPU TTS 或不受控 GPU 共享形成端到端 Pareto 改进，并满足显存、正确性和播放门槛。No-Go 条件：分区后端不可用、实际 SM 数无法验证，或所有配置都在 TTFA、RTF、Diffusion slowdown 和转换成本上被现有方案支配。

- --

## 9. 当前实现差距

截至 2026-08-30，当前代码与目标机制之间至少存在以下差距：

1. `services/phase_aware/phase_model.py`
   - LLM 仍是 GPU_NORMAL / CPU_DEGRADED / RESTORING 三态。
   - TTS 预设默认 CPU，不会动态比较完整流水线。
2. `services/phase_aware/gpu_lease.py`
   - Resident 和 Burst 仍按重任务独占处理。
   - SHARED lease 尚未进入真实执行路径。
3. `services/phase_aware/cost_model.py`
   - burst 时长预测主要用于记录。
   - GPU_REQUIRED 最终仍直接触发完整迁移。
   - benefit 和 queue pressure 阈值没有形成闭环决策。
4. `services/phase_aware/phase_scheduler.py`
   - LLM 剩余工作时间被固定为 0。
   -每个 burst 后立即恢复，不能处理 burst train。
5. `services/cpp_engine_migration.py`
   - Python llama.cpp 路径支持完整 CPU/GPU 实例替换和 KV 保存恢复。
   -尚未提供任意 `n_gpu_layers` 目标档位的统一重新配置接口。
6. C++ 原生 worker
   -真实动态迁移尚未完成，不能将 C++ 绑定可导入等同于迁移可用。
7. TTS
   - 当前源码已包含真实 Qwen3-TTS worker、GPU 句长 profiling、CPU/GPU smoke、确定性断句、有界队列与取消/顺序测试；这些结果仍属于 `experiment_results/` 探索数据。
   - 2026-08-28 的 GPU 句长画像中 45/45 条 RTF 均大于 1；当前后端尚不能支持持续实时播放，不能声称稳态无 underrun。
   - GPU句长 profiling 使用未限制 CPU affinity 的主机基线；还需要补充保留系统核心后的对照。
   - TTS 尚未接入 Phase-Aware，T0–T3 组件放置也尚未实现。
8. N 阶段实验
   - `experiments/phase_aware/exp2_*` 和 `exp3_*` 主要使用 `asyncio.sleep`，只能验证调度语义。
9. 辅助任务与播放环境
   - BGE ONNX 已完成接口 smoke；Faster-Whisper 权重、固定 Python 依赖和 CPU int8 单核真实接口 smoke 均已就绪。3.42 s 参考音频的单轮 RTF 约为 1.44，只能证明链路可用，不能替代 1/2/4 核正式测量。
   - WSL2 已通过 ALSA→WSLg Pulse 桥接打开真实 PortAudio 输出，并实现 Windows WASAPI 回退与显式采样率协商。两条路径的 1 s 静音 smoke 均为 0 underflow；这只解除环境门槛，真实 TTS chunk 与并发负载下的 underrun 仍须正式测量，缓冲时间线不能替代真实播放 callback。
10. GPU SM 管理
   - 当前没有 `GpuShareManager`、MPS/Green Context 能力探针或有效 SM 数验证；
   - TTS、LLM 与 Forge 仍由默认 CUDA 调度器竞争，平均 GPU utilization 不能证明任一阶段只需要对应比例的 SM；
   - WSL2 上的 MPS 可用性尚未通过真实客户端 smoke，不能把文档层面的支持假设写成已实现能力；
   - `PlacementPlan` 尚无 `gpu_compute_backend`、`effective_sm_count` 和 `gpu_memory_limit` 字段；
   - 尚无受控 SM 份额下的 TTS 独立曲线、LLM→TTS 流水线或 TTS→Diffusion 共运行证据。

实现和论文必须诚实反映这些边界。

- --

## 10. 分阶段实施路线

### P0：独立真实 TTS 路径（探索实现已核对）

目标：在 EdgeSched 中接入真实 Qwen3-TTS-12Hz-0.6B-Base，替代真实实验中的 TTS Mock。

最低要求：

- TTS 模型在一个运行阶段内保持常驻；
- 接收确定性句子 chunk；
- 支持 CPU 和 GPU 两种固定启动配置；
- 支持配置 CPU 线程/核心预算；
- 记录每个 chunk 的队列时间、合成时间、音频时长、RTF、设备和资源峰值；
- 为后续真实 `PhaseTask("tts")` 接入保留稳定 adapter 契约；
- 保留 Mock 供单元测试使用；
- 真实后端失败必须明确失败，不能静默记录成 Mock 性能。

P0 暂不要求：

- LLM、TTS、Diffusion 的联合最优调度；
- 动态设备迁移；
- C++ TTS 实现；
- 完整论文图表。

P0 的独立 worker、adapter、smoke、原始 GPU 句长迭代和 manifest 已在当前工作副本核对。它尚未连接 LLM、播放器、ResourceManager 或 Phase-Aware，因此只能进入探索性 TTS profiling，不直接进入正式论文主实验。

### P1：单模型 profiling

分别建立：

1. LLM：`n_gpu_layers × CPU cores × KV length`；
2. TTS：`residency profile × CPU cores × chunk length`，至少包含完整 CPU、完整 GPU，并在后端可行时加入组件级拆分；
3. Diffusion：`resolution × steps × batch size`。

所有结果先写入 `experiment_results/`，保留原始迭代与 manifest。

#### P1-A / 执行阶段 B：TTS 驻留与主机核心对照（B1 smoke，B2 正式矩阵）

首先验证已完成 GPU 句长曲线是否受到主机 affinity 的显著影响：

| 配置 | 主机 CPU 约束 |
|---|---|
| U | 16C/32T，不限制 affinity |
| R2-SMT | 保留 2 个物理核，TTS 可使用 14C/28T |
| R2-NoSMT | 保留 2 个物理核，每个可用物理核只使用一个硬件线程，即 14C/14T |

三种配置各测试短、中、长文本，快速校准至少收集 15 个 chunk。若 R2 与 U 的 first-chunk-ready 或 RTF 差异不超过 5%，原句长结果保留为 `unrestricted_host baseline`；否则使用 R2 重新运行完整句长 profiling。

随后测试 TTS 的显式组件级驻留：

| 配置 | 主要生成组件 | Tokenizer / Codec | Speaker / Prompt 组件 |
|---|---|---|---|
| T0 | GPU | GPU | GPU |
| T1 | GPU | CPU | GPU |
| T2 | GPU | CPU | CPU |
| T3 | GPU | CPU | GPU 构建一次 prompt，缓存后释放 speaker encoder |

这里的 Tokenizer / Codec 指 speech tokenizer/codec；文本 tokenizer 没有 GPU 推理模块，始终留在 CPU。不使用不透明的自动切分作为论文配置。每种组件映射必须明确写入 manifest，并测试约 20、50、100 个中文字符的输入。记录 first-chunk-ready、合成时间、RTF、GPU allocated/reserved/进程总显存、主机 RSS、CPU 利用率和组件间传输时间。

只保留在显存、TTFA 和 RTF 上属于 Pareto 前沿，并且实际能够为其他 GPU 阶段释放有用显存的配置。若组件拆分不能产生这种配置，则将 TTS 定义为 `WHOLE_MIGRATABLE`。

#### P1-B / 执行阶段 C：LLM `n_gpu_layers` 驻留曲线（C1 smoke，C2 正式曲线）

当前 GGUF 元数据已核验 `qwen35.block_count=32`，即 \(N=32\)。固定 \(R_{sys}=2\) 和 \(c_L=14\)，进行粗粒度扫描：

\[
k\in\{0,\lceil0.25N\rceil,\lceil0.5N\rceil,
\lceil0.75N\rceil,N,FULL\}.
\]

当前模型展开为 `k={0,8,16,24,32,FULL}`。`FULL` 与 `k=32` 分开保留，因为 llama.cpp 的完整 offload 还可能包含非 block 张量。

每个配置测试约 512、2048 和 4096 tokens 的上下文，固定生成 128 tokens，独立预热并至少重复 5 次。记录加载/重新配置时间、GPU显存、主机内存、TTFT、prefill tokens/s、decode tokens/s、token gap、CPU利用率和KV保存恢复时间。

根据真实画像求两个临界档位：

\[
H_{TTS}=\max k:\ M_L(k)+M_T+G_{safe}\le V_{GPU},
\]

\[
H_D=\max k:\ M_L(k)+M_D(b=1)+G_{safe}\le V_{GPU}.
\]

在每个临界点附近补测 \(k\pm2\) 和 \(k\pm4\)。第一版使用 llama.cpp 的连续层 offload，不自行设计注意力层、MLP 层或单个张量的任意拆分。

已有 Forge batch=1 的约 5904 MiB 峰值只能用于预估。计算正式 \(H_D\) 前必须在同一代码、驱动、采样器和空闲基线下复测 Forge，不能把历史整卡峰值与新的进程显存口径直接相加。

**C2 旧 CLI 完成记录（2026-08-29）**：`c2_llm_residency_curve_20260829_verified_v3` 基于干净 commit `63e75ba` 完成 `6 配置 × 3 上下文 × 5 轮 = 90/90` 条真实驻留曲线，0 failure、0 fallback，有效性门与 30 ms 性能目标全部通过。该批次现位于 `_archive/superseded/c2/`，只作工程基线；当前性能真源是 C2/C3 direct NVML v2。完整目录映射见 `docs/ELASTIC_RESIDENCY_RESULT_CATALOG_20260830.md`。

#### P1-C：GPU SM 能力探针与单模型计算份额曲线

目标：确认当前 RTX 5070 Laptop + WSL2 + CUDA/PyTorch 环境是否能够真实限制并验证不同客户端的 GPU 计算份额，并得到 TTS、LLM 和 Diffusion 的离散 SM 性能曲线。

第一版优先使用 MPS 独立进程 profile；只有 MPS 在当前 WSL2 环境不可用或无法验证实际有效 SM 数时，才评估 Green Context。不得同时实现两套生产后端。

**SM-A 正式结果（2026-08-30）**：`sm_a_20260830_direct_nvml_v1` 基于干净 commit
`ca7a7ff0832342fc76f261b6b7d8d720b5a116fa` 在 WSL2 真实 CUDA 环境完成。
普通 CUDA 基线成功，RTX 5070 Laptop 报告 36 个物理 SM，direct NVML、Git provenance、
0 failure、0 fallback 与 0 cleanup failure 均满足；但环境中不存在
`nvidia-cuda-mps-control`，100/75/50/25% 四个配置均为 `UNSUPPORTED`，最终
`verdict=NO_GO`。因此当前停止 SM-B/SM-C，不宣称受控 SM 分配，不修改在线规划器；
Green Context 只保留只读能力探测结果，不自动转为生产后端。

建议新增：

```text
experiments/elastic_residency/
  sm_capability.py
  mps_runtime.py
  tts_sm_curve.py
  llm_sm_curve.py
  diffusion_sm_curve.py
  tts_diffusion_sm_colocation.py
tests/elastic_residency/
  test_sm_capability.py
  test_sm_profiles.py
  test_sm_plan_validation.py
```

执行顺序：

1. `SM-A`：能力 preflight，确认 daemon/client、SM 总数、请求比例、实际有效 SM 数、显存限制和退出清理；
2. `SM-B`：TTS T0/T3 的 `100/75/50/25%` 隔离曲线，之后只在性能临界点附近细化；
3. `SM-C`：Diffusion b=1 的对应曲线；
4. `SM-D`：LLM FULL/H 候选的必要曲线，仅在 S0 最终候选确实需要 GPU 计算共享时展开；
5. `SM-E`：先测 `LLM_CPU + TTS_GPU + Diffusion_GPU`，再决定是否测试三阶段 GPU 共享。

所有 profile 均先写入探索目录。SM-A 未通过时，后续结果必须标记为普通 CUDA 不受控共享，不能写成受控 SM 分配。

### P2：静态共存可行性

不修改在线规划器，先人工固定并测试 S0 与 S1 的候选配置。

#### P2-A：S0 无图像静态放置

候选包括：

1. `LLM_FULL_GPU + TTS_CPU`；
2. `LLM_CPU + TTS_FULL_GPU`；
3. `LLM_H_TTS + TTS_FULL_GPU`；
4. `LLM_H_k + TTS_H_COMP`，仅在 P1-A 证明 TTS 组件拆分有效时加入；
5. `LLM_FULL_GPU + TTS_FULL_GPU`，仅在显存安全时加入。

若两个阶段均在 GPU 上执行，每个驻留候选还需比较：

1. 默认 CUDA 不受控共享；
2. 静态 SM 分配，例如由单模型曲线推导的 `75/25` 或相邻有效 SM 档位；
3. 串行执行但保持显存共驻留，用于分离“减少加载成本”和“同时计算”的收益。

`LLM_CPU + TTS_CPU` 只作为故障回退，不进入 S0 主性能候选。实验必须让 LLM 生成后续文本与 TTS 合成前一段音频真实重叠，并记录两个阶段同时处于 running 的交集时长；交集为零的轮次不能计为流水线重叠。场景覆盖单段回复、3个 TTS chunk 的中等回复、6个以上 chunk 的长回复和连续 5 轮语音会话。

记录第一 token、第一完整句、TTFA、音频 gap/underrun、完整回复时间、LLM tokens/s、TTS RTF、显存、内存和 CPU 峰值。若最终候选中 LLM 与 TTS 均存在活跃 CPU 分区，才补测：

\[
c_T^{component}\in\{1,2,4\},\qquad
c_L=14-c_T^{component}.
\]

这只是 S0 局部细化，不展开完整等分核矩阵。

Go 条件：至少存在一个 `PARTITIONABLE` 阶段的混合驻留配置，在不 OOM 的前提下，相对完整 CPU 降级或完整设备切换形成真实 Pareto 改进。

No-Go 条件：

- 部分 GPU 驻留没有提高 LLM/TTS 流水线；
- 或 Diffusion slowdown、显存风险和转换开销抵消全部收益。

若触发 No-Go，停止弹性驻留开发，保留二元迁移并将其解释为实测后的设计结果。

### P3：弹性驻留接口

调度接口不应硬编码为只操作 LLM。将：

```text
offload_llm_to_cpu()
restore_llm_to_gpu()
```

泛化为：

```text
reconfigure_residency(phase_id, target_profile, transition_mode)
```

各后端通过 adapter 实现该契约。LLM 第一版通过新实例加载、KV 保存恢复和实例替换实现；TTS 可以通过显式组件放置或完整实例切换实现。任何后端均不宣称尚未实现的原地显存分页。

### P4：联合规划器

- 使用真实画像生成 `PlacementPlan`；
- 加入显存约束、CPU 核心预算、GPU SM 预算、语音实时约束和迟滞；
- 图片队列非空时避免重复迁移；
- 只允许经过 profiling 且实际有效 SM 数可验证的共享 GPU 配置；
- 先选择驻留可行配置，再在其内部选择计算份额；不得以 SM 约束替代显存准入；
- 提供保守回退计划和可解释决策日志。

### P5：C++ 契约同步

Python 控制面先完成和验证策略。

如果论文继续声称 C++ 承担核心弹性迁移，则必须同步：

- 驻留状态；
- `PlacementPlan`；
- 显存配额；
- GPU SM 预算及实际分区后端；
- CPU 核心预算；
- pybind11 绑定；
- Python/C++ 契约测试。

否则论文必须明确：C++ 负责原生执行，弹性驻留和迁移决策由 Python 控制面完成。

- --

## 11. 实验阶段划分

### 11.1 探索与校准

探索阶段用于：

- 找到可行驻留档位；
- 确定核心扩展曲线；
- 估计迁移、加载和 KV 代价；
- 校准安全余量、EMA、迟滞和候选配置；
- 排除没有收益的配置。

探索数据可以作为画像图或特性分析，但不能同时用于挑参数和证明主策略优于所有基线。

### 11.2 冻结

正式评测前必须冻结：

- 代码版本和提交哈希；
- 模型文件和哈希；
- 驻留档位；
- 核心候选；
- 安全余量；
- EMA、迟滞和决策规则；
- prompts、TTS 文本、声音配置和随机种子；
- 实验脚本和统计口径。

### 11.3 正式实验

正式实验使用新运行数据和预先冻结的配置。流程为：

\[
Implementation \rightarrow Calibration \rightarrow Freeze
\rightarrow Formal\ Evaluation \rightarrow Paper\ Update.
\]

- --

## 12. 正式实验矩阵

### E0：正确性与真实后端确认

目的：证明实验没有静默回退到 Mock。

检查：

- 真实 Qwen3.5 GGUF 加载；
- 真实 Qwen3-TTS Base 输出；
- 真实 Forge 生成；
- KV 保存恢复；
- 设备与资源采样；
- SM 分区后端、请求比例和实际有效 SM 数验证；
- 取消、队列和失败闭合。

### E1：单模型画像

#### E1-L：LLM

- 驻留档位；
- CPU 核心数；
- 不同 KV / context 长度；
- TTFT、prefill、decode、token gap；
- 显存、内存和转换代价。

#### E1-T：TTS

- 完整 CPU、完整 GPU，以及后端支持时的组件级驻留配置；
- 每个组件配置的明确 device map 与主机/GPU 内存占用；
- 不同物理核心预算；
- 短、中、长句子 chunk；
- TTFA、RTF、资源峰值。

#### E1-D：Diffusion

- 标准 b=1 请求；
- 同 prompt 的 b=2、b=4；
- 不同分辨率与 steps；
- 冷加载和稳态执行；
- 显存时间线。

#### E1-G：GPU 计算份额

- MPS/Green Context 能力 preflight；
- TTS T0/T3 在离散有效 SM 档位下的 TTFA、RTF、显存和输出正确性；
- Diffusion b=1 在离散有效 SM 档位下的延迟、显存和利用率；
- LLM 仅对进入 S0 共享候选的驻留档位补测 SM 曲线；
- 请求百分比、实际有效 SM 数、后端、客户端 CUDA context 创建顺序和异常清理；
- 不受控共享作为必需基线。

### E2：无图像的 LLM→TTS 流水线

比较 S0-A、S0-B、S0-C(k)，S0-D 仅在可行时加入。对同时在 GPU 执行的 LLM/TTS 配置，额外比较不受控共享、显存共驻留但计算串行，以及由 E1-G 推导的静态 SM 分配。

场景：

- 单轮短回复；
- 单轮长回复；
- 连续多轮语音会话。

指标：

- 第一段文本时间；
- TTFA；
- 音频 gap / underrun；
- 完整语音回复时间；
- LLM tokens/s；
- 请求/实际有效 SM 数与 SM 配置切换成本；
- 转换成本及其跨轮摊销。

### E3：Diffusion 期间的语音流水线

比较 S1-A、S1-B(k)、S1-C，S1-D(k) 仅在可行时加入。S1-C/S1-D 必须分别测试显存共驻留但计算串行、不受控并发和受控 SM 分配，不能把三种模式混成一个“共享 GPU”配置。

至少包含：

1. 普通单图请求；
2. 明确标记为压力测试的持续 GPU 占用场景。

指标：

- 图片资源准备和第一张图片时间；
- Diffusion slowdown；
- LLM TTFT、tokens/s 和 token gap；
- TTS TTFA、RTF 和音频断流；
- TTS/Diffusion 的请求与实际有效 SM 数；
- 不受控共享相对受控 SM 分配的尾延迟差异；
- 峰值显存、内存和 CPU 利用率。

### E4：CPU 核心与 GPU SM 分配

候选必须由 E1 推导，不能预先穷举等分组合。

步骤：

1. 求 \(c_T^{min}\)；
2. 测试其邻域；
3. 将剩余核心分配给 LLM；
4. 对最优两个候选做绑核、CCD、SMT 和系统保留核心消融。

随后对需要 GPU 共享的流水线：

1. 求满足 TTS 门槛的最小 \(s_T^{min}\)；
2. 将剩余 SM 分配给 LLM 或 Diffusion；
3. 只在临界点邻域补测，不穷举所有 CPU 核 × SM × 驻留档位组合；
4. 比较 CPU 分核、GPU 分 SM 以及二者联合启用，隔离每个维度的贡献。

### E5：代价感知策略对比

主基线：

| 策略 | 含义 |
|---|---|
| Serial/Wait | 保持当前驻留，竞争任务等待 |
| Always Full Migration | 当前完整 LLM 迁出/恢复策略 |
| Static Best | 使用校准阶段得到的固定最优配置 |
| Uncontrolled GPU Co-run | 显存允许时直接并发，不限制 SM |
| Static SM Split | 使用离线标定得到的固定 SM 份额 |
| Cost-Aware Joint Planner | 联合选择驻留、CPU 核、SM、batch 和转换方式 |
| Oracle | 使用真实未来执行时长进行事后最优选择 |

迁移机制基线可保留：

- Cold Restart；
- Symmetric Migration；
- Urgent Down + Atomic Hot-Swap Up。

这些基线回答的是状态转换机制问题，不能替代新版在线规划策略基线。

### E6：图片队列与多候选图

分别测试：

- 单张普通图片；
- 同一 prompt 多候选图；
- 不同 prompt 连续请求；
- 图片队列非空时是否避免重复恢复；
- 队列满时的拒绝和取消。

### E7：真实 N 阶段工作流

至少使用：

\[
ASR/RAG\rightarrow LLM\rightarrow TTS+Image.
\]

ASR/RAG 可以轻量，但必须是真实执行。Mock N-phase 实验保留为调度语义测试，不作为真实性能结果。

### E8：消融与稳健性

消融：

- 无时长预测；
- 禁用 `PARTITIONABLE`，将相关阶段退化为完整设备配置；
- 无CPU核心隔离；
- 无 SM 分配，退化为默认 CUDA 不受控共享；
- 固定 SM 分配，不根据工作流状态或画像调整；
- 无驻留迟滞；
- 无KV保存；
- 固定设备放置。

敏感性：

- KV / context 长度；
- TTS chunk 长度；
- Diffusion 时长；
- 图像队列深度；
- 预测误差；
- 显存安全余量。
- SM 分区粒度与 TTS 最小有效 SM 数；
- GPU utilization 或带宽压力预测误差。

### E9：控制面与可靠性补充

以下内容继续保留，但不作为新版核心创新主图：

- 状态背板与崩溃闭合；
- 取消语义；
- 有界队列与 Mock 压力测试；
- Python 原语/信号量控制面开销。

- --

## 13. 统一指标

### 13.1 LLM

- TTFT；
- prefill tokens/s；
- decode tokens/s；
- 最大和 p95 token gap；
- 迁移窗口内最大请求延迟；
- 上下文一致性。

### 13.2 TTS

- 第一段音频 TTFA；
- queue wait；
- chunk synthesis time；
- 音频时长；
- RTF mean / p95；
- gap / underrun count；
- 完整语音回复时间。

### 13.3 Image

- 准入到资源准备开始的时间；
- 迁移/释放完成时间；
- 第一张图片完成时间；
- 任务执行时间；
- 队列等待；
- batch throughput；
- Diffusion slowdown。

### 13.4 系统

- 峰值 GPU VRAM；
- GPU 物理 SM 总数、请求比例、实际有效 SM 数与分区后端；
- GPU utilization、功耗、温度，以及可获得时的显存带宽压力代理；
- 峰值 RSS 和 WSL 可用内存；
- CPU 利用率和亲和配置；
- 迁移次数、方向和时间；
- 驻留档位变化；
- OOM、失败和回退次数；
- 调度决策开销；
- SM profile 创建/销毁或 worker 重启成本；
- 预测值与真实值误差。

“Restore Gap=0”只能在采样分辨率和指标定义明确时使用。不得将异步预加载时间等同于用户完全无感知。

- --

## 14. 统计与复现要求

### 14.1 运行次数

- 每个配置独立预热。
- 每个正式配置至少 5 次完整独立运行。
- 用于 p95 的事件或 chunk 至少 30 个，优先 50 个以上。
- 少量样本只报告原始值、均值和范围，不滥用 p99。

### 14.2 顺序控制

- 使用随机顺序或正反交替顺序降低温度漂移偏差。
- 冷启动与稳态实验分开。
- 不同 batch shape 分别预热。
- 在运行前后记录温度、功耗或至少保留足够冷却时间。

### 14.3 固定输入

正式实验固定并记录：

- prompts；
- 输出 token 上限；
- TTS 文本和分句结果；
- 声音配置；
- 图像 seed、分辨率、steps、sampler、CFG；
- context/KV 长度；
- 核心掩码与线程变量。

### 14.4 证据链

每个正文数字必须能够定位到：

```text
paper figure/table
  -> aggregated JSON
  -> raw iterations
  -> experiment script
  -> manifest
  -> code/model/environment hash
```

manifest 至少记录：

- 是否真实 LLM/TTS/Forge；
- 模型文件和哈希；
- CUDA、驱动和主要库版本；
- WSL2 内存与 swap；
- CPU 核心和亲和配置；
- SM 分区后端、请求份额、实际有效 SM 数、MPS server/client 状态与 GPU UUID；
- 运行参数与随机种子；
- 代码提交哈希；
- 是否发生 fallback。

- --

## 15. 现有数据的使用边界

### 15.1 可以保留为历史或辅助证据

- `paper/experiment_data/01_paper_main/migration/` 中的旧迁移结果；
- Cold-Start vs Hot-Swap；
- 崩溃恢复和状态背板；
- 取消语义；
- Mock 控制面压力测试。

这些结果来自旧模型、旧策略或不同实验口径。它们不能直接证明 Qwen3.5 + Qwen3-TTS + 弹性驻留的性能。

### 15.2 当前探索性结果

- `experiment_results/batch_scaling/batch_scaling_20260828_130742.json`
  - 真实 Forge；
  -同一 prompt；
  - b=1、2、4、8；
  -只对 b=1 做了统一预热；
  -可用于设计下一轮实验，尚未晋升为正文证据。
- `experiment_results/streaming_pipeline/smoke_tts_gpu.wav`
  -只能证明一次 GPU TTS 输出存在；
  -没有对应可复现脚本和 manifest，不能引用性能。
- `experiment_results/tts/profiling/gpu_sentence_length/gpu_sentence_length_xvector_r3_20260828/`
  -真实 Qwen3-TTS Base GPU、x-vector 音色消融；
  -包含 45 条原始迭代、manifest、汇总、拟合和图表；
  -45/45 条成功且顺序正确，但 45/45 条 RTF 都大于 1；
  -只证明当前后端的探索性句长画像，尚未证明 LLM→TTS 重叠或真实音频播放无 underrun。
- `experiment_results/elastic_residency/c2/c2_llm_residency_curve_20260829_direct_nvml_v2/`
  - 真实 llama.cpp/Qwen3.5-9B 粗驻留曲线，`6 配置 × 3 上下文 × 5 轮 = 90/90` 条全部通过，0 failure、0 fallback；
  - 干净 commit `17c40d3495cb01237bdf2e7429f2635362e908cf`，GPU 后端为 direct NVML，未启动 `nvidia-smi` CLI；
  - 与 C3 direct v2 合并后构成 16 档完整驻留曲线，旧 `verified_v3` 仅保留为 CLI 工程基线。
- `experiment_results/elastic_residency/c3/c3_llm_critical_refinement_20260829_direct_nvml_v2/`
  - 真实临界细化曲线，`10 配置 × 3 上下文 × 5 轮 = 150/150` 条全部通过，0 failure；
  - 与 C2 使用同一模型哈希、commit、direct NVML 和 `context+128` 状态口径；
  - K32 与 FULL 整卡增量只差约 140 MiB但 decode 差异明显，驻留收益不能按层数线性插值。
- `experiment_results/elastic_residency/kv_calibration/c3_kv_cache_calibration_20260830_direct_nvml_v1/`
  - 独立 KV 容量校准，`16 配置 × 3 上下文 × 1 轮 = 48/48` 条全部通过，0 failure；
  - 干净 commit `820ee0fbcbbf671a791d31786bf171bf2a241d1a`，模型哈希与 C2/C3 一致，全部使用 direct NVML；
  - `n_ctx/KV tokens=640/2176/4224` 时，预分配 KV buffer 总量为 `24/72/136 MiB`；FULL 的 GPU KV 部分分别为 `24/72/136 MiB`；
  - 完整 `save_state()` 序列化状态为 `70.263/118.292/182.331 MiB`，不是纯 GPU KV 显存，也不是此前误写的 1--2 GiB；
  - 该数据补齐容量语义，不改变 C2/C3 的性能计时结果；正式阈值仍须等待 TTS/Forge 同后端校准与真实共置残差。
- `experiment_results/elastic_residency/c3_analysis/c3_h_screen_20260829_legacy_forge.json`
  - 基于 C2 v3 + B2 v1 + 旧 Forge 峰值的 `G_safe`/`H_TTS`/`H_D` 容量筛选（`analyze_results.py` 产物）；
  - `G_safe=501.3 MiB`（allocator slack 主导，colocated 残差 pending）；`H_TTS(T0)=16`、`H_TTS(T3)=32 且 FULL 可行`、`H_D=0`（三上下文一致）；
  - Forge 使用 2026-08-28 旧峰值 legacy 估算，仅用于排程，不得用于冻结或论文结论。
- `experiment_results/elastic_residency/forge/forge_b1_reprofile_20260829_verified_v2/`
  - Forge b=1 同日复测（10 项门禁全通过，swap 增长 0，干净 commit `50d2362`）；
  - 空闲高水位 415 MiB、常驻基线 419 MiB、5 轮峰值稳定 5667 MiB、b=1 延迟 mean 4.844 s；
  - **`M_D = 5252 MiB`**，比旧估算低约 338 MiB。
- `experiment_results/elastic_residency/c3_analysis/c3_h_reprofiled_20260829.json`
  - 复测后正式容量筛选：`H_TTS(T0)=16`、`H_TTS(T3)=32 且 FULL 可行`（三上下文一致）；
  - **`H_D` 按上下文分化**：512 下为 8，2048/4096 下为 0——按 context 分组保留，不合并成常数；
  - 仍是容量筛选（colocated 残差 pending），最终可行性待真实共置验证。
- `experiment_results/phase_aware/exp2_*`、`exp3_*`
  -使用 Mock/sleep；
  -只能验证流程和统计字段。

### 15.3 必须重跑

换用 Qwen3.5-9B-Q4_K_M 后，下列性能需在冻结环境重跑：

- 完整迁出与恢复；
- 真实工作流；
- T0/T3 与 Forge 参与容量阈值的整卡数据（统一为 direct NVML）；
- 真实共置残差、无图像 LLM→TTS 重叠和后续正式弹性策略评测。

- --

## 16. 论文结构建议

### 16.1 Introduction

围绕以下问题展开：

1. 本地助手已经形成 LLM、语音和图像流水线。
2. 单个 8GB GPU 无法让所有模型以最高性能长期驻留。
3. 整体迁出虽然安全，但忽略了剩余显存、CPU 核心、GPU SM 和流水线阶段之间的互补性。
4. 仅做 GPU SM 分配又假设所有活跃模型已经能够共驻留，无法解决 8GB 设备上的容量冲突。
5. EdgeSched 使用弹性驻留、CPU/GPU 计算份额、代价预测和状态感知规划进行联合调度。

贡献应正面陈述，不再反复写“priority queue、offloading 本身并不新”。

### 16.2 Background / Motivation

需要用画像数据展示：

- 不同模型的显存需求；
- CPU/GPU 性能差异；
- 不同 SM 份额下的非线性性能曲线及显存带宽干扰；
- TTS实时约束；
- 整体迁移成本；
- 为什么固定放置不能覆盖 S0、S1、S2。

### 16.3 System Design

建议结构：

1. workflow 和 phase model；
2. 设备亲和性、驻留能力与 GPU 计算份额能力的正交任务抽象；
3. 配置相关任务画像；
4. 弹性驻留状态；
5. 显存容量与 GPU 执行兼容性的两级可行配置过滤；
6. 联合驻留、CPU 核、GPU SM 和 batch 的代价感知 PlacementPlan；
7. 非对称状态转换；
8. 队列、迟滞与保守回退。

### 16.4 Implementation

明确区分：

- Python 控制面；
- llama.cpp LLM 执行；
- Qwen3-TTS worker；
- Forge 图像服务；
- MPS/Green Context 能力探针、静态 SM profile 与实际有效 SM 验证；
- C++ 原生执行面目前实际承担的职责。

不能把 Python 迁移实现写成 C++ 已完成能力。

### 16.5 Evaluation

建议主线：

1. 单模型画像与动机；
2. CPU 核与 GPU SM 的资源—性能曲线；
3. 无图像时的语音流水线；
4. 图像期间的联合调度；
5. 弹性策略对比；
6. 多任务工作流与消融；
7. 控制面和可靠性补充。

### 16.6 Related Work 与创新边界

提交前必须优先引用已有正式会议版本；只有尚无正式版本的近期工作使用 arXiv preprint，并在参考文献中明确标注版本与日期。截至 2026-08-30，至少需要覆盖：

- GSLICE（SoCC 2020）：MPS 控制的 GPU 空间共享；
- GPUlets（USENIX ATC 2022）：面向多模型推理的时空 GPU 共享与资源画像；
- Miriam（SenSys 2023）：边缘 GPU 上基于弹性 kernel 的多 DNN 实时推理；
- Chrion（EMSOFT 2023）：CPU 核与 GPU 协同执行 RNN 推理；
- VoxServe（2026 arXiv）：流式语音模型的 TTFA 与后续音频可播放性调度；
- Fish Audio S2（2026 arXiv）：通过 MPS 共运行 LLM decode 与 vocoder；
- vLLM-Omni、StreamWise（2026 arXiv）：多阶段/多模态生成服务；
- HorizonServe（2026-08-03 arXiv）：单 GPU Omni 模型的时间共享、SM 空间共享和带宽感知分配。

论文不得将以下内容单独声称为新贡献：MPS/Green Context 本身、按比例分 SM、LLM 与语音解码阶段并发、TTFA/RTF 指标、CPU/GPU 协同或模型 offload。创新边界应表述为：

> 现有空间共享系统主要决定已经能够共驻留的阶段如何分享 GPU；EdgeSched 面向所有模型无法完整共驻留的 8GB 级设备，先决定独立模型及其组件在哪里驻留，再联合决定 CPU 物理核、GPU SM、batch 与跨层转换方式。

相关工作对比表至少包含：目标硬件、模型是否完整共驻留、是否支持 CPU 模型计算、是否支持部分驻留、是否控制 SM、是否保持有状态流式会话，以及主要优化指标。

- --

## 17. 图表规划

### 主图

1. **资源流与控制流架构图**
   -显示 LLM、TTS、Diffusion、DDR、GPU、队列和规划器；
   -明确区分 `DeviceAffinity`、`ResidencyCapability` 与 GPU 计算份额能力，避免把部分卸载或 SM 分配画成特定模型特例；
   -在 GPU 池中显示显存配额与有效 SM 预算是两个独立维度；
   -区分数据流、控制流和状态转换；
   -不再只画软件工程分层框。
2. **驻留状态与转换图**
   - CPU、Hybrid、Full GPU；
   - S0、S1、S2；
   - Urgent 与 Atomic 转换。
3. **单模型资源—性能曲线**
   - LLM GPU layers；
   - TTS cores/RTF；
   - TTS 与 Diffusion 的 SM/性能曲线；
   - Diffusion batch/VRAM。
4. **LLM→TTS 流水线时间线**
   -文本 chunk；
   - TTS chunk；
   -播放与 gap。
5. **Diffusion 期间联合调度时间线**
   -迁移、图像、LLM token、TTS音频。
6. **策略 Pareto 图**
   - TTFA、第一张图、LLM吞吐、Diffusion slowdown；
   - 区分不受控 GPU 共运行、静态 SM 分配与联合规划器。

### 主表

1. 工作负载的设备亲和性、驻留能力、GPU 计算份额能力与可选档位；
2.系统状态、候选计划与回退；
3.主策略和基线能力对比；
4.端到端结果；
5.消融结果。

原有信息量较低的组件对比表不再作为主要贡献表。

- --

## 18. 有效性与局限性

### 18.1 内部有效性

- 调参与正式评测使用不同运行数据。
- 配置顺序随机化或交替。
- 所有策略使用同一模型、输入和后端。
- 显存与时间采样频率一致。
- 明确区分冷加载、稳态和转换窗口。

### 18.2 外部有效性

- 主结果来自一台消费级 NVIDIA 笔记本，不能直接推广到所有 CUDA GPU。
- 不同 GPU 架构、内存带宽、功耗限制和驱动可能改变最优配置。
- 结果证明方法和决策流程，而不是宣称固定核心数适用于所有设备。

### 18.3 当前机制边界

- 不实现通用显存分页。
- `PARTITIONABLE` 只表示调度抽象能够描述部分驻留，不代表任意后端或任意拆分粒度已经实现。
- SM 分配不等于逐个 CUDA core 管理，也不等于显存分页；第一版只使用后端支持的离散静态 profile。
- MPS active-thread percentage 是客户端执行资源上限时，不得表述为严格独占的 SM 保留。
- Green Context 或 MPS 在当前 WSL2 环境未经 capability smoke 前，不进入论文实现主张。
- 不硬中断正在运行的 Diffusion kernel。
- 第一版 TTS 以句子/chunk 级流式为主。
- 部分驻留通过实例重新配置实现，不是原地层迁移。
- 如果共享 GPU 无 Pareto 收益，系统允许退化为经过实验证明的独占策略。

- --

## 19. Go / No-Go 检查点

### Gate A：真实 TTS

Go：

- CPU/GPU 路径可重复运行；
- 模型保持常驻；
- chunk 指标完整；
- 无静默 Mock fallback。

### Gate B：混合驻留

Go：

- 至少一个声明为 `PARTITIONABLE` 的阶段存在可稳定运行且可复现的离散混合配置；
- 该配置的拆分粒度、组件/层映射和转换方式均写入 manifest；
- 相对全 CPU 降级存在流水线收益；
- 没有不可接受的显存风险。

No-Go：

- 部分驻留只增加转换和干扰，没有改善端到端指标。

### Gate C：共享 GPU

Go：

- 能力探针能够验证请求比例与实际有效 SM 数，且无静默普通共享 fallback；
- 至少一个 LLM→TTS 或 TTS→Diffusion 配置在受控 SM 份额下形成可解释的 Pareto 点；
- 容量可行和执行可行均通过，真实播放无 underrun；
- 受控分配相对 CPU 回退或默认 CUDA 不受控共享至少改善一个主要端到端指标，且没有突破冻结的其他交互约束。

No-Go：

- MPS/Green Context 在目标 WSL2 环境不可用或实际有效 SM 数无法验证；
- 共存导致明显 OOM 风险；
- 所有受控 SM 配置都被 CPU 回退、不受控共享或显存共驻留但计算串行方案支配。

### Gate D：正式实验

进入正式实验前必须满足：

- 策略已冻结；
- 所有真实后端通过 E0；
- calibration 与 evaluation 数据分离；
- manifest 完整；
- 正文中计划使用的每个指标均能实际采集。

- --

## 20. 当前任务顺序

真实 TTS worker 和基础 smoke 已经完成，但其运行结果仍处于 `experiment_results/` 的探索阶段。当前与归档批次的边界见 `docs/ELASTIC_RESIDENCY_RESULT_CATALOG_20260830.md`，近期可执行步骤以 `docs/ELASTIC_RESIDENCY_EXPERIMENT_EXECUTION_PLAN_20260828.md` 为准：

- [x] 真实 Qwen3-TTS-0.6B-Base worker 与基础可复现 smoke
- [x] A1：`a1_smoke_20260829_verified_v7` 已在干净 commit `ae68608` 上通过 GPU/jitter/CPU 首样本/provenance 全部 17 项硬门槛；v6 仅保留为旧门槛调试记录
- [x] B1：`b1_tts_components_20260829_verified_v2` 已在真实 Qwen3-TTS/CUDA 后端完成 T0–T3 组件发现与单文本 smoke，4/4 配置通过；该冷态单轮数据不用于性能结论
- [x] B2：`b2_tts_matrix_20260829_verified_v1` 已完成 60/60 条；T0/T3 为条件式 Pareto，T1/T2 被支配，且 60 条 RTF 全部大于 1
- [x] C1：`c1_llm_residency_512_20260829_verified_v1` 已完成 K0/FULL 的 512-token 真实端点 smoke，两端均通过 19/19 项门禁；它是单轮基础设施证据，不作为正式性能曲线
- [x] C2 direct NVML：`c2_llm_residency_curve_20260829_direct_nvml_v2` 已完成 90/90 条粗驻留曲线，0 failure；旧 CLI 曲线只保留为工程基线
- [x] C3 direct NVML 临界补测：`c3_llm_critical_refinement_20260829_direct_nvml_v2` 已完成 150/150 条；与 C2 合并后覆盖 16 个驻留档位
- [x] C3 KV 校准：`c3_kv_cache_calibration_20260830_direct_nvml_v1` 已完成 48/48 条，冻结 GPU/宿主 KV buffer 与完整序列化状态的独立语义
- [ ] **当前 C3 收尾（还剩 4 个动作）**：
  1. T0/T3 的 direct NVML 隔离矩阵（30 条）；
  2. Forge b=1 的 direct NVML 同后端校准；
  3. 用完整 16 档曲线重新筛选最小共置候选；
  4. 共置 smoke + 每候选 5 轮，加入 colocated 残差后冻结 `G_safe/H_TTS/H_D`
- [x] **SM-A：GPU 计算份额能力探针**：真实 WSL 记录 `sm_a_20260830_direct_nvml_v1` 已完成；MPS binary 不存在，结论 NO_GO
- [x] **SM-B：TTS T0/T3 隔离 SM 曲线**：因 SM-A NO_GO 按门禁停止，不运行、不伪造受控 SM 曲线
- [x] **SM-C：Diffusion b=1 隔离 SM 曲线**：因 SM-A NO_GO 按门禁停止，不运行、不伪造受控 SM 曲线
- [ ] **SM-D：LLM→TTS 平衡实验**：比较不受控共享、显存共驻留但计算串行、静态 SM 分配，并记录第一句、TTFA、chunk deadline 与真实 underrun
- [ ] **SM-E：TTS→Diffusion 共运行实验**：先固定 LLM 在 CPU，求 TTS 最小可行 SM 数，将剩余 SM 给 Diffusion；前序形成 Pareto 后才测试三阶段 GPU 共享
- [ ] JSON RAG/BGE/Faster-Whisper 核心预算；音频路由就绪后补真实播放探针
- [ ] `R_sys={0,1,2,4}` 正式辅助任务验证
- [ ] S0 无图像真实重叠静态放置实验
- [ ] 根据真实结果判断弹性驻留 Go / No-Go
- [ ] Go 后才实现 `PlacementPlan` 与新代价模型，再开展 S1 和正式策略评测
- [ ] 同步论文、图表和必要的 C++ 契约

B1/B2 已确认 T0/T3 是条件式 Pareto；C2+C3 direct NVML 已完成 16 档性能曲线，独立 KV 校准也已完成。SM-A 已在真实 WSL 中因 MPS binary 不存在判定 NO_GO，SM-B/SM-C 按门禁停止。当前继续完成 T0/T3、Forge 的同后端整卡校准和真实共置残差，冻结 `G_safe/H_TTS/H_D`；不修改在线规划器或宣称受控 SM 分配。

- --

## 21. 完成标准

新版研究工作完成时，应满足：

- 真实 LLM、TTS、Diffusion 能形成可复现端到端工作流；
- 调度器能够输出完整、可解释的 PlacementPlan；
- PlacementPlan 同时包含驻留档位、CPU 物理核预算、GPU 有效 SM 数、显存上限、batch 和转换方式；
- 所有允许的共享配置都经过真实 profiling；
- 异常和未知状态能安全回退；
- 不同 prompt 不会错误合批；
- 图片队列不会触发无意义的重复迁移；
- 核心数由测量曲线推导，而不是固定写死；
- SM 数由能力探针和性能曲线推导，而不是根据平均 GPU utilization 猜测百分比；
- LLM→TTS 的后续 chunk 能在播放截止时间前准备完成，或论文明确报告当前后端无法满足持续流式条件；
- 至少一个场景中，新策略相对二元完整迁移形成真实 Pareto 改进；
- 若不存在该改进，论文主动收缩主张；
- 所有正文性能数字可追溯到脚本、原始迭代、manifest 和运行环境。

这份指南的目的不是预先写出“漂亮结论”，而是确保实现、实验和论文主张始终由同一组可验证事实支撑。
