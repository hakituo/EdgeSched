# 任务画像与资源可行性约束

> 导航：[总索引](README.md) · 兼容入口：[THESIS_EXPERIMENT_GUIDE_DRAFT.md](../THESIS_EXPERIMENT_GUIDE_DRAFT.md)

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

---

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

---
