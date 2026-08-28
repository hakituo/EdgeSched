你说得对：不能直接把“等分核实验”当成核心需求。分核只是联合调度中的一个执行参数，而且必须由单模型扩展曲线推导，不能先拍脑袋列组合。

我重新看了现有代码和实验，下面这份才是可以据此动工的完整需求。

# EdgeSched 新版需求规格

## 1. 研究目标

在单用户、单机、8GB GPU、WSL2 可用约16GB内存的环境中，同时运行：

- Qwen3.5-9B-Q4_K_M：常驻交互式 LLM；
- Qwen3-TTS-0.6B：句子级流式语音；
- Forge Diffusion：GPU 必需的图像任务；
- 轻量 ASR/RAG：CPU 辅助阶段。

系统需要根据当前工作流动态决定：

\[
a_t=(r_L,r_T,c_L,c_T,b,\mu)
\]

其中：

- \(r_L\)：LLM 驻留配置，CPU、部分 GPU、全 GPU；
- \(r_T\)：TTS 位于 CPU 或 GPU；
- \(c_L,c_T\)：LLM、TTS 的 CPU 核心预算；
- \(b\)：Diffusion batch size；
- \(\mu\)：保持、迁出、回迁或重新配置。

核心贡献应升级为：

> 面向多模态流水线的代价感知弹性驻留与异构资源调度，而不再只是 LLM 整体迁出和迁回。

---

## 2. 当前代码与新需求之间的差距

现有实现还不能直接支撑这项主张：

- [phase_model.py](/codex_workspace/6d9b9694-fcbe-43a1-bb0c-e411eeb565d4/services/phase_aware/phase_model.py) 中 LLM 只有 GPU、CPU、恢复中三态，没有部分驻留。
- [gpu_lease.py](/codex_workspace/6d9b9694-fcbe-43a1-bb0c-e411eeb565d4/services/phase_aware/gpu_lease.py) 仍规定 Resident 和 Burst 重任务互斥，`SHARED` 没有进入真实调度。
- [cost_model.py](/codex_workspace/6d9b9694-fcbe-43a1-bb0c-e411eeb565d4/services/phase_aware/cost_model.py) 虽然预测 burst 时长，但 GPU_REQUIRED 最后仍然直接迁移；收益阈值和队列压力阈值没有真正控制决策。
- [phase_scheduler.py](/codex_workspace/6d9b9694-fcbe-43a1-bb0c-e411eeb565d4/services/phase_aware/phase_scheduler.py) 将 LLM 剩余执行时间固定为 0，并在每个图像任务后立即恢复，连续图像可能反复迁移。
- TTS 目前只有接口和 Mock，没有可复现的真实 Qwen3-TTS worker。
- N 阶段实验使用 `asyncio.sleep`，不能作为真实性能证据。
- C++ 原生 LLM worker暂不支持动态迁移；真实迁移位于 Python llama.cpp 路径。
- 现有部分卸载只支持重新加载 `n_gpu_layers=0` 或恢复原配置，不支持任意驻留档位。

因此，这不是“补几个实验”的任务，而是一次调度模型和实验框架升级。

---

## 3. 工作流语义

### 3.1 语音工作流

\[
ASR/RAG\rightarrow LLM_{\text{stream}}
\rightarrow SentenceSplitter
\rightarrow TTSChunkQueue
\rightarrow TTS
\rightarrow Playback
\]

TTS 按句子或稳定文本块执行。

一个 TTS chunk 内部暂不抢占，但 chunk 之间允许调度和切换设备。

实验必须使用确定性的断句规则。现有随机逗号断句需要关闭，否则不同实验无法比较。

### 3.2 图像工作流

普通图像请求：

\[
LLM\rightarrow Diffusion(b=1)
\]

要求立即准入，不人为等待其他请求到达。这里的“立即”是指不增加聚批等待窗口，必要的迁移和显存释放仍计入图片启动延迟。

### 3.3 多图语义

必须区分：

- 同一 prompt 请求多张候选图：允许 b=2 或 b=4；
- 不同 prompt 的独立请求：保持串行队列，不能拿当前同一 prompt 的 batch 实验直接证明可以合批；
- 不主动等待未来请求；
- b=8 排除；
- 队列未清空时不反复恢复 LLM。

现有 batch 数据只能证明同一 prompt 的 `batch_size` 扩展特性。

---

## 4. 运行状态

### 状态 S0：没有图像任务

候选配置：

1. `LLM_FULL_GPU + TTS_CPU`
2. `LLM_CPU + TTS_GPU`
3. `LLM_HYBRID(k) + TTS_GPU`
4. 显存允许时测试 `LLM_FULL_GPU + TTS_GPU`

选择依据是整个 LLM→TTS 流水线，而不是单模型速度。

\[
TTFA =
T_{\mathrm{reconfigure}}
+T_{\mathrm{first\ sentence}}^{LLM}
+T_{\mathrm{first\ chunk}}^{TTS}
\]

稳态吞吐近似受较慢阶段约束：

\[
T_{\mathrm{pipeline}}
\approx
\max(T_{\mathrm{LLM\ chunk}},T_{\mathrm{TTS\ chunk}})
\]

### 状态 S1：一个图像任务运行

候选配置：

1. `Diffusion_GPU + LLM_CPU + TTS_CPU`
2. `Diffusion_GPU + LLM_HYBRID(k) + TTS_CPU`
3. `Diffusion_GPU + LLM_CPU + TTS_GPU`
4. 显存允许时测试 `Diffusion_GPU + LLM_HYBRID(k) + TTS_GPU`

不能预先假定第四种一定可行。需要同时验证显存容量和 GPU 算力干扰。

### 状态 S2：多个图像任务排队

- 保持当前降级/混合驻留状态；
- 不在两个图像任务之间恢复 LLM；
- 不同 prompt 串行；
- 同 prompt 多候选图可以根据画像选择 b=1、2、4；
- 队列清空后，重新配置到 S0 的最优状态，而不一定恢复成 LLM 全 GPU。

---

## 5. 任务画像

每种模型不能再只有一个固定显存值，而要形成配置相关画像：

\[
\mathcal P_i(z)=
\langle
M_i(z),T_i(z),C_i(z),E_i(z)
\rangle
\]

其中 \(z\) 是设备、驻留层数、线程数和输入特征。

### LLM画像

输入特征：

- `n_gpu_layers`；
- CPU线程数；
- 上下文/KV长度；
- prefill长度；
-输出长度。

记录：

- TTFT；
- prefill tokens/s；
- decode tokens/s；
-显存、内存；
-重新配置时间；
- KV保存和恢复时间。

### TTS画像

输入特征：

- CPU或GPU；
- CPU线程数；
- 文本字符数；
- 分句长度；
-语音时长。

记录：

- 首音频块延迟；
- chunk合成时间；
-音频时长；
- RTF；
-峰值显存和内存。

### Diffusion画像

输入特征：

-模型；
-分辨率；
-steps；
-batch size；
-是否有其他GPU任务共存。

记录：

-模型加载时间；
-生成时间；
-峰值显存；
-LLM/TTS共存后的 slowdown。

---

## 6. 可行配置约束

显存约束：

\[
M_L(r_L,K)+M_T(r_T)+M_D(b)+G_{\mathrm{safe}}
\le V_{\mathrm{GPU}}
\]

CPU约束：

\[
c_L+c_T+R\le C_{\mathrm{physical}}
\]

其中：

- \(G_{\mathrm{safe}}\) 不能继续使用拍脑袋静态值，要根据多轮峰值波动确定；
- \(R\) 是给 Forge驱动、音频播放、调度器和系统预留的物理核；
- SMT逻辑线程不直接视为额外物理核心。

未知配置、画像缺失或者预测显存不安全时，必须回退到：

\[
Diffusion_{GPU}+LLM_{CPU}+TTS_{CPU}
\]

---

## 7. 调度目标

不采用难以解释的主观加权和，而使用分层约束。

第一层：正确性

- 不OOM；
- KV和对话状态不丢失；
- TTS chunk不丢失、不重复、顺序正确；
- 图像队列不绕过背压。

第二层：交互约束

- 单图请求不主动等待聚批；
-稳态语音无断流；
-目标 \(RTF_{\mathrm{TTS}}^{p95}\le0.8\)；
-运行中的 Diffusion 不被强行中断。

第三层：在可行配置中优化

1. 最小化第一段语音 TTFA；
2. 最小化第一张图片完成时间；
3. 最大化 LLM decode tokens/s；
4. 减少迁移次数和驻留抖动；
5. 降低完整回复完成时间。

Diffusion slowdown 的 5%不能直接当绝对真理。实验应报告 5%、10%、20%限制下的可行配置和 Pareto前沿。

---

## 8. 分核要求

不能直接从 8+6、7+7 开始。

首先分别测 LLM 和 TTS 的单独核心扩展曲线：

\[
c\in\{2,4,6,8,10,12,14,16\}
\]

然后求 TTS 的最低可用核心数：

\[
c_T^{\min}
=
\min\left\{
c:
RTF_{\mathrm{TTS}}^{p95}(c)\le0.8
\land Gap_{\mathrm{audio}}=0
\right\}
\]

并发实验只测试：

\[
c_T\in
\{c_T^{\min}-2,c_T^{\min},c_T^{\min}+2\}
\]

\[
c_L=C-R-c_T
\]

之后才对最优两个候选比较：

- 只限制线程数；
- 限制线程但不绑核；
-分离绑核；
-按CCD隔离；
-使用SMT；
-不预留系统核心。

7+7只能作为对称基线，不能作为默认设计。

---

## 9. 代价模型

代价模型需要预测“已经到达的任务将占用多久”，而不是为了合批去预测未来请求。

需要维护：

\[
\hat T_D(model,resolution,steps,b)
\]

\[
\hat T_L(r_L,c_L,K)
\]

\[
\hat T_T(r_T,c_T,length)
\]

\[
\hat T_{\mathrm{transition}}
(r_{\mathrm{src}},r_{\mathrm{dst}},K)
\]

迁移代价必须与源、目标驻留档位相关：

\[
C_{\mathrm{transition}}
=
T_{\mathrm{load}}
+T_{\mathrm{KVsave}}
+T_{\mathrm{KVrestore}}
+T_{\mathrm{swap}}
\]

不再只有“全GPU→全CPU”一个值。

需要增加驻留迟滞：

- 图片队列非空：不恢复；
-当前配置刚切换：在最小驻留时间内避免反复切换；
-连续语音会话：迁移成本可跨多轮摊销；
-会话结束或队列清空：重新选择 S0 配置。

---

## 10. 实现阶段

### 阶段一：真实画像与可行性，不改核心调度

需要新增可复现实验入口：

1. LLM不同 `n_gpu_layers × CPU线程 × KV长度`。
2. TTS不同 `device × CPU线程 × 句子长度`。
3.无图像状态下的静态共存配置。
4. Diffusion运行时的静态共存配置。
5.每种配置的显存时间线和失败原因。

如果没有任何混合驻留配置优于全CPU降级，就停止弹性驻留开发，保留二元迁移并如实报告。

### 阶段二：真实TTS流水线

需要：

- Qwen3-TTS真实adapter/worker；
-模型常驻，不能每句重新加载；
-确定性分句器；
-句子队列和音频播放缓冲；
-每个chunk完整时间戳；
- CPU线程数和进程亲和性控制；
-设备切换和加载耗时记录。

现有单个 `smoke_tts_gpu.wav` 没有脚本、参数和manifest，不能作为论文证据。

### 阶段三：弹性驻留

将 LLM 状态改为离散配置，例如：

\[
r_L\in
\{\mathrm{CPU},H_1,H_2,H_3,\mathrm{FULL\ GPU}\}
\]

MVP不做CUDA通用分页，也不宣称原地逐页迁移。

当前 llama.cpp 的 `n_gpu_layers` 是加载配置，因此第一版实现应是：

> 创建目标驻留配置的新实例、保存/恢复KV、原子或紧急替换。

接口应泛化为：

```text
reconfigure_llm_residency(target_profile, transition_mode)
```

而不是只有：

```text
offload_llm_to_cpu()
restore_llm_to_gpu()
```

### 阶段四：联合规划器

AdmissionController不再只返回 CPU/GPU和是否迁移，而应返回完整计划：

```text
PlacementPlan
- llm_residency
- tts_device
- llm_cpu_cores
- tts_cpu_cores
- image_batch_size
- transition_mode
- predicted_metrics
- fallback_plan
- decision_reason
```

GPU lease改为显存配额模型。只有经过真实共存画像验证的组合，才允许共享GPU。

### 阶段五：C++同步

先在 Python控制面完成真实机制。

如果论文继续声称C++负责核心迁移与调度，则必须同步：

-驻留状态枚举；
- PlacementPlan契约；
-显存配额；
- CPU核心预算；
- pybind11绑定；
- C++/Python一致性测试。

否则论文必须明确：C++负责原生执行，弹性驻留和迁移决策由Python控制面完成。

---

## 11. 完整实验矩阵

### E1：单模型画像

- LLM核心扩展和驻留扩展；
- TTS CPU/GPU与核心扩展；
- Diffusion显存和持续时间；
-迁移时间随KV长度变化。

### E2：无图像的LLM→TTS流水线

比较：

- LLM全GPU＋TTS CPU；
- LLM全CPU＋TTS GPU；
- LLM不同混合驻留＋TTS GPU。

分别测试：

-单轮语音回复；
-连续多轮语音会话；
-短、中、长回复。

### E3：Diffusion期间的流水线

比较S1中的四种配置。

需要两个场景：

-普通单图请求；
-明确标记为压力实验的持续GPU占用场景。

测量TTFA、音频断流、LLM tokens/s、图片时间和显存峰值。

### E4：CPU核心分配

由E1数据推导候选，不预先穷举所有等分方案。

最优配置再做绑核、CCD和SMT消融。

### E5：代价感知策略对比

基线至少包括：

1. GPU串行等待；
2.当前Always Full Migration；
3.固定静态最佳配置；
4.新版Cost-Aware Elastic Residency；
5.根据真实执行时长事后选择的Oracle。

原来的Cold Restart、Symmetric Migration可保留为迁移机制基线，但不是新版调度策略的唯一对手。

### E6：图片队列

分别测：

-单张图片；
-同prompt多候选图；
-不同prompt连续请求；
-队列清空前是否避免重复迁移。

### E7：真实N任务工作流

至少使用：

\[
ASR/RAG\rightarrow LLM\rightarrow TTS+Image
\]

ASR/RAG可以轻量，但必须是真实执行，不能继续用sleep证明性能。

### E8：消融与稳健性

关闭以下机制分别测试：

-无时长预测；
-无部分驻留；
-无CPU核心隔离；
-无驻留迟滞；
-无KV保存；
-固定设备放置。

同时改变：

- KV长度；
- TTS文本长度；
- Diffusion时长；
-图像队列深度；
-预测误差。

---

## 12. 统计与证据要求

每个配置：

-独立预热，不能只预热b=1；
-随机或正反顺序执行，降低温度漂移影响；
-至少5次完整独立运行；
-报告p95时至少收集30个有效chunk或事件；
-冷启动与稳态结果分开；
-保存原始JSON，不能只保存汇总；
-保存模型哈希、参数、种子、CPU亲和、WSL内存限制和后端类型；
-记录是否真实LLM、真实TTS、真实Forge；
-结果先进入 `experiment_results/`，人工确认后才能晋升到论文正文数据。

换成Qwen3.5后，原来的迁移数字不能直接继续作为新版结果。

---

## 13. 验收标准

机制完成需要同时满足：

-所有驻留切换后对话上下文一致；
-异常时能安全回退到全CPU或完整迁出；
-单图请求没有人为聚批等待；
-不同prompt不会错误合批；
- TTS chunk顺序正确且无丢失；
-调度日志能够解释为什么选择当前配置；
-预测值和真实值均写入结果；
-至少存在一个场景，新策略相对二元迁移形成真实Pareto改进；
-如果不存在Pareto改进，论文不得声称弹性驻留有效；
-所有正文数字可追溯到原始运行和manifest。

## 最实际的动工顺序

现在不要先改C++调度器，也不要先写等分核实验。

正确顺序是：

1. 做真实Qwen3-TTS worker和可复现smoke；
2. 做LLM、TTS、Diffusion三个独立profiler；
3. 做静态放置/部分驻留可行性实验；
4. 根据结果确定真正存在的候选配置；
5. 才实现PlacementPlan和代价模型；
6. 再做CPU分核与端到端实验；
7.最后同步C++、论文和图表。

这才是一套完整、可实施、不会在中途因为错误假设而全部返工的需求。