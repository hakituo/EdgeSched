# 代价模型与规划结果契约

> 导航：[总索引](README.md) · 兼容入口：[THESIS_EXPERIMENT_GUIDE_DRAFT.md](../THESIS_EXPERIMENT_GUIDE_DRAFT.md)

当前独立公式稿见 [`../edgesched_phase_aware_formulation.tex`](../edgesched_phase_aware_formulation.tex)；本文负责解释实验与实现契约，LaTeX 文件负责论文公式表达，二者均不得把未验证配置写成可执行候选。

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

视觉编码阶段补充 \(\hat T_V(r_V,c_V,s_V,resolution,image\_count)\)，并单列 V 的加载、预热、释放和缓存持有代价。
V+L 形成有依赖的请求路径；把 V 作为候选阶段不会增加一套独立 LLM 权重。

\[
C_{transition}=
T_{load}+T_{KVsave}+T_{KVrestore}+T_{swap}+P_{service}.
\]

代价必须与源配置、目标配置和 KV 大小关联，不再只有“全 GPU→全 CPU”的一个常数。

上式是转换成本组成示意，具体边只计实际发生的步骤：仅加载/释放 V 通常不需要保存或恢复 L 的 KV；
若为容纳 V 同时重建 L 实例，才计对应的 L/KV 转换。V 的正常编码属于执行成本，不再重复计入转换成本。
视觉端到端收益使用请求到达至有效回答/语音的关键路径度量，不能用排除了加载的 encode 或 decode 吞吐替代。

### 7.3 驻留迟滞

为避免反复加载和释放，需要加入迟滞：

- 图像队列非空时不恢复到 S0；
- 刚完成配置转换时，在最小驻留窗口内不因微小预测变化再次切换；
- SM 配置只在新会话、首个 TTS chunk、图像到达/完成、队列清空或后端异常等安全事件点重选，不在每个 token 或短时 utilization 波动后重建；
- 连续语音会话可将一次转换成本摊销到多轮回复；
- 已到达的连续视觉请求可摊销 V 的加载成本；队列清空后比较保温与释放，不能把未知未来请求作为已知收益。固定空闲保温窗口若采用，须作为单独参数消融并记录保留显存的机会成本；
- 队列清空或会话状态改变时重新评估目标配置；
- 任何迟滞不能给普通单图请求增加主动聚批等待。

### 7.4 算法主线：队列视界内的转换摊销

仅执行“枚举 SLO-feasible 配置 → Pareto 过滤 → 固定字典序选择”不足以构成
EdgeSched 的主要算法贡献。Pareto 规则负责在不同单位的 latency、throughput 与
reconfiguration 指标之间避免任意加权，但它只回答**目标配置选哪个**；核心调度问题还必须回答：

- 当前是否值得改变 residency；
- 为已知的下一阶段临时牺牲哪一阶段的 residency；
- 采用完整迁移、分层驻留还是保持当前配置；
- burst train 结束后何时恢复，且不在相邻任务间反复 evict/restore。

因此算法主线定义为 **phase-aware + transition-cost-aware +
queue-horizon-aware elastic residency scheduling**。在时刻 $t$，规划器只从已经到达的
任务、正在执行的 phase 及其已知后继阶段构造有界视界 $\mathcal H_t$。视界在未知分支、
队列尾或预设上限处停止；它不预测尚未到达的请求，也不为了扩大视界主动等待聚批。

对当前配置 $a_{cur}$ 和候选配置 $a'$，先在两者均满足硬 SLO 与容量约束的前提下，计算
视界内已知工作的累计关键路径收益：

\[
\Delta_{\mathcal H_t}(a_{cur},a')=
\sum_{i\in\mathcal H_t}
\left[\hat T_i(a_{cur})-\hat T_i(a')\right].
\]

吞吐收益必须先按视界内已知 token、chunk 或 image 数转换为预计完成时间，不能与秒、
百分比或 SLO penalty 做任意 weighted sum。只有同时满足以下条件才允许转换：

\[
\Delta_{\mathcal H_t}(a_{cur},a')
>
C_{tr}(a_{cur}\rightarrow a')+\eta,
\]

\[
t-t_{last\_transition}\ge \tau_{min}.
\]

其中 $C_{tr}$ 包含加载、KV 保存/恢复、同步、缓存重建和转换期间的服务损失，
$\eta$ 是预测误差与测量噪声的安全余量，$\tau_{min}$ 是最小驻留窗口。
容量不安全、后端失败或硬 SLO 即将失守时可以触发 fail-safe override，但必须记录原因，
不能把普通收益波动伪装成紧急切换。

这个视界定义自然区分单个 burst 与 burst train：单个 diffusion job 的收益可能不足以摊销
一次 LLM 迁出和恢复；已经排队的连续 burst 会延长 $\mathcal H_t$，累计收益超过转换代价后
才迁移，并在队列清空后重新评估恢复。Pareto/lexicographic policy 仍保留为候选目标的
确定性选择规则，但不能单独作为算法创新点。

### 7.5 可实现的预测器基线

预测器不能只作为架构图中的黑盒。首个可实现版本采用“离线画像插值 + 在线残差修正”，
而不是先引入难以解释的机器学习模型：

\[
\hat T_i(u_i,z_i,a,t)=
\operatorname{Interp}\!\left(P_i;u_i,z_i,a\right)
\cdot \alpha_{i,a}(t),
\]

\[
\alpha_{i,a}(t+1)=
\operatorname{clip}\left(
(1-\beta)\alpha_{i,a}(t)
+\beta\frac{T_i^{obs}}{T_i^{offline}},
\alpha_{min},\alpha_{max}
\right).
\]

- $P_i$ 来自 calibration 数据中的配置级画像表；
- $u_i$ 表示 input tokens、output budget、文本/chunk 长度、分辨率、steps、batch 等输入规模；
- $z_i$ 表示 context/KV、queue depth、co-runner、冷热状态和当前 residency；
- $\alpha_{i,a}$ 是按 phase/config 维护的有界 EWMA 乘性修正；
- 冷加载、稳态执行、转换和共置干扰分别建模，不能用一个系数混在一起；
- 超出画像覆盖范围、样本不足或置信度过低时保持当前安全配置，或使用保守上界，禁止激进外推。

转换判断应使用保守收益：由收益下界与转换成本上界比较，而不是只比较点预测。
至少记录预测值、实际值、残差、画像桶、在线修正系数和置信边界，使一次错误迁移可以完整回放。

预测器验收不只报告平均误差，还必须包含：

- 按 phase/config/input bucket 的 MAPE、绝对误差 p95 和覆盖率；
- 相对离线 oracle 的 decision regret；
- 已执行转换中真正完成 payback 的比例；
- 预测误差为 10%/20%/30% 时的 SLO、迁移次数和 thrashing 敏感性；
- 去掉在线修正、queue horizon、transition cost 或 minimum residency window 的消融。

### 7.6 当前实现顺序与停止条件

上述算法是 **C3 真实性与可行性门禁通过后的实现路线**，不是当前立即编码的功能。
在 T0/T3 与 Forge 的 direct NVML 校准、16 档候选重筛、LLM+TTS 单组合 smoke、正式共置矩阵
和 colocated 正残差完成前，不实现在线 predictor，不冻结转换阈值，也不把 Pareto planner
描述为已验证的核心算法。

当前阶段只要求实验产物保留未来建模所需的原始事实：输入规模、context/KV、当前/候选配置、
phase、queue depth、co-runner、冷热状态、实测执行时间、转换时间、显存峰值、SLO 结果与失败原因。
在 `G_safe/H_TTS/H_D` 冻结且真实共置 smoke 通过后，才能：

1. 用 calibration split 拟合画像插值与 EWMA 初值；
2. 冻结候选配置、$\eta$、$\tau_{min}$ 和视界上限；
3. 用独立 evaluation trace 比较 Pareto-only、instantaneous、transition-aware、
   queue-horizon-aware 与 oracle；
4. 若真实共置或转换收益无法覆盖成本，主动收缩算法与论文主张，不用复杂 predictor 包装 No-Go。

### 7.7 泛化验证边界

Phase abstraction 不能只靠接口命名证明泛化。主链路稳定后，evaluation 至少应沿以下维度中
选择两个工作负载变化，并尽量包含一个替代模型或后端实例：

- LLM 模型规模或 context/KV（例如 7B/9B、4K/16K/32K）；
- TTS backend、文本长度或 chunk 数；
- diffusion 之外的 GPU-required burst，或不同分辨率/steps；
- 单 burst、burst train、不同 queue depth 与 arrival pattern；
- 不同 co-runner 和冷热状态。

扩展实验应复用同一个 phase/profile/PlacementPlan 接口，不为新模型硬编码新的调度分支。
优先先完成一条可信主链路，再做少量有解释力的变化；不要在真实性门禁之前扩成模型组合穷举。

### 7.8 规划结果

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

---
