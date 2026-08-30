# 实验阶段与正式矩阵

> 导航：[总索引](README.md) · 兼容入口：[THESIS_EXPERIMENT_GUIDE_DRAFT.md](../THESIS_EXPERIMENT_GUIDE_DRAFT.md)

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

---

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

---
