# 证据使用边界与论文组织

> 导航：[总索引](README.md) · 兼容入口：[THESIS_EXPERIMENT_GUIDE_DRAFT.md](../THESIS_EXPERIMENT_GUIDE_DRAFT.md)

当前方法公式与架构图分别维护在 [`../edgesched_phase_aware_formulation.tex`](../edgesched_phase_aware_formulation.tex) 和 [`../system_architecture.tex`](../system_architecture.tex)；它们是设计材料，不是性能证据。

## 15. 现有数据的使用边界

### 15.1 可以保留为历史或辅助证据

- `paper/archive/pre_rewrite_20260830/experiment_data/` 中的旧迁移结果；
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
  - Forge b=1 旧采样后端复测（10 项门禁全通过，swap 增长 0，干净 commit `50d2362`）；
  - 空闲高水位 415 MiB、常驻基线 419 MiB、5 轮峰值稳定 5667 MiB、b=1 延迟 mean 4.844 s；
  - `M_D = 5252 MiB` 只用于排程；参与阈值冻结前必须用 direct NVML 同后端校准。
- `experiment_results/elastic_residency/c3_analysis/c3_h_reprofiled_20260829.json`
  - 旧 TTS/Forge 口径下的候选容量筛选：`H_TTS(T0)=16`、`H_TTS(T3)=32 且 FULL 可行`（三上下文一致）；
  - **`H_D` 按上下文分化**：512 下为 8，2048/4096 下为 0——按 context 分组保留，不合并成常数；
  - 不是正式冻结值：T0/T3、Forge direct NVML 校准与 colocated 残差均 pending。
- `experiment_results/elastic_residency/sm_capability/sm_a_20260830_direct_nvml_v1/`
  - WSL2 真实 CUDA 基线成功，36 个物理 SM，direct NVML、0 failure/fallback/cleanup failure；
  - MPS binary 不存在，四个 percentage 均为 `UNSUPPORTED`，因此 SM-A 在 WSL2 暂定 NO_GO；
  - 该记录是环境能力结论，不是受控 SM 性能曲线；原生 Linux 重新通过前不展开 SM-B/C。
- `experiment_results/phase_aware/exp2_*`、`exp3_*`
  -使用 Mock/sleep；
  -只能验证流程和统计字段。

### 15.3 必须重跑

换用 Qwen3.5-9B-Q4_K_M 后，下列性能需在冻结环境重跑：

- 完整迁出与恢复；
- 真实工作流；
- T0/T3 与 Forge 参与容量阈值的整卡数据（统一为 direct NVML）；
- 真实共置残差、无图像 LLM→TTS 重叠和后续正式弹性策略评测。

---

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

---

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

---

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

---
