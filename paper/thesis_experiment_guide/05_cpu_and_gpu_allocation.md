# CPU 核数与 GPU 计算份额方法

> 导航：[总索引](README.md) · 兼容入口：[THESIS_EXPERIMENT_GUIDE_DRAFT.md](../THESIS_EXPERIMENT_GUIDE_DRAFT.md)

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

LLM 与 TTS 共运行时，资源平衡以端到端流水线为准。当前非增量 TTS 的首句波形就绪延迟，
仅在所列阶段串行时为：

\[
T_{sentence\ ready}=T_{transition}+T_{LLM}^{first\ sentence}(r_L,c_L,s_L)
+T_{TTS}^{first\ sentence}(r_T,c_T,s_T),
\]

重叠时使用真实关键路径；首次实际播放另行计时，并计入播放器等待与预缓冲。
此处 chunk 指当前非增量 TTS 的句级合成任务及其完整波形，不指句内增量音频块，
与第 02、08 模块的口径一致。无既往断流且忽略设备供给提前量时，对每个后续 chunk 检查：

\[
t_{j+1}^{ready}\le t_j^{playback\_end}.
\]

正式验收使用第 08 模块固定的连续播放 deadline、设备供给提前量和实际 underrun 采集口径，
不在发生断流后移动 deadline。当前 S0 截止阈值仍待确认；本节的 SM 设计目标不等于 S0 已通过。

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

---
