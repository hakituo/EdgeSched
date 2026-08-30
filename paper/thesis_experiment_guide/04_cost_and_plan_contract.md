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

---
