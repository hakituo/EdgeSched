# 当前实现差距与分阶段路线

> 导航：[总索引](README.md) · 兼容入口：[THESIS_EXPERIMENT_GUIDE_DRAFT.md](../THESIS_EXPERIMENT_GUIDE_DRAFT.md)

## 9. 当前实现差距

截至 2026-08-31，当前代码与目标机制之间至少存在以下差距：

1. `services/phase_aware/phase_model.py`
   - LLM 仍是 GPU_NORMAL / CPU_DEGRADED / RESTORING 三态。
   - TTS 预设默认 CPU，不会动态比较完整流水线。
   - C2/C3、KV、LLM+TTS 与 LLM+Forge 入口仍只执行纯文本 `Llama(model_path=...)`，不加载 `mmproj`；4B 统一文本/视觉路径目前只存在于独立的端点门子包 `experiments/elastic_residency/vision/`，尚未回灌到这些正式入口。
   - 现有 `data/adapters/vl_adapter.py` 是独立 Qwen2-VL Transformers 路径，不能冒充新主模型的 llama.cpp/libmtmd 多模态证据。
   - 端点门子包已实现真实 libmtmd 执行（projector warm/编码/释放、视觉 token 统计、`TEXT_ONLY`/`VISION_WARM_*` 状态分离），但**只通过了静态与假后端验证**；真实 GPU 视觉 smoke 未跑前不得据此宣称多模态可用。
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
   - TTS 尚未接入 Phase-Aware；T0/T3 direct NVML 隔离矩阵与 Qwen3.5-9B 的正式 LLM+TTS 共置均已完成，但模型迁移后必须重算新主模型的候选并重跑相关共置矩阵。
8. N 阶段实验
   - `experiments/phase_aware/exp2_*` 和 `exp3_*` 主要使用 `asyncio.sleep`，只能验证调度语义。
9. 辅助任务与播放环境
   - BGE ONNX 已完成接口 smoke；Faster-Whisper 权重、固定 Python 依赖和 CPU int8 单核真实接口 smoke 均已就绪。3.42 s 参考音频的单轮 RTF 约为 1.44，只能证明链路可用，不能替代 1/2/4 核正式测量。
   - WSL2 已通过 ALSA→WSLg Pulse 桥接打开真实 PortAudio 输出，并实现 Windows WASAPI 回退与显式采样率协商。两条路径的 1 s 静音 smoke 均为 0 underflow；这只解除环境门槛，真实 TTS chunk 与并发负载下的 underrun 仍须正式测量，缓冲时间线不能替代真实播放 callback。
10. GPU SM 管理
   - `experiments/elastic_residency/sm/` 已实现实验期 SM-A 能力探针、MPS 生命周期、客户端证据、清理硬门与 Green Context 只读探测，但尚无生产态 `GpuShareManager`；
   - TTS、LLM 与 Forge 仍由默认 CUDA 调度器竞争，平均 GPU utilization 不能证明任一阶段只需要对应比例的 SM；
   - WSL2 正式 SM-A 已运行：普通 CUDA 基线成功，但 `nvidia-cuda-mps-control` 不存在，100/75/50/25% 受控配置均为 `UNSUPPORTED`，当前结论为暂定 NO_GO；
   - MPS/静态分区的实际有效 SM 数仍未验证，后续只能在原生 Linux SM-A 重新通过后恢复；
   - `PlacementPlan` 尚无 `gpu_compute_backend`、`effective_sm_count` 和 `gpu_memory_limit` 字段；
   - 尚无受控 SM 份额下的 TTS 独立曲线、LLM→TTS 流水线或 TTS→Diffusion 共运行证据。

实现和论文必须诚实反映这些边界。

---

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

历史 Qwen3.5-9B GGUF 已核验 `qwen35.block_count=32`，并完成 direct NVML 画像；这些结果转为模型特定压力基线。新主实验计划使用 Qwen3.5-4B-Q4_K_M 与对应 BF16/F16 `mmproj`，官方结构同为 32 层，但必须在本地 GGUF 中重新核验，不能沿用 9B 的显存、RSS、KV 或性能数字。固定 \(R_{sys}=2\) 和 \(c_L=14\)，进行粗粒度扫描：

\[
k\in\{0,\lceil0.25N\rceil,\lceil0.5N\rceil,
\lceil0.75N\rceil,N,FULL\}.
\]

当前模型展开为 `k={0,8,16,24,32,FULL}`。`FULL` 与 `k=32` 分开保留，因为 llama.cpp 的完整 offload 还可能包含非 block 张量。

新模型进入正式曲线前增加多模态迁移门（入口：`experiments/elastic_residency/vision/vision_endpoint_smoke.py`）：

1. **已完成**：固定主 GGUF 与对应 `mmproj` 的来源、版本、SHA-256、量化方式和文件大小；对应关系由 `clip.vision.projection_dim == <arch>.embedding_length` 与 `general.basename`/`general.base_model.0.*` 一致证明，`mmproj` 声明 `clip.has_vision_encoder=true` 且 `mtmd_get_cap_from_file` 报告 `inp_vision=true`。本地 4B GGUF 的 `qwen35.block_count=32` 已重新核验，与 9B 的 32 层是各自独立结论，不互相沿用；
2. **已实现、待真实运行**：用 libmtmd 对固定真实图片执行 projector。`split_encode` 模式先 `mtmd_encode_chunk` 再核对 `mtmd_get_output_embd` 非空，然后 `mtmd_helper_decode_image_chunk` 把视觉嵌入送进同一个语言模型上下文；`chunk_single` 模式与上游 `MTMDChatHandler` 调用一致。两种模式都没有静默文本 fallback；
3. **已实现、待真实运行**：分别测量 `TEXT_ONLY`、`TEXT_ONLY_CHAT`（同 prompt 无图片对照）、`VISION_WARM_GPU`、`VISION_WARM_CPU` 与 projector 释放后的整卡显存差值与进程 RSS 差值。禁止把 `mmproj` 文件大小当作 \(M_V\)；
4. **待执行**：用 K0/FULL 完成 512+128 上下文的文本与视觉端点 smoke，核验视觉 token 数、KV、输出文本、direct NVML 新鲜度、swap 与资源归还；
5. 端点通过后才重跑 4B 的 C2/C3、KV 和 LLM+TTS/Forge 共置。TTS 与 Forge 隔离校准在环境和后端未变化时可作为输入复用，但最终候选和共置结论必须重算。

每个配置测试约 512、2048 和 4096 tokens 的上下文，固定生成 128 tokens，独立预热并至少重复 5 次。记录加载/重新配置时间、GPU显存、主机内存、TTFT、prefill tokens/s、decode tokens/s、token gap、CPU利用率和KV保存恢复时间。

根据真实画像求两个临界档位：

\[
H_{TTS}=\max k:\ M_L(k)+M_T+G_{safe}\le V_{GPU},
\]

\[
H_D=\max k:\ M_L(k)+M_D(b=1)+G_{safe}\le V_{GPU}.
\]

对于视觉 ready 状态，另行计算包含 \(M_V(r_V,u)\) 的 `H_TTS^vision/H_D^vision`；不能把未执行的 projector 文件大小直接加到文本曲线，也不能用文本 `H` 代表真实识图阶段。

在每个临界点附近补测 \(k\pm2\) 和 \(k\pm4\)。第一版使用 llama.cpp 的连续层 offload，不自行设计注意力层、MLP 层或单个张量的任意拆分。

历史 Forge batch=1 的约 5904 MiB 峰值和后续旧采样后端复测都只能用于候选筛选。计算正式 \(H_D\) 前必须使用与 C2/C3 相同的 direct NVML 后端重新校准，不能跨后端拼接整卡峰值。

**C2 旧 CLI 完成记录（2026-08-29）**：`c2_llm_residency_curve_20260829_verified_v3` 基于干净 commit `63e75ba` 完成 `6 配置 × 3 上下文 × 5 轮 = 90/90` 条真实驻留曲线，0 failure、0 fallback，有效性门与 30 ms 性能目标全部通过。该批次现位于 `_archive/superseded/c2/`，只作工程基线；当前性能真源是 C2/C3 direct NVML v2。完整目录映射见 `docs/ELASTIC_RESIDENCY_RESULT_CATALOG_20260830.md`。

#### P1-C：GPU SM 能力探针与单模型计算份额曲线

目标：确认当前 RTX 5070 Laptop + WSL2 + CUDA/PyTorch 环境是否能够真实限制并验证不同客户端的 GPU 计算份额，并得到 TTS、LLM 和 Diffusion 的离散 SM 性能曲线。

第一版以 MPS 独立进程 profile 为主。WSL2 已确认无法获得 MPS binary，因此后续只在原生 Linux 重新运行 SM-A；Green Context 保持只读能力探测，不因 WSL2 No-Go 自动升级为第二套生产后端。

**SM-A 正式结果（2026-08-30）**：`sm_a_20260830_direct_nvml_v1` 基于干净 commit
`ca7a7ff0832342fc76f261b6b7d8d720b5a116fa` 在 WSL2 真实 CUDA 环境完成。
普通 CUDA 基线成功，RTX 5070 Laptop 报告 36 个物理 SM，direct NVML、Git provenance、
0 failure、0 fallback 与 0 cleanup failure 均满足；但环境中不存在
`nvidia-cuda-mps-control`，100/75/50/25% 四个配置均为 `UNSUPPORTED`，最终
`verdict=NO_GO`。因此 WSL2 路径暂停 SM-B/SM-C，不宣称受控 SM 分配，不修改在线规划器；
待原生 Linux 重新运行 SM-A 后再决定是否恢复。Green Context 只保留只读能力探测结果，不自动转为生产后端。

已实现与待实现边界：

```text
experiments/elastic_residency/sm/
  sm_capability.py                 # 已实现：SM-A 编排
  mps_runtime.py / mps_daemon.py   # 已实现：MPS 生命周期与清理
  green_context_probe.py           # 已实现：只读能力探测
experiments/elastic_residency/     # 以下仍待原生 Linux SM-A 通过后实现
  tts_sm_curve.py
  llm_sm_curve.py
  diffusion_sm_curve.py
  tts_diffusion_sm_colocation.py
tests/elastic_residency/
  test_sm_capability*.py           # 已实现
  test_sm_profiles.py              # 已实现
  test_sm_plan_validation.py       # 生产计划契约实现时再新增
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

---
