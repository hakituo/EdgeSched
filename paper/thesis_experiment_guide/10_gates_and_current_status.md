# Go/No-Go 门禁与当前任务状态

> 导航：[总索引](README.md) · 兼容入口：[THESIS_EXPERIMENT_GUIDE_DRAFT.md](../THESIS_EXPERIMENT_GUIDE_DRAFT.md)

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

- MPS/Green Context 在冻结的目标环境不可用或实际有效 SM 数无法验证；
- 共存导致明显 OOM 风险；
- 所有受控 SM 配置都被 CPU 回退、不受控共享或显存共驻留但计算串行方案支配。

### Gate D：正式实验

进入正式实验前必须满足：

- 策略已冻结；
- 所有真实后端通过 E0；
- calibration 与 evaluation 数据分离；
- manifest 完整；
- 正文中计划使用的每个指标均能实际采集。

---

## 20. 当前任务顺序

状态符号：`[x]` 表示已完成并有证据；`[ ]` 表示待完成；`PAUSED` 表示前置门禁未通过，既不算完成也不应继续跑。

真实 TTS worker 和基础 smoke 已经完成，但其运行结果仍处于 `experiment_results/` 的探索阶段。当前与归档批次的边界见 `docs/ELASTIC_RESIDENCY_RESULT_CATALOG_20260830.md`，近期可执行步骤以 `docs/ELASTIC_RESIDENCY_EXPERIMENT_EXECUTION_PLAN_20260828.md` 为准：

- [x] 真实 Qwen3-TTS-0.6B-Base worker 与基础可复现 smoke
- [x] A1：`a1_smoke_20260829_verified_v7` 已在干净 commit `ae68608` 上通过 GPU/jitter/CPU 首样本/provenance 全部 17 项硬门槛；v6 仅保留为旧门槛调试记录
- [x] B1：`b1_tts_components_20260829_verified_v2` 已在真实 Qwen3-TTS/CUDA 后端完成 T0–T3 组件发现与单文本 smoke，4/4 配置通过；该冷态单轮数据不用于性能结论
- [x] B2：`b2_tts_matrix_20260829_verified_v1` 已完成 60/60 条；T0/T3 为条件式 Pareto，T1/T2 被支配，且 60 条 RTF 全部大于 1
- [x] C1：`c1_llm_residency_512_20260829_verified_v1` 已完成 K0/FULL 的 512-token 真实端点 smoke，两端均通过 19/19 项门禁；它是单轮基础设施证据，不作为正式性能曲线
- [x] C2 direct NVML：`c2_llm_residency_curve_20260829_direct_nvml_v2` 已完成 90/90 条粗驻留曲线，0 failure；旧 CLI 曲线只保留为工程基线
- [x] C3 direct NVML 临界补测：`c3_llm_critical_refinement_20260829_direct_nvml_v2` 已完成 150/150 条；与 C2 合并后覆盖 16 个驻留档位
- [x] C3 KV 校准：`c3_kv_cache_calibration_20260830_direct_nvml_v1` 已完成 48/48 条，冻结 GPU/宿主 KV buffer 与完整序列化状态的独立语义
- [x] T0/T3 direct NVML 隔离矩阵：v2 已基于干净 `0261e4da` 完成 30/30 技术核验，0 failure/fallback，运行期 swap 与同步查询新鲜度门禁完整；未晋升论文正文数据
- [x] **Qwen3.5-9B C3 已按资源门禁停止，保留为模型特定压力证据**：
  1. [x] Forge b=1 direct NVML v1 已完成 5/5，逐窗新鲜度、swap 增长和核心归还门禁全部通过，`M_D=5484 MiB`；
  2. [x] 完整 16 档 + B2 direct v2 + Forge direct v1 已生成 `c3_h_direct_nvml_20260831_v1.json`：T0 边界为 512/K24、2048/K20、4096/K20，T3 三个 context 均筛到 FULL；这些仍是容量候选；
  3. [x] C4096/K20+T0 artifact smoke 已 1/1 技术通过：`P_colocated=6686.793 MiB`，T0 共置增量 2252 MiB，正残差 0，计入 501.282 MiB 安全余量后容量可行，0 failure/fallback；
  4. [x] `llm_tts_colocated_candidates_20260831_direct_nvml_v1` 已完成 6 候选×5 轮，共 30/30、0 failure/fallback；全部真实后端、direct NVML，新鲜度和 swap 门禁通过，colocated 正残差最大值为 0；
  5. [x] `c3_h_tts_colocated_20260831_v1.json` 已离线复算并冻结 **9B 模型口径**的 `G_safe=501.282 MiB` 与按 context 的 `H_TTS`：T0 为 512/K24、2048/K20、4096/K20，T3 三个 context 均为 FULL；模型迁移后这些值不再代表主实验模型；
  6. [x] 512/K6+Forge 真实 smoke 的后端、PNG、direct NVML、新鲜度、KV、GPU 容量和 provenance 均有效，但 swap 增长约 290.44 MiB，整体失败；
  7. [x] 探索性 512/K4+Forge smoke 同样只有 swap 门失败，增长约 2070.32 MiB；降低 GPU 层数释放了显存，却显著增加主机换页，因此不再继续 K2，也不运行 9B 正式 15 条矩阵；
  8. [x] 9B 的 `H_D` 未冻结；两次失败作为“仅看显存会选错目标档”的负面证据保留，不得晋升为 9B 共置可行结论。
- [ ] **主实验模型迁移：Qwen3.5-4B-Q4_K_M + 对应 BF16/F16 `mmproj`**：
  1. [x] 已冻结研究口径：统一模型必须真实执行文本与图片理解，`mmproj` 是正式 session-warm 视觉组件，不是静态填充；9B 保留为压力/NO_GO 对照；
  2. [x] 主 GGUF 与 `mmproj` 已在 WSL2 固定并核验：`Qwen3.5-4B-Q4_K_M.gguf`（`qwen35`、32 层、`context_length=262144`、`embedding_length=2560`、`file_type=15`、426 张量、2740937888 B、SHA-256 `00fe7986…ef11a4`）与 `mmproj-BF16.gguf`（`clip`/`mmproj`、`qwen3vl_merger`、`has_vision_encoder=true`、24 层视觉块、`image_size=768`、`patch=16`、`merge=2`、`projection_dim=2560`、298 张量、675569344 B、SHA-256 `302b92d5…21f3a6`）。对应关系判据为 `projection_dim == embedding_length` 加 `general.basename`/`general.base_model.0.*` 一致，不靠文件名推断；旧 9B GGUF 已从 WSL 删除，其结果只作模型特定压力与负面边界证据；
  3. [x] 环境能力已核验：`llama-cpp-python==0.3.35` 随包提供 `libmtmd.so` 与 `llama_cpp/mtmd_cpp.py`，`llama_chat_format` 已含通用 `MTMDChatHandler`，`mtmd_get_cap_from_file` 对该 `mmproj` 报告 `inp_vision=true`/`inp_audio=false`，media marker 为 `<__media__>`；**无需新增或修改依赖**；
  4. [x] 已实现多模态端点门代码链路 `experiments/elastic_residency/vision/`：artifact 门禁、确定性 448×448 固定图片（`pixel_payload_sha256=a1514034…43cb00`，期望 196 个视觉 token）、libmtmd projector warm/执行/释放、`TEXT_ONLY` / `TEXT_ONLY_CHAT` / `VISION_WARM_GPU` / `VISION_WARM_CPU` 四状态与 K0/FULL 端点 smoke 编排；Windows 静态与假后端测试 72 项通过，父进程门禁失败路径已在 WSL2 用合成 artifact 实跑；
  5. [ ] **待执行**：WSL2 真实 GPU 视觉 smoke（`vision/vision_endpoint_smoke.py`，K0/FULL × `TEXT_ONLY`/`VISION_WARM_GPU`，512+128 上下文），核验 projector 真实执行、视觉 token 数、模型文字输出、projector warm/释放的 GPU/RSS 增量、KV、direct NVML 新鲜度、swap 与资源归还；
  6. [ ] 端点通过后才重跑 4B 的 C2/C3、KV 与临界补测；
  7. [ ] 重算 4B 的 `G_safe/H_TTS/H_D`，分别保留 `TEXT_ONLY` 与 `VISION_WARM` 候选，并重跑真实 LLM+TTS/Forge 共置；TTS T0/T3 与 Forge b=1 的隔离结果在后端与环境不变时可复用为输入；
  8. [ ] 迁移完成前不得把任何 9B 阈值改名为 4B 结论，不得把 `mmproj` 文件大小加到旧 9B 曲线上冒充测量，也不得晋升到 `paper/experiment_data/01_paper_main/`。
- [x] **SM-A：GPU 计算份额能力探针**：真实 WSL 记录 `sm_a_20260830_direct_nvml_v1` 已完成；MPS binary 在 WSL2 不存在，结论**暂定 NO_GO**。注意：这是环境能力门禁导致的「暂停」，不是永久取消——待具备可用的原生 Linux 启动环境后重做 SM-A，再决定是否进入 SM-B/C/D/E
- [ ] **SM-B（PAUSED）：TTS T0/T3 隔离 SM 曲线**：因 SM-A 暂定 NO_GO 按门禁暂停，不运行、不伪造受控 SM 曲线；待 SM-A 在原生 Ubuntu 恢复后重排
- [ ] **SM-C（PAUSED）：Diffusion b=1 隔离 SM 曲线**：因 SM-A 暂定 NO_GO 按门禁暂停，不运行、不伪造受控 SM 曲线；待 SM-A 在原生 Ubuntu 恢复后重排
- [ ] **SM-D：LLM→TTS 平衡实验**：比较不受控共享、显存共驻留但计算串行、静态 SM 分配，并记录第一句、TTFA、chunk deadline 与真实 underrun
- [ ] **SM-E：TTS→Diffusion 共运行实验**：先固定 LLM 在 CPU，求 TTS 最小可行 SM 数，将剩余 SM 给 Diffusion；前序形成 Pareto 后才测试三阶段 GPU 共享
- [ ] **CPU 核数实验（未到正式矩阵阶段）**：物理核拓扑、affinity 与租约基础设施已完成；BGE 与 Faster-Whisper 仅单轮接口 smoke；正式入口 `auxiliary_core_budget.py` 尚未实现；JSON RAG/BGE/ASR 的 1/2/4 核曲线未跑；`R_sys={0,1,2,4}` 未跑；LLM/TTS 的 `{2,4,6,8,10,12,14,16}` 核扩展曲线未跑。合理顺序：先完成 C3 收尾 → 实现 CPU 核数画像入口 → 辅助任务 1/2/4 核曲线与 `R_sys` → 必要时补 LLM/TTS 核数曲线 → 原生 Ubuntu 恢复 SM-A → 最终 CPU＋显存驻留＋SM 联合调度
- [ ] S0 无图像真实重叠静态放置实验
- [ ] 根据真实结果判断弹性驻留 Go / No-Go
- [ ] Go 后才实现 `PlacementPlan` 与新代价模型，再开展 S1 和正式策略评测
- [ ] **队列视界 predictor/planner（GATED）**：C3 校准、真实共置 smoke/矩阵、colocated 残差和 S0 Go/No-Go 完成前不实现。Go 后先用 calibration split 建立“画像插值 + 有界 EWMA 残差”，再冻结 queue horizon、转换安全余量与 minimum residency window；正式 evaluation 比较 Pareto-only、仅转换成本、queue-horizon-aware EdgeSched 与 oracle
- [ ] 同步论文、图表和必要的 C++ 契约

B1/B2 已确认 T0/T3 是条件式 Pareto，T0/T3 direct NVML v2 已完成 30/30 技术核验；Qwen3.5-9B 的 C2+C3 16 档（240 条）、KV 48/48、Forge direct NVML v1 5/5 与 LLM+TTS 正式共置 30/30 也已完成。9B 的 `G_safe/H_TTS` 是模型特定冻结值；K6/K4+Forge 均因真实 swap 失败，正式 H_D 矩阵停止。主实验现迁移到 Qwen3.5-4B-Q4_K_M + 真实 `mmproj`：artifact、环境能力与端点门代码已就绪并通过静态验证，但**真实 GPU 视觉 smoke 尚未运行**，因此当前没有任何 4B 性能、显存或识图结论，也不得开始 4B 的 C2/C3 与共置。SM-A 在 WSL2 中仍因 MPS binary 不存在暂定 NO_GO，当前不修改在线规划器、不伪造受控 SM 曲线。完成新模型 C3 与真实共置后再实现 CPU 核数画像入口；只有真实性和可行性门禁形成 Go 结论后，才实现 predictor 与 queue-horizon-aware transition planner。

---

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
