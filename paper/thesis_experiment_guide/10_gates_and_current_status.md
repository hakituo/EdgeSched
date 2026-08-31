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
- [ ] **当前 C3 收尾（还剩 2 个动作）**：
  1. [x] Forge b=1 direct NVML v1 已完成 5/5，逐窗新鲜度、swap 增长和核心归还门禁全部通过，`M_D=5484 MiB`；
  2. 用完整 16 档曲线、B2 direct v2 与 Forge direct v1 重新筛选最小共置候选；
  3. 共置 smoke + 每候选 5 轮，加入 colocated 残差后冻结 `G_safe/H_TTS/H_D`
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

B1/B2 已确认 T0/T3 是条件式 Pareto，T0/T3 direct NVML v2 已完成 30/30 技术核验；C2+C3 direct NVML 已完成 16 档（共 240 条）性能曲线，独立 KV 校准 48/48 与 Forge direct NVML v1 5/5 也已完成。SM-A 已在真实 WSL2 中因 MPS binary 不存在判定**暂定 NO_GO**——这是环境门禁导致的暂停而非永久取消，待原生 Ubuntu 恢复 SM-A 后再决定是否进入 SM-B/C/D/E；当前不修改在线规划器、不伪造受控 SM 曲线。当前用完整 16 档曲线重新筛选候选，随后运行 LLM+TTS 共置 smoke 与正式矩阵并冻结 `G_safe/H_TTS/H_D`；其后实现 CPU 核数画像入口，再跑辅助任务 1/2/4 核曲线与 `R_sys={0,1,2,4}`，必要时补 LLM/TTS 核数扩展曲线，最后在原生 Ubuntu 恢复 SM-A，形成 CPU＋显存驻留＋SM 联合调度。只有上述真实性和可行性门禁形成 Go 结论后，才实现 predictor 与 queue-horizon-aware transition planner；Pareto 过滤保留为目标配置选择规则，不单独作为算法创新点。

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
