# EdgeSched 论文与实验指南索引

**当前主线：S0（LLM→TTS）已完成协议冻结、启动策略对照与四配置正式成对比较，结论是收缩主张。** small 9B 的 C512 容量/可行性口径已收案；`s0_formal_20260917_v1`（20 次、19 通过）显示没有任何混合配置相对同 LLM 参考 `K24:T0` 形成有意义收益，驻留最满的一档反而最好，说明该场景下**容量不是瓶颈、TTS 后端吞吐才是**（逐句 RTF 全部大于 1，B2 隔离矩阵同样 0/15 低于 1）。因此按 Gate B No-Go 第 1 条不进入以收益为前提的 planner 实现，转为报告静态放置结论与实时性边界。已完成选型、16 档曲线、KV 校准、离线筛选、兼容性 smoke、K24 边界修正、五轮正式矩阵、S0 四配置静态放置、测量范围分层、启动策略小对照。合并产物 `small_9b_s0_colocated_20260908_v1.json` 冻结 `G_safe=915.015625 MiB` 与四候选 `H_TTS`（K0:T0、K24:T0/T3/T4）；K0:T4 全 CPU 以负面边界收案。K24 是已验证候选，不是最大可行档位的证明；不自动安排 K28/K30 或重跑已收案矩阵。见 [当前门禁](10_gates_and_current_status.md) 与 [E2 协议 `s0_protocol_v1`](07_experiment_stages_and_matrix.md#e2无图像的-llmtts-流水线)。

> **small 视觉画像补齐**：`20260907_171635` 五轮 5/5，swap 0；加载/预热、编码、回答后释放阶段成本已采集。独立进程画像不证明跨请求保温或编码后提前释放；见 [验收](../../docs/SMALL_9B_VISION_PROFILE_20260907.md)。

> **画像身份记录（2026-09-07）**：small 纯文本 C2 90/90、C3 150/150、KV 48/48 已核验，swap 0；该阶段版本为 `1a35ab89…`，后续批次以各自 manifest 为准。small 视觉 Answer 1/1 已通过；旧 large 五轮视觉画像仍属 `423fb3e5…`，不可混用。C512 阈值以后续合并冻结记录为准，不能外推到视觉或 C2048/C4096。见 [完整验收](../../docs/SMALL_9B_CURVES_20260907.md)。

**2026-09-17 进展（本地日期 UTC+8）**：完成 S0 正式比较的**测量范围分层**与**协议冻结**，补了启动策略小对照，并跑完四配置正式成对比较。测量范围拆两层：**调度主指标**（首句/末句完整波形就绪、句级排队与合成时间、应用层供给缺口、资源与输出契约）构成 Gate B 形式判据；**设备体验指标**（可信实际首播时间、设备 underflow）在设备时钟映射通过验证前一律 `pending_device_validation`，只保留 callback 代理语义，不冒充实际播放时刻。这是对现有门禁的**调整建议，需先确认**。`s0_protocol_v1` 已冻结输入与生成、分句、启动策略、重复次数、输出差异与失败处理，真源在 `s0_pipeline/config.py`。启动策略小对照 `s0_startup_20260917_v1`（K24:T0）：首句即播 8 段供给缺口 / 窗口内 44.1%，全量预缓冲 0 段 / 0%，代价是首播等待 12.63 s → 68.28 s（proxy），两者输出完全一致。

**2026-09-17 正式成对比较结论（收缩主张）**：`s0_formal_20260917_v1` 四配置 × 5 轮 = 20 次、19 通过。全部混合候选在首句波形就绪与逐句 RTF 上都不优于同 LLM 参考 `K24:T0`，而 `K24:T0`（驻留最满）三主指标同时最优——容量不是瓶颈、TTS 后端吞吐才是（逐句 RTF 1.75–1.95，全部大于 1，与 B2 隔离矩阵的 T0 中位数 1.81 一致）。唯一表层改善来自文本与句数都不同的 `K0:T0`（缺口段数 7 vs 8），其窗口内缺口占比反而更差（0.543 vs 0.423）。按 Gate B No-Go 第 1 条**收缩主张**：不进入以收益为前提的 planner 实现，改为报告静态放置结论与后端实时性边界。`K24:T4`（TTS 全 CPU）r2 因 swap 增长 63.1 MiB 失败，计入分母。

**2026-09-12 进展（本地日期 UTC+8；同段 UTC 为 2026-09-11）**：已在**代码侧工作区**复核源码与产物，并完成 S0 输出契约修正与四配置小规模真实复验。核对结论：9 月 9 日 pilot 运行时 `s0_child` 的 `--system-prompt` 默认为空（配置事实；统一 prompt 是 `ebee12a` 才加入、从未实跑），但仅加系统提示仍会输出分析正文，提示与模板生成前缀需共同处理，本轮采用空 think 前缀后所测输入通过契约；播放路径实际已实现未实跑，不是未接入；旧批次重叠/背压只有汇总报告、缺持久化事件。本轮统一 prompt 由 parent 下传并落盘，用模型自带模板渲染后追加空 think 块关闭 thinking，格式适配与契约判据只判失败不删除并全量落盘，事件由只记 `event_count` 修正为全量持久化。真实运行 `s0_pipeline_20260912_contract_v1`（四档仅输出契约失败）与 `s0_pipeline_20260912_contract_v3`（四档全通过，含事件时间线）。播放方面已完成测量口径修正（缺供给 callback 次数 vs 独立断流段、阶段划分、生产者/设备下标分离）、播放器 smoke 与一次启用播放的 K24:T0（`s0_pipeline_20260912_playback_tr32d4d4e`，提交 `32d4d4e`）：9 句全部实际播放，10 段缺音（startup 14.7 s / 句间 8 段 27.8 s / tail 0.08 s），窗口内占比 45.8%，设备 underflow 0。注入 1.0 s 间隙在 Windows 与 WSL2 均测得 0.987 s、占比 0.1979，测量可跨平台复现。设备时钟映射两端均未通过验证，实际播放时刻仍为 callback 代理，零 underrun 保持 pending。数值及解释边界见第 07 模块 E2，状态见第 10 模块。

**后续顺序**：测量范围分层与协议冻结（2026-09-17 完成）→ 启动策略小对照（完成）→ 四配置 × 5 轮正式成对比较（完成，20 次 19 通过）→ 离线复算并回答三个冻结问题（完成，结论为收缩主张）。**当前不进入 `PlacementPlan`/predictor/planner 实现**；若要重新开启，必须先给出一个前提成立且可运行的新场景（例如长上下文 C2048/C4096 使 LLM 自身无法放满、或图像/视觉阶段的可运行配置），并按同一 Gate B 重新判读。设备时钟映射未通过前不得报可信实际首播时间或设备 underflow；`H_D` 仍 pending，保留 16G Forge 共置负面边界；SM-B/C/D/E 等原生 Ubuntu + MPS，BGE 缓存补测与 D3 并发为辅助线。

**用途**：把研究设计、实验协议和当前状态拆成可按任务选择读取的模块，避免 AI 或维护者每次加载整份长文档。

**2026-09-07 视觉生命周期修订**：01 明确 V+L 的依赖，02 增加 q_V/r_V，03 分开 warm/编码峰值与端到端延迟，04 计入视觉加载/释放及队列保温代价；公式稿和新版提纲同步。当前只证明单轮回答后释放，提前释放/保温策略仍待实现验证。

## 首读顺序

新会话默认只读：

1. 本索引；
2. [当前门禁与任务状态](10_gates_and_current_status.md)；
3. 与当前任务直接相关的一个模块；
4. 需要执行实验时，再读 `docs/ELASTIC_RESIDENCY_EXPERIMENT_EXECUTION_PLAN_20260828.md` 和 `docs/ELASTIC_RESIDENCY_RESULT_CATALOG_20260830.md`。

不要为了回答单个实验问题一次性加载全部模块。

## 模块地图

| 模块 | 何时读取 | 内容 |
|---|---|---|
| [01 研究定位、工作流与任务抽象](01_research_scope_and_workloads.md) | 改研究问题、术语或贡献边界 | 目标场景、语音/图像工作流、设备亲和性、驻留与 GPU 计算份额能力 |
| [02 系统状态与运行模式](02_state_and_runtime_modes.md) | 改状态机或决策动作 | 调度状态、候选驻留档位、S0/S1/S2 |
| [03 画像与资源约束](03_profiles_and_constraints.md) | 改 profiler、容量公式或回退约束 | LLM/TTS/Diffusion 画像、显存/CPU/SM 约束、分层目标 |
| [04 代价与规划契约](04_cost_and_plan_contract.md) | 改代价模型、转换算法、预测器或 PlacementPlan | 队列视界、转换摊销、画像插值与在线残差、迟滞、泛化边界、规划输出 |
| [05 CPU 与 GPU 分配方法](05_cpu_and_gpu_allocation.md) | 做核数或 SM 实验 | CPU 扩展曲线、最低核心预算、拓扑消融、SM/流水线平衡 |
| [06 实现差距与路线](06_implementation_and_roadmap.md) | 判断代码是否已实现 | 当前实现审计、P0–P5 路线、SM-A 能力探针 |
| [07 实验阶段与矩阵](07_experiment_stages_and_matrix.md) | 设计或审查正式实验 | 探索/冻结/正式阶段、E0–E9 |
| [08 指标与复现](08_metrics_and_reproducibility.md) | 改统计、输入或证据链 | 统一指标、轮数、顺序、固定输入、manifest |
| [09 证据与论文组织](09_evidence_and_paper_plan.md) | 写论文或决定数据能否使用 | 当前数据边界、必须重跑项、章节与图表规划、局限性 |
| [10 门禁与当前状态](10_gates_and_current_status.md) | 每个新会话必读 | Go/No-Go、完成项、暂停项、下一批实验、完成标准 |

## 历史模型与批次摘要（2026-09-06）

以下为历史身份和证据边界，不是当前 small C512 的阈值或下一步任务；当前主线以上方摘要和第 10 模块为准。

- Qwen3.5-9B 的 C2+C3 direct NVML 已完成 16 个 LLM 驻留档位，共 240 条性能记录；KV 容量与语义校准已完成 48/48 条，均保留为 9B TEXT_ONLY 模型特定证据，不能当作当前视觉阈值。
- Qwen3.5-4B 的 C2 90/90、C3 150/150 与纯文本 KV 校准 48/48 已完成并通过；C2+C3 在每个 context 合并为完整 16 档。C2/C3 artifact 保存的是短 commit `a35d4f5`，可唯一解析到完整提交 `a35d4f5f9b17b01b5d90bd6fd0db794cfdb54eb2`，晋升前仍须补独立 provenance 审计记录；KV artifact 已保存完整提交。
- B2 已筛出 T0/T3，direct NVML v2 隔离矩阵已完成 30/30 技术核验。
- Forge b=1 direct NVML：20260831 批次 `M_D=5484 MiB`（9B screen 用）；20260906 同日复测 `M_D=5273.21 MiB`，为当前 4B screen 输入。
- Qwen3.5-9B 的 LLM+TTS 正式共置已完成 30/30，并冻结该模型口径的 `G_safe=501.282 MiB` 与 `H_TTS`。
- 4B 文本共置正式矩阵已完成并冻结 4B 文本口径：`llm_tts_colocated_4b_text_only_20260904_direct_nvml_v1` 30/30 + 离线复算 `c3_h_4b_tts_colocated_20260906_v1.json`，G_safe=501.282 MiB 校准维持、H_TTS 在 T0/T3×512/2048/4096 全部 FULL。
- 4B `C4096/FULL+T0` 真实共置 smoke 已 1/1 技术通过：`P_colocated=6406.699 MiB`、物理余量 1744.301 MiB、swap 0、正残差 0；该旧 smoke 未在 JSON 中持久化 WAV 哈希，只作正式矩阵前链路证据。入口已补非空 WAV 大小/SHA-256 硬门，正式矩阵须使用新 commit 与 run id。
- Qwen3.5-9B 的 512/K6+Forge 与探索性 512/K4+Forge smoke 均只有 swap 门禁失败，增长分别约 290.44 MiB 和 2070.32 MiB；正式 H_D 矩阵已停止，`H_D` 未冻结。
- 后续主实验 LLM 使用 Qwen3.5-9B-q4_k_m.gguf + Qwen3.5-9B-f16.mmproj；用户实跑 `v1_vision_9b_probe_20260906_201923` FULL 文本/视觉端点 2/2 通过，真实视觉嵌入 `[4096,196,1]`，swap 0。4B 保留为跨尺度对照；9B 五轮视觉画像 5/5 已通过、回答补测已完成（漏黑边）；先离线筛容量与阶段切换候选，视觉与 TTS 共置尚未验证，详见 [复核记录](../../docs/VISION_9B_PROBE_20260906.md)。
- 多模态端点门 `v1_vision_endpoint_20260831_direct_nvml_v3` 已通过 4/4 单元；VISION_WARM 多轮画像 `v1_vision_endpoint_4b_warm_profile_20260906_direct_nvml_v1` 已完成 5/5（warm GPU 增量 mean 761.2 MiB，含 warmup 预留 223.30 MiB）。4B `H_D` 已按 16G 论文环境以负面边界证据收案：Forge 同日复测 `M_D=5273.21 MiB` 后 FULL 因 GPU 容量不可行，16G 下首个 K16@C512 即 swap 增长 405.15 MiB 失败即停，20G 同代码归因诊断 3/3 通过证明根因为主机内存容量；与 9B K6/K4 构成跨尺度负面证据（8GB GPU + 16GB 主机下 LLM+Diffusion 无任何可共置档位）。
- SM-A 在 WSL2 中因 MPS binary 不存在暂定 NO_GO；待原生 Ubuntu 恢复验证，不伪造 SM-B/C 曲线。
- CPU 物理核拓扑、affinity 与租约基础设施已完成；`auxiliary_core_budget.py` 已实现；用户完成 R_sys=2 曲线 40/40，BGE 候选缓存模式待用户补测；D3 主流水线并发验收尚未完成。
- `paper/experiment_data/01_paper_main/` 当前没有已晋升的新版正文数据。

## 真源与更新规则

发生冲突时按以下顺序判断：

1. 当前源码、测试与真实运行日志；
2. 实验 `manifest.json`、`iterations.jsonl`、`summary.json`；
3. `docs/ELASTIC_RESIDENCY_RESULT_CATALOG_20260830.md`；
4. `docs/ELASTIC_RESIDENCY_EXPERIMENT_EXECUTION_PLAN_20260828.md`；
5. 本目录中的研究设计与状态摘要；
6. `paper/archive/` 和 `docs/archive/` 仅供追溯。

运行结果变化时，优先更新第 10 模块和结果目录；研究模型变化时，更新对应主题模块。不要把短期 Git HEAD、临时测试计数或尚未产生的 run id 固化为长期研究结论。
