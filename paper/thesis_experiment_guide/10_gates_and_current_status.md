# Go/No-Go 门禁与当前任务状态

**2026-09-12 S0 复验更新（本地日期 UTC+8；同段 UTC 为 2026-09-11）**：已在**代码侧工作区**复核原始产物与源码，并修正三处旧表述——(1) `e5f9a43` 时代的 pilot 运行时 `s0_child` 的 `--system-prompt` 默认为空（配置事实），但仅加系统提示仍会输出分析正文，提示与模板生成前缀需共同处理，本轮采用空 think 前缀后**所测输入**通过契约，不推广为「该模型不再输出思考正文」；(2) 旧批次重叠/背压只有汇总报告，`events.jsonl` 为空、缺事件级证据，`v3` 事件不能反向补证；(3) 播放路径并非未接入，而是已实现未实跑。本轮完成统一 prompt + 空 think 块 + 确定性格式适配 + 契约判据（只判失败不删除、全量落盘）与事件持久化，并以同一协议对四配置各做一次真实复验：`s0_pipeline_20260912_contract_v1` 四档仅 `output_contract_ok` 失败（思考正文泄漏被正确判定并保留），`s0_pipeline_20260912_contract_v3` 四档全通过且含完整事件时间线；`v3` 运行代码未提交，已在产物目录保存 dirty diff 与逐文件哈希快照，不足以还原时只作探索级定位。播放仍无设备级证据（WSL2 PortAudio 无法打开输出流），零 underrun 与实际播放保持 pending。详见第 07 模块 E2；本次不是正式性能或弹性驻留 Go。

**2026-09-09 pilot 记录（已被上方复核结论取代的状态描述）**：K24:T0、K24:T3、K0:T0、K24:T4 的生成→分句→排队→合成 pilot 四配置通过，报告 `all_passed=true`，生成×合成重叠、背压及 swap 门禁通过。报告的提交标识为 `e5f9a43`。该批的产物已在本工作区复核（见第 07 模块 E2）。

small 9B 离线筛选 `20260907_204516_653987` 已复算一致：T0/K20、T3/K30 仅为容量候选，长上下文余量不足 16 MiB；实测修正为 C512 已验证候选 K24（T3 加载期最坏，非最大可行档位证明），K28/K30 不再自动安排。**S0 静态放置基线已完成并冻结**：正式 5 轮 ×4 配置（K24:T0、K24:T3、K0:T0、K24:T4）共 20 条全部通过，K0:T4 全 CPU 降级负面边界收案；合并复算 `small_9b_s0_colocated_20260908_v1.json` 冻结 `G_safe=915.015625 MiB` 与四候选 `H_TTS`（K0:T0、K24:T0/T3/T4）。C512 的 LLM+TTS 容量/可行性口径就此收案；`H_D` 仍 pending，实现准入 Go/No-Go 前仅剩 S0 端到端流水线证据，见 [验收与命令](../../docs/SMALL_9B_NEXT_COMBINATIONS_20260907.md)。产物名称保留原批次标识，不表示本次文档同步重新运行或审计了实验。

> **small 视觉画像补齐**：`20260907_171635` 五轮 5/5，swap 0；加载/预热、编码、回答后释放阶段成本已采集。独立进程画像不证明跨请求保温或编码后提前释放；见 [验收](../../docs/SMALL_9B_VISION_PROFILE_20260907.md)。

> **画像身份记录（2026-09-07）**：small 纯文本 C2 90/90、C3 150/150、KV 48/48 已核验，swap 0；该阶段版本为 `1a35ab89…`，后续批次以各自 manifest 为准。small 视觉 Answer 1/1 已通过；旧 large 五轮视觉画像仍属 `423fb3e5…`，不可混用。C512 的 `G_safe/H_TTS` 已由后续合并产物冻结，视觉及长上下文不能沿用；SM-A 等原生 Ubuntu。见 [完整验收](../../docs/SMALL_9B_CURVES_20260907.md)。

> 导航：[总索引](README.md) · 兼容入口：[THESIS_EXPERIMENT_GUIDE_DRAFT.md](../THESIS_EXPERIMENT_GUIDE_DRAFT.md)

## 19. Go / No-Go 检查点

**组合推进顺序**：先 LLM+TTS 基础放置/兼容性，再分开验证 LLM+TTS+V 与 LLM+TTS+Diffusion。
四组件同时驻留/计算不是强制收尾项，只有工作流需要且容量/主机门允许才进入候选。
离线入口与 provisional 候选见 [下一步计划](../../docs/SMALL_9B_NEXT_COMBINATIONS_20260907.md)；Diffusion 16G 负面边界继续保留，SM 管理等待原生 Ubuntu。

### 视觉生命周期的新增验收边界（2026-09-07）

已修订工作流、状态和代价文档：V 是可独立加载/释放的视觉组件，输出依赖 L，不能冒充独立 VL 模型或直接继承 Diffusion 逐出策略。
已具备 small 纯文本 288 条与视觉单轮回答证据；small 视觉加载/编码/释放的五轮阶段画像已 5/5 通过；仍待补常驻、回答后释放、已知队列保温的持续会话公平对照。
端到端比较计入真实加载/释放和等待；编码后提前释放须先验证同步、表示所有权与 KV/后续生成正确性，再进入候选。
本次仅文档与公式稿修订，未实现新调度策略、未运行新实验，也不要求重跑已完成的 288 条文本/KV。

### Gate A：真实 TTS

Go：

- CPU/GPU 路径可重复运行；
- 模型保持常驻；
- chunk 指标完整；
- 无静默 Mock fallback。

### Gate B：混合驻留的实现准入与动态验收

Go：

- 至少一个声明为 `PARTITIONABLE` 的阶段存在可稳定运行且可复现的离散混合配置；
- 该配置的拆分粒度、组件/层映射和转换方式均写入 manifest；
- 真实 S0 句级流水线满足预先冻结的正确性、资源、首句就绪/实际播放截止与播放连续性约束；
- 至少一个混合配置相对本轮可行的非混合基线形成有意义的端到端 Pareto 收益，不以击败已不可用的 K0:T4 全 CPU 为充分条件；
- 收益不是由输出长度差异、不同切句、额外预缓冲或漏算必要加载造成；比较容差与主要指标在正式评价前冻结。

本门禁的 Go 仅准许实现 `PlacementPlan`、代价模型与 predictor/planner，不表示动态弹性策略已通过。动态验收仍须验证状态正确性、转换成本、已到达队列内的真实 payback，并与最佳静态放置及二元完整迁移比较。当前 S0 协议见 [E2](07_experiment_stages_and_matrix.md#e2无图像的-llmtts-流水线)；截止阈值尚待确认，不登记 Go。

No-Go：

- 所有混合候选均违反冻结约束或被可行基线支配，则不进入以收益为前提的 planner 实现；采集故障或样本不足只记证据不足，不当作算法 No-Go。
- 后续动态实验若显示转换和干扰抵消收益，则收缩动态规划主张，保留已成立的静态放置结论。
- 仅有限回复通过预缓冲实现零 underrun 时，只报告该有限回复能力及启动代价，不宣称持续实时通过。

门禁按工作流适用：S0 使用已冻结的 `G_safe/H_TTS` 和四候选；`H_D` pending 继续阻止未经验证的 Forge 共置候选进入规划，但不阻塞 S0。视觉和长上下文另行验收；Gate C 的受控 SM 门禁不阻塞当前不受控 S0 验证。

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
- [x] **Qwen3.5-9B 历史 TEXT_ONLY 校准与 Forge 停止记录**：
  1. [x] Forge b=1 direct NVML v1 已完成 5/5，逐窗新鲜度、swap 增长和核心归还门禁全部通过，`M_D=5484 MiB`；
  2. [x] 完整 16 档 + B2 direct v2 + Forge direct v1 已生成 `c3_h_direct_nvml_20260831_v1.json`：T0 边界为 512/K24、2048/K20、4096/K20，T3 三个 context 均筛到 FULL；这些仍是容量候选；
  3. [x] C4096/K20+T0 artifact smoke 已 1/1 技术通过：`P_colocated=6686.793 MiB`，T0 共置增量 2252 MiB，正残差 0，计入 501.282 MiB 安全余量后容量可行，0 failure/fallback；
  4. [x] `llm_tts_colocated_candidates_20260831_direct_nvml_v1` 已完成 6 候选×5 轮，共 30/30、0 failure/fallback；全部真实后端、direct NVML，新鲜度和 swap 门禁通过，colocated 正残差最大值为 0；
  5. [x] `c3_h_tts_colocated_20260831_v1.json` 已离线复算并冻结 **9B 模型口径**的 `G_safe=501.282 MiB` 与按 context 的 `H_TTS`：T0 为 512/K24、2048/K20、4096/K20，T3 三个 context 均为 FULL；这些值仍仅代表原 artifact 的 TEXT_ONLY 口径，不能代表当前 9B VISION_WARM；
  6. [x] 512/K6+Forge 真实 smoke 的后端、PNG、direct NVML、新鲜度、KV、GPU 容量和 provenance 均有效，但 swap 增长约 290.44 MiB，整体失败；
  7. [x] 探索性 512/K4+Forge smoke 同样只有 swap 门失败，增长约 2070.32 MiB；降低 GPU 层数释放了显存，却显著增加主机换页，因此不再继续 K2，也不运行 9B 正式 15 条矩阵；
  8. [x] 9B 的 `H_D` 未冻结；两次失败作为“仅看显存会选错目标档”的负面证据保留，不得晋升为 9B 共置可行结论。
- [ ] **历史 4B 阶段记录（主模型选型已由 2026-09-06 的 9B+视觉决定取代）**：
  1. [x] 当时的研究口径：统一模型必须真实执行文本与图片理解，`mmproj` 是正式 session-warm 视觉组件，不是静态填充；9B 保留为压力/NO_GO 对照；
  2. [x] 主 GGUF 与 `mmproj` 已在 WSL2 固定并核验：`Qwen3.5-4B-Q4_K_M.gguf`（`qwen35`、32 层、`context_length=262144`、`embedding_length=2560`、`file_type=15`、426 张量、2740937888 B、SHA-256 `00fe7986…ef11a4`）与 `mmproj-BF16.gguf`（`clip`/`mmproj`、`qwen3vl_merger`、`has_vision_encoder=true`、24 层视觉块、`image_size=768`、`patch=16`、`merge=2`、`projection_dim=2560`、298 张量、675569344 B、SHA-256 `302b92d5…21f3a6`）。对应关系判据为 `projection_dim == embedding_length` 加 `general.basename`/`general.base_model.0.*` 一致，不靠文件名推断；当时旧 9B GGUF 已从 WSL 删除；2026-09-06 的新 9B 端点从 D: 挂载路径读取对应 GGUF/mmproj，历史结果保持原身份；
  3. [x] 环境能力已核验：`llama-cpp-python==0.3.35` 随包提供 `libmtmd.so` 与 `llama_cpp/mtmd_cpp.py`，`llama_chat_format` 已含通用 `MTMDChatHandler`，`mtmd_get_cap_from_file` 对该 `mmproj` 报告 `inp_vision=true`/`inp_audio=false`，media marker 为 `<__media__>`；**无需新增或修改依赖**；
  4. [x] 已实现多模态端点门代码链路 `experiments/elastic_residency/vision/`：artifact 门禁、确定性 448×448 固定图片（`pixel_payload_sha256=a1514034…43cb00`，期望 196 个视觉 token）、libmtmd projector warm/执行/释放、`TEXT_ONLY` / `TEXT_ONLY_CHAT` / `VISION_WARM_GPU` / `VISION_WARM_CPU` 四状态与 K0/FULL 端点 smoke 编排；Windows 静态与假后端测试 72 项通过，父进程门禁失败路径已在 WSL2 用合成 artifact 实跑；
  5. [x] **已通过**：WSL2 真实 GPU 视觉端点门，`run_id=v1_vision_endpoint_20260831_direct_nvml_v3`、commit `c9a75da`，四单元（`K0/FULL × TEXT_ONLY/VISION_WARM_GPU`，512+128 上下文）全部 `passed=true`、0 失败、swap 0、direct NVML 且无 GPU CLI fallback。projector 在 GPU 上真实执行：448×448 图片编码出 `[2560, 196, 1]` 嵌入，视觉 token 196（与 patch/merge 期望一致），M-RoPE 下塌缩为 14 个序列位置，KV 位置 167/167 精确匹配；模型文字输出正确描述固定图片（白底、居中红圆、左上蓝方块、右下绿方块、黑边框）。projector warm/释放 0.28–0.44 s / 0.008–0.011 s，warm GPU 增量 710–870 MiB、RSS +55.9 MB。FULL 档 decode 86–90 tok/s、prefill 文本 2579.9 / 视觉 119.6 tok/s；K0 档 decode 6.66–6.91 tok/s。
  6. [x] 4B TEXT_ONLY C2 90/90、C3 150/150 与纯文本 KV 校准 48/48 已完成并通过；C2+C3 在 512/2048/4096 三个 context 下均合并为完整 16 档，KV buffer 为 24/72/136 MiB。C2/C3 artifact 的短 commit `a35d4f5` 已唯一解析到完整提交 `a35d4f5f9b17b01b5d90bd6fd0db794cfdb54eb2`，晋升前仍须保留独立 provenance 审计记录；KV artifact 使用完整提交；
  7. [x] 重算并冻结 4B 的 `G_safe/H_TTS`；`H_D` 按 16G 论文环境以负面边界证据收案：`llm_tts_colocated_4b_text_only_20260904_direct_nvml_v1` 30/30 + 离线复算 `c3_h_4b_tts_colocated_20260906_v1.json` 冻结 G_safe=501.282 MiB 与 H_TTS（T0/T3×512/2048/4096 全部 FULL）；VISION_WARM 多轮画像 `v1_vision_endpoint_4b_warm_profile_20260906_direct_nvml_v1` 已完成 5/5。Forge 同日复测 `forge_b1_reprofile_20260906_direct_nvml_v1`（`M_D=5273.21 MiB`）注入新 screen `c3_h_4b_tts_direct_nvml_20260906_v1.json` 后 FULL 因 GPU 容量不可行，H_D 候选为 512/K16、2048/K14、4096/K14；正式矩阵 `llm_forge_colocated_4b_20260906_direct_nvml_v1`（干净 `a39778f`）首个 K16@C512 唯一失败门为 swap 增长 405.15 MiB，失败即停（1/15），其余门禁全有效；归因诊断 `llm_forge_colocated_4b_20260906_mem20g_diag_v1` 在 `.wslconfig memory=20GB` 下同代码/同候选/同参数 3/3 通过（swap 0.0–0.2 MiB），证明根因为 16G 主机内存容量而非 K 档固有换页。4B `H_D` 不冻结，与 9B K6/K4 构成跨尺度负面证据：8GB GPU + 16GB 主机下 LLM+Diffusion 无任何可共置档位；
  8. [ ] 迁移完成前不得把任何 9B 阈值改名为 4B 结论，不得把 `mmproj` 文件大小加到旧 9B 曲线上冒充测量，也不得晋升到 `paper/experiment_data/01_paper_main/`。
- [x] **SM-A：GPU 计算份额能力探针**：真实 WSL 记录 `sm_a_20260830_direct_nvml_v1` 已完成；MPS binary 在 WSL2 不存在，结论**暂定 NO_GO**。注意：这是环境能力门禁导致的「暂停」，不是永久取消——待具备可用的原生 Linux 启动环境后重做 SM-A，再决定是否进入 SM-B/C/D/E
- [ ] **SM-B（PAUSED）：TTS T0/T3 隔离 SM 曲线**：因 SM-A 暂定 NO_GO 按门禁暂停，不运行、不伪造受控 SM 曲线；待 SM-A 在原生 Ubuntu 恢复后重排
- [ ] **SM-C（PAUSED）：Diffusion b=1 隔离 SM 曲线**：因 SM-A 暂定 NO_GO 按门禁暂停，不运行、不伪造受控 SM 曲线；待 SM-A 在原生 Ubuntu 恢复后重排
- [ ] **SM-D：LLM→TTS 平衡实验**：比较不受控共享、显存共驻留但计算串行、静态 SM 分配，并记录第一句、TTFA、chunk deadline 与真实 underrun
- [ ] **SM-E：TTS→Diffusion 共运行实验**：先固定 LLM 在 CPU，求 TTS 最小可行 SM 数，将剩余 SM 给 Diffusion；前序形成 Pareto 后才测试三阶段 GPU 共享
- [ ] **CPU 核数实验（D1 已有用户实跑；D3 待完成）**：`d1_auxiliary_manual_v1` 完成 R_sys=2 下 40/40 次、240 个事件，离线复算通过，swap 增长为零；JSON RAG 和 Whisper 的最低预算为 1 核，BGE 全量重编码 1/2/4 核均未满足 150 ms。新增独立 `bge_cached` 候选缓存模式，准备成本与请求耗时分离，静态验证通过，等待用户执行 15 次补测。四档 R_sys 完整矩阵、主流水线并发/播放验收和 LLM/TTS 扩展曲线尚未完成。详情见 `docs/AUXILIARY_CORE_BUDGET.md`。
- [x] **9B+视觉主模型端点**：`v1_vision_9b_probe_20260906_201923`，FULL × TEXT_ONLY_CHAT/VISION_WARM_GPU 各 1 轮，2/2 通过；196 视觉 tokens、非空 4096 维嵌入、swap 0。输出正确识别蓝方块与红圆但在 128 tokens 截断；仅完成链路验收。
- [x] **9B 五轮视觉画像完成**：`v1_vision_9b_profile_20260906_204939`，5/5、0 failure、swap 0；整卡峰值 7542.47 MiB 已包含 n_ctx=640 的 24 MiB GPU KV，余量 608.53 MiB 尚未扣安全余量。
- [x] **9B 回答补测完成**：`v1_vision_9b_answer_20260906_204651`，1/1 技术通过，三种主要图形/颜色/位置及白底正确；漏黑边，不能登记全部细节质量通过。
- [x] **small 离线 GPU 容量候选已验收**：`small_stage_screen_20260907_204516_653987.json` 复算一致；完整阶段切换策略仍待验证；编码后释放 projector 并继续 decode 尚未实测，视觉与 TTS 共置及长上下文未验收，不能直接用旧 TEXT_ONLY H_TTS 替代。
- [x] **small C512 LLM/TTS 共置兼容性 smoke**：K18/K20/K24 × T0/T3 共 6/6 `passed=true`、0 failure、swap 0、真实后端无 fallback、音频 WAV 跨批次 SHA 一致；T3 最坏时刻在加载期（高于推理期 106–424 MiB），最坏时刻口径下候选收敛至已验证的 K24（非最大可行档位证明），K28/K30 不再自动安排。每档每配置仅 1 轮，属 smoke/pilot 而非正式重复。
- [x] **small C512/K24 五轮正式共置矩阵与阈值冻结**：`small_9b_tts_compat_20260907_225155_3728828` 10/10、0 failure、swap 0、direct NVML、真实后端无 fallback、Git provenance 干净（`e0f1779`）；五轮零漂移（idle/B_LLM/P 全一致，时间线峰值漂移仅 2 MiB），WAV 五轮 SHA 一致。离线复算 `small_9b_tts_colocated_20260907_225155_v1.json` 冻结 **C512 口径** `G_safe=915.015625 MiB`（含首次非零的共置正残差 168.90 MiB）与 `H_TTS`（T0/T3 均为 K24，安全余量下界 568.89/704.89 MiB）。复算入口新增 `small_screen_view.py` 与 `--screen-schema small`，候选判据为“已画像档位内”，两个候选均标记 `measured_beyond_screen_boundary=true`。边界：只验证 K24 一档（非上界证明）、只覆盖 C512、T3 按加载期最坏时刻裁决、`H_D` 仍 pending。
- [x] **S0 无图像真实重叠静态放置实验（容量/可行性口径）**：正式 5 轮 ×4 配置（K24:T0、K24:T3、K0:T0、K24:T4）共 20 条全部通过且五轮漂移极小；K0:T4 全 CPU 降级以负面边界收案（正式矩阵 r2 swap 增长 53.3 MiB 失败，加 RTF 6.2 与 RSS 11.4 GiB，三重论据支撑“全 CPU 不可用”）；新增 T4 全 CPU 放置与纯 CPU 门禁豁免。多批次合并复算 `small_9b_s0_colocated_20260908_v1.json`（`colocated_batch_merge.py` + `--colocated-summaries`）冻结 `G_safe` 与四候选 `H_TTS`；关键发现：K24 下最坏时刻由 TTS 权重加载过 GPU 决定（T0/T3/T4 加载期峰值 7619/7589/7611 MiB 几乎相同）。详见 [S0 证据](../../docs/SMALL_9B_NEXT_COMBINATIONS_20260907.md) 与 [放置复核](../../docs/LLM_TTS_PLACEMENT_REVIEW_20260906.md)。
- [ ] S0 端到端协议确认：第 07 模块已补草案，建议 LLM 流式输出 → 固定分句 → 逐句完整波形 → 连续播放；启动缓冲、首句就绪/实际播放截止、underrun 采集分辨率及比较容差待冻结。
- [x] S0 生成与合成链路 pilot（代码侧报告通过；2026-09-12 已在代码侧工作区复核产物）：四配置重叠、背压及 swap 门禁通过，但**只有汇总报告、没有持久化事件**（该批 `events.jsonl` 为 0 字节），重叠与背压记为「汇总有记录、缺事件级证据」；第 07 模块记录本批数值和异常输入，不作为正式性能结论。
- [x] **S0 输出契约修正与四配置小规模复验（2026-09-12，本地日期 UTC+8）**：先在代码侧工作区核对源码与产物——统一 prompt 是 `ebee12a` 才加入、pilot 运行时为空（配置事实），但仅加系统提示仍输出分析正文，故提示与模板生成前缀共同处理；播放路径实为「已接入未实跑」；旧批次重叠/背压缺持久化事件。修正后统一 prompt 由 parent 下传并落盘，用模型自带模板渲染后追加空 think 块，格式适配与契约判据全部放在工作负载适配层，思考正文/空输出只判失败不删除，并保存原文、处理后文本、句子映射与失败原因；事件由只记 `event_count` 修正为全量落盘。真实复验 `s0_pipeline_20260912_contract_v1`（四档仅 `output_contract_ok` 失败，证明契约可判定）→ `v3`（四档全通过，含完整事件时间线；运行代码未提交，产物目录保存 dirty diff 与逐文件哈希）。所测输入通过契约，不推广为模型不再泄漏思考正文。仍未启用播放，不登记 Go。完整记录见第 07 模块 E2。
- [x] **S0 播放测量 smoke（2026-09-12）**：新增 `playback_smoke.py`，判定拆成 `measurement_ok`（测量可信）与 `continuity_ok`（零断流，不参与通过）。Windows 真实设备与 WSL2 各跑通，注入 1.0 s 间隙测得 1.029 s 缺音，证明断流可测；`device_underflow_count` 恒为 0，确认「供给不足」与「设备 underflow」是两个口径。WSL2 设备时钟与 host monotonic 不同源（偏移约 −1.79e18 ns），已加合理上界校验并降级为 `proxy`。这不是真实流水线实验。
- [x] **启用播放的 K24:T0 真实运行**：`s0_pipeline_20260912_playback_k24t0`，9 句全部提交并播放，消费 76.5 s（音频 32.9 s），starve 1023 次 / 43.5 s，`device_underflow_count=0`，输出契约通过。**测量能力已验证、性能 SLO 未通过**：断流由合成 RTF≈2.2 > 1 直接解释，不声称零 underrun，不声称实际播放时刻可信（proxy）。见第 07 模块 E2。
- [ ] S0 播放链路验收（剩余）：设备时钟与 host monotonic 同源（才能测实际播放时刻）、断流到句的直接归属（当前 starve `chunk_id` 为 `None`，需按 `sample_position` 与 `playback_submit` 样本区间离线重建）、启动缓冲策略与「有限回复预缓冲能否零 underrun」及其启动代价。
- [ ] S0 正式端到端流水线证据：输出契约与播放验收完成并冻结协议后，独立正式成对测量首句就绪、实际播放、chunk deadline 与真实 underrun；不重复已收案容量矩阵，尚不登记实现准入 Go。
- [ ] 根据真实结果判断弹性驻留 Go / No-Go
- [ ] Go 后才实现 `PlacementPlan` 与新代价模型，再开展 S1 和正式策略评测
- [ ] **队列视界 predictor/planner（GATED）**：C3 校准、真实共置 smoke/矩阵、colocated 残差和 S0 Go/No-Go 完成前不实现。Go 后先用 calibration split 建立“画像插值 + 有界 EWMA 残差”，再冻结 queue horizon、转换安全余量与 minimum residency window；正式 evaluation 比较 Pareto-only、仅转换成本、queue-horizon-aware EdgeSched 与 oracle
- [ ] 同步论文、图表和必要的 C++ 契约

B1/B2 已确认 T0/T3 是条件式 Pareto，T0/T3 direct NVML v2 已完成 30/30 技术核验；Qwen3.5-9B 的 C2+C3 16 档（240 条）、KV 48/48、Forge direct NVML v1 5/5 与 LLM+TTS 正式共置 30/30 也已完成。9B 的 `G_safe/H_TTS` 是模型特定冻结值；K6/K4+Forge 均因真实 swap 失败，正式 H_D 矩阵停止。作为跨尺度对照，Qwen3.5-4B + 真实 `mmproj` 的多模态端点门、TEXT_ONLY C2 90/90、C3 150/150、KV 48/48、LLM+TTS 正式共置 30/30 与 VISION_WARM 画像 5/5 已通过，4B 文本口径 `G_safe/H_TTS` 已冻结。图像侧 `H_D` 已按 16G 论文环境以负面边界证据收案：FULL 被 GPU 容量排除、K16/K14 档被 16G 主机内存排除（swap 405.15 MiB），20G 同代码诊断 3/3 通过证明根因为主机内存容量；与 9B K6/K4 构成跨尺度负面证据，直接支撑图像期迁移/逐出设计，`.wslconfig` 诊断用 `memory=20GB` 须恢复默认。SM-A 在 WSL2 中仍因 MPS binary 不存在暂定 NO_GO，当前不修改在线规划器、不伪造受控 SM 曲线。C3 收尾已完成，CPU 核数画像入口 `auxiliary_core_budget.py` 已实现，用户已完成 R_sys=2 核数曲线 40/40，BGE 候选缓存补测待用户运行；正式 D3 仍需并发主流水线证据；只有真实性和可行性门禁形成 Go 结论后，才实现 predictor 与 queue-horizon-aware transition planner。

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
