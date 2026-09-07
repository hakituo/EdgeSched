# small 9B 后续组合与离线筛选（2026-09-07）

C512 的 LLM/TTS 共置兼容性已在真实后端跑过 K18/K20/K24 三档 × T0/T3，共 6 条全部 `passed=true`、0 failure、swap 0。
T3 的推理期余量不可用于筛选：加载期峰值比推理期高 106–424 MiB，最坏时刻出现在 TTS 加载/放置阶段。
按最坏时刻口径，C512 实测可用上界为 **K24**（余量 531.59/561.59 MiB）；K28 预测余量仅约 32–62 MiB，K30 预测超出总容量，**不再自动安排**。
G_safe/H_TTS 仍未冻结；见 [驻留档兼容性实测](#c512-驻留档兼容性实测k18k20k24)。

## 用户手动命令

```powershell
& 'D:\AI\xiaoyou-core\legacy\mvp_core\scripts\experiments\run_small_9b_screen.ps1'
```

Windows venv_core 离线计算，不启动模型、WSL、Forge 或 GPU。产物为
`experiment_results/elastic_residency/c3_analysis/small_stage_screen_<timestamp>.json`，不覆盖已有输出。
入口 `experiments/elastic_residency/screen_small_9b.py` 复用既有 `compute_h`、`compute_g_safe`、
TTS/Forge 提取和 C2/C3 合并校验，不改原始画像。

输入是 small C2/C3 `20260907_143917`、small 视觉 Profile `20260907_171635`、
B2 T0/T3 `20260831_direct_nvml_v2` 与 Forge `20260906_direct_nvml_v1`。
C2/C3/视觉核对固定模型身份、来源快照和原始 child；TTS/Forge 作为通过门禁的历史独立画像先验。
显存使用完整 child 生命周期峰值减基线；视觉 FULL 联合峰值已经含 LLM，不能再加一份 LLM 或 KV。
G_safe 暂取既有算法各已知分量最大值，并纳入当前观察到的 915.015625 MiB 基线；
共置正残差、主机内存重叠、CPU 核心分配和实际兼容性仍待验证，不冻结阈值。

实现时在 Windows 对现有产物完成离线验证（`.test_tmp/small_screen_review.json`）：
三个上下文 T0 对应最高测得 GPU 容量候选 K20，T3 对应 K30，FULL 均不满足该筛选预算。
长上下文候选余量很小，不能按“筛选通过”直接当成安全配置。
FULL/C512 视觉联合画像叠加 GPU T0/T3 的估计余量分别为 −2706.15 / −1430.21 MiB。
这个否定只限已测 FULL 视觉状态，不外推为所有混合档、所有 CPU 放置均不可行。

## 用户筛选验收与首轮兼容性命令（20:45 批次）

用户产物：`c3_analysis/small_stage_screen_20260907_204516_653987.json`，
SHA-256 `7fa1b5ae48371d7bddcf7ac8e0138324e2c454d85bd4760bfa2f859284caaf41`。
Windows 离线重新核验全部输入并复算，JSON 与用户产物完全一致；没有启动真实模型。

| context | T0 最高候选 / 扣除 G_safe 后余量 MiB | T3 最高候选 / 扣除 G_safe 后余量 MiB |
|---|---:|---:|
| 512 | K20 / 109.85 | K30 / 123.79 |
| 2048 | K20 / 59.85 | K30 / 73.79 |
| 4096 | K20 / 15.85 | K30 / 9.79 |

G_safe=915.015625 MiB 仍为 provisional；上述是已测档位中的 GPU 筛选边界，
不含未测共置正残差，不能冻结 H_TTS 或证明主机内存可行。FULL 叠加 TTS 未满足此预算。
FULL/C512 的视觉+TTS 估计缺口 T0=2706.15 MiB、T3=1430.21 MiB；混合视觉档待测。

下一步由用户执行：

```powershell
& 'D:\AI\xiaoyou-core\legacy\mvp_core\scripts\experiments\run_small_9b_tts_compat.ps1'
```

启动器复用 `llm_tts_colocated.py`，固定 small SHA 和本次 screen SHA，增量同步代码、保存源码快照，
按顺序运行 `512:K18:T0`、`512:K18:T3` 各 1 轮，失败即停，输出独立
`llm_tts_colocated/small_9b_tts_compat_<timestamp>/`，不覆盖旧记录。
14 个工作物理核（使用 SMT）+2 个系统核、seed=42、n_ctx=512+128=640；
TTS 默认同一句短文本、最多 256 个生成 token。沿用隔离 T0/T3 增量作为残差先验，保存实际模型树和参考音频哈希，结果复核时须对齐 B2 身份。

两个配置统一从 K18 起步，筛选余量分别约 355.85 / 1631.79 MiB。
代码当前先在 GPU 加载完整 TTS，再将 T3 codec 移到 CPU，所以 T3 推理期候选 K30
不自动代表加载期安全；不直接安排边界 K30 或更高混合档。实际加载与完整时间线仍须复核。

本端点先将 LLM 填充到 640 个 KV 位置并保持其驻留，再生成非流式 TTS/WAV，最后卸载并归还租约。
没有 LLM 同时 decode，也没有 TTS 后 LLM 继续回答验证；不代表 SM 平衡、LLM→TTS 端到端流水线或持续会话可用性。
T0 为 GPU TTS，T3 为 GPU talker + CPU codec；不代表完整 TTS_CPU。
先验和单轮 smoke 不能替代正式重复或两个反向静态放置基线；本批不加载 V、Forge，不自动继续三/四组件实验。

验证：新增 PowerShell 测试从非项目目录以 `-PrintOnly` 检查命令，禁止调用 WSL，覆盖 small/screen 身份、
两单元矩阵、LF 换行和 fail-fast；筛选、候选矩阵、KV/音频/容量及离线共置分析共 33 项 Python 测试通过。
pytest 临时目录显式落在本仓库 `.test_tmp`，避免父仓库缓存和系统临时目录的沙箱权限限制。

## C512 驻留档兼容性实测（K18/K20/K24）

三批均由同一启动器执行，`run_small_9b_tts_compat.ps1` 新增可选 `-KConfig`（默认 `K18`，上限 `K24`）：

```powershell
& 'D:\AI\xiaoyou-core\legacy\mvp_core\scripts\experiments\run_small_9b_tts_compat.ps1' -KConfig 'K20'
```

批次：`small_9b_tts_compat_20260907_222013_7184271`（K18）、`20260907_222804_5599277`（K20）、
`20260907_223027_8624462`（K24）。共同口径：C512、`n_ctx=640`、KV 640 token、seed 42、
14 工作物理核（SMT）+2 系统核、`G_safe=915.015625` MiB 仍为 provisional、TTS 同一句短文本、每档 T0/T3 各 1 轮。

后端为 `real_llama_cpp_gguf` + `real_qwen3_tts_component_split`，`fallback_occurred=false`；
T0 为 speaker/talker/codec 全 `cuda:0`，T3 为 talker 在 `cuda:0`、codec 在 `cpu` 且 speaker 建完 prompt 后卸载。

| 档位 | B_idle | B_LLM | T0 共置增量 | T0 最坏峰值 / 物理余量 | T3 推理期 P | T3 加载期峰值 / 物理余量 |
|---|---:|---:|---:|---:|---:|---:|
| K18 | 952.31 | 4577.41 | 2238.00 | 6815.41 / 1335.59 | 6361.41 | 6785.41 / 1365.59 |
| K20 | 952.31 | 4823.41 | 2254.00 | 7077.41 / 1073.59 | 6607.41 | 6981.41 / 1169.59 |
| K24 | 952.31 | 5335.41 | 2284.00 | 7619.41 / 531.59 | 7483.41 | 7589.41 / 561.59 |

单位为 MiB，`gpu_total=8151.0`；最坏峰值取同步 `P_colocated` 与周期采样整卡峰值的较大者，物理余量 = 8151 − 最坏峰值。
三档 `B_LLM` 减去 `B_idle` 后分别为 3625.10/3871.10/4383.10 MiB，与 small C3 曲线 `m_l_usage_mb` 完全一致。

关键观察：

- **T3 的推理期余量是虚的**。代码先在 GPU 加载完整 TTS 再把 codec 移到 CPU，加载期峰值比推理期高 106–424 MiB，
  因此筛选只能用加载期/最坏时刻，不能用 `P_colocated`。
- **隔离 TTS 增量先验偏保守**。实测共置增量 T0 为 2238–2284（隔离先验 3255.04）、T3 为 1784–2148（隔离先验 1979.10），
  共置正残差仍为 0，说明共置没有产生额外开销。
- **screen 候选需要修正**。screen 的“T3 最高候选 K30”建立在隔离推理期增量上，按实测加载期口径不成立：
  K28 预测余量约 32–62 MiB，K30 预测超出总容量约 188–218 MiB（按 K24 实测峰值外推）。T0 侧 screen 给 K20 偏保守，实测 K24 仍可行。
- **音频确定性**：三批同配置 WAV 的 SHA-256 完全一致（T0 `abf2cfeb…`、T3 `2306e941…`），
  157484 B、3.28 s、非静音（峰值约 0.402），每轮均记录大小与哈希。
- **主机内存无压力**：RSS 峰值约 6.29 GiB，系统可用最低约 11.2 GiB，swap 增长 0（16G 环境）。

门禁：每轮 `acceptance_checks` 全真（direct NVML 无 CLI fallback、GPU 样本新鲜度上界 ≤ 244 ms、
KV 位置与 640 token 精确、租约与物理核全部归还、音频非静音且有限）。

边界与未做项：每档每配置仅 **1 轮 smoke**，不构成正式重复或阈值冻结；
未验证 LLM 与 TTS 同时计算、SM 份额、TTS 后 LLM 继续回答、视觉/Forge 共置，也没有两个反向静态放置基线。

下一步候选（按推荐顺序）：

1. 在 K24 上做 T0/T3 的 5 轮正式重复，确认最坏峰值可复现后再考虑冻结候选；
2. 补 S0 反向静态放置基线（LLM_CPU/TTS_GPU、LLM_GPU/TTS_CPU），当前 9B CPU 侧证据仍缺；
3. K28 边界负面证据需显式决定并另开口径，当前启动器拒绝 K24 以上档位。

## C512/K24 五轮正式矩阵与 H_TTS 冻结

正式批次 `small_9b_tts_compat_20260907_225155_3728828`：C512/K24 × T0/T3 各 5 轮，10/10 `passed=true`、
0 failure，`evidence_scope` 为 `llm_tts_colocated_formal_repeated_matrix`，源码提交 `e0f1779` 且工作树干净。

```powershell
& 'D:\AI\xiaoyou-core\legacy\mvp_core\scripts\experiments\run_small_9b_tts_compat.ps1' -KConfig 'K24' -Runs 5
```

五轮几乎零漂移：`B_idle=952.31`、`B_LLM=5335.41` 全 10 条一致；T0 的 `P_colocated=7619.41` 五轮相同，
时间线峰值 7619.41（一轮 7617.41）；T3 推理期 `P_colocated=7483.41` 五轮相同，加载期时间线峰值 7589.41 五轮相同。
WAV 五轮 SHA-256 一致（T0 `abf2cfeb…`、T3 `2306e941…`），3.28 s、非静音；swap 增长 0，全部 `acceptance_checks` 为真。

复算入口原先只认历史 `per_context`/`g_safe` 结构，本轮按职责拆出适配层
`experiments/elastic_residency/small_screen_view.py`（只做结构映射，不做容量计算），
`analyze_colocated_results.py` 新增 `--screen-schema auto|legacy|small`：small 模式把候选判据从
“必须等于筛选 H_TTS 边界”放宽为“落在已画像档位内”，并输出 `screen_boundary` 与 `measured_beyond_screen_boundary` 供人工核对。

```powershell
cd 'D:\AI\xiaoyou-core\legacy\mvp_core'
& 'D:\AI\xiaoyou-core\venv_core\Scripts\python.exe' -m experiments.elastic_residency.analyze_colocated_results `
  --screen-analysis 'experiment_results/elastic_residency/c3_analysis/small_stage_screen_20260907_204516_653987.json' `
  --colocated-summary 'experiment_results/elastic_residency/llm_tts_colocated/small_9b_tts_compat_20260907_225155_3728828/summary.json' `
  --output 'experiment_results/elastic_residency/c3_analysis/small_9b_tts_colocated_20260907_225155_v1.json'
```

产物 `c3_analysis/small_9b_tts_colocated_20260907_225155_v1.json` 的结论：

- **`G_safe=915.015625 MiB` 已冻结**（`frozen_for_llm_tts`）：分量为 idle 高水位 915.015625、
  allocator slack 501.28、重复漂移 0.0、共置正残差 168.90，取最大值。
- **共置正残差首次非零**：T3 的共置增量 2148.00 大于隔离先验 1979.10，残差 168.90 MiB；T0 仍为 0。
  该值已计入 G_safe 分量，但未超过 idle 高水位，所以 G_safe 数值不变。
- **`H_TTS`(C512) 冻结**：T0/T3 均为 K24，`frozen_real_colocated`、5 轮，安全余量下界 568.89 / 704.89 MiB。
- **`H_D` 仍为 pending**：small screen 只有 Diffusion 主机内存诊断，没有可复核档位，映射后档位保持 `None`（不填 0）。
- 两个候选都标记 `measured_beyond_screen_boundary=true`（离线边界 T0=K20、T3=K30），即实测修正了离线筛选口径。

必须一起读的边界：

- 只验证了 **K24 这一档**，不等于“C512 上界是 K24”。K28/K30 未实测，K30 按实测外推在加载期超出总容量约 188–218 MiB。
- 只对 **C512**；2048/4096 未做共置，长上下文余量仍按离线筛选的小余量处理。
- T3 一律按最坏时刻（加载期 7589.41）而非推理期（7483.41）裁决：复算产物的
  `physical_headroom_min_mb` 仍按同步 `P_colocated` 给出（T3 667.59 MiB），按加载期峰值则为 561.59 MiB，
  两者并存，人工裁决取更严的后者。
- 未验证 LLM 与 TTS 同时计算、SM 份额、TTS 后 LLM 继续回答、视觉/Forge 共置。

## S0 反向静态放置基线（进行中）

两个 smoke 批次（2026-09-07 23:13/23:14，均 1 轮、全门禁通过、swap 0）：

- `512:K24:T2`（`20260907_231256_7638174`）**不构成 TTS_CPU 基线**。`PlacementSpec`
  字段顺序为 `(talker, codec, speaker)`，T2 的 talker 仍在 `cuda:0`；实测 `P=7481.41`、
  时间线峰值 7589.41，与 K24:T3 实质相同。T0–T3 没有任何配置把 talker 放到 CPU。
- `512:K0:T0`（`20260907_231359_6390285`）是有效的 LLM_CPU+TTS_GPU 一侧：LLM 全 CPU
  （`B_LLM=1079.41`，减 idle 后 127.10 与 C3 曲线 K0 一致），`P=3333.41`、峰值 3331.41、
  物理余量 4817.59 MiB；RSS 峰值 8.36 GiB（比 GPU 侧高 2.07 GiB）、系统可用最低 11.71 GiB、swap 0；
  WAV 与 GPU 侧逐字节一致（TTS 输入是固定文本，不接 LLM 输出）。

为此新增 **T4（talker/codec/speaker 全 CPU）**：`tts_component_offload.py` 的 `PLACEMENTS`
加入 T4，`CONFIG_NAMES` 扩为 T0–T4 以便 `--child-config` 接受；B1/B2 的 `--configs` 默认显式固定为
T0–T3，历史默认矩阵与既有数据不受影响。colocated 入口显式传 `T4` 即可用。

S0 补全步骤（每步失败即停）：

```powershell
# 1) T4 隔离 smoke：采集 GPU 隔离增量与 CPU talker 耗时量级
& 'D:\AI\xiaoyou-core\legacy\mvp_core\scripts\experiments\run_small_9b_t4_isolate.ps1'

# 2) 从 B1 产物 summary.json 读 T4 的 max_card_delta_mb，代入共置 smoke
& 'D:\AI\xiaoyou-core\legacy\mvp_core\scripts\experiments\run_small_9b_tts_compat.ps1' -KConfig 'K24' -TtsConfigs 'T4' -Isolated 'T4=<B1 实测值>'
```

预期与口径提醒：CPU talker 预计显著慢于 GPU（B2 全部配置 RTF>1），慢本身就是“全 CPU 降级不可用”
的论据；T4 未在 B2 direct NVML v2 里画像，其隔离增量来自 B1 单文本 smoke，登记时必须标注口径差异；
K0:T0 与将来的 K24:T4 都只是 1 轮 smoke，作基线证据须补 5 轮正式矩阵，且 S0 完整形态仍缺端到端流水线。

T4 smoke 实测（首次 `20260907_232617` 因 `torch_phase_peaks_available` 门禁不适配纯 CPU 而 failed；
该门要求 CUDA 活动峰值，纯 CPU 下恒为 0，已在 `tts_component_offload.py` 与 `llm_tts_colocated.py`
按 `talker_device == "cpu"` 豁免；重跑 `20260907_233333` 全过）：

- talker 14.10 s、RTF 5.49、音频 2.72 s（peak 0.3867，与首次一致）；放置后 GPU 隔离增量
  149.09765625 MiB，加载瞬间增量 2349.09765625 MiB（权重先落 GPU 再移 CPU）。
- `512:K24:T4` 共置 smoke（`20260907_233523_5430884`）通过：设备全 `cpu`；`B_idle=986.42`、
  `B_LLM=5369.52`（减 idle 后 4383.10 与 K24 曲线一致）、推理期 `P=5377.52`——TTS 全 CPU 后
  GPU 共置增量仅 8.0 MiB，正残差 0；**加载期整卡峰值 7623.52 MiB**（余量 527.48 MiB）仍是约束；
  RSS 峰值 7.35 GiB（比 K24:T3 高 1.06 GiB）、swap 0、WAV 与 B1 T4 输出一致；talker 15.35 s。

至此 S0 的四个静态放置均有 1 轮 smoke：K24:T0（GPU/GPU）、K24:T3（GPU talker+CPU codec）、
K0:T0（CPU/GPU）、K24:T4（GPU/CPU）；尚缺二元全 CPU 降级 K0:T4 与各配置的 5 轮正式重复。

### S0 五轮正式矩阵（2026-09-07 23:41–23:50）

三个批次全部 `formal_repeated_matrix`、0 failure、`Git provenance` 干净：

| 配置 | 批次 | 推理期 P（MiB） | 加载期峰值（MiB） | 最坏物理余量 | TTS GPU 增量 | talker | swap |
|---|---|---|---|---:|---:|---:|---|
| K0:T0 | `20260907_234152_0966476` | 3341.82–3355.74 | ≈推理期 | 4795.26–4809.18 | 2254–2268 | ~5.6 s | 0 |
| K24:T4 | `20260907_234703_5300946` | 5365.05 ×5 | 7611.05 ×5 | 539.95 | 8.0 | 15.35 s | 0 |

K0:T0 五轮内 `B_idle` 从 960.72 漂到 973.95（+13.2 MiB 环境漂移），但 `B_LLM−idle` 恒为 127.10；
K24:T4 五轮完全零漂移（含加载期峰值 7611.05），WAV 五轮 SHA 一致。

K0:T4 全 CPU 降级 smoke（`20260907_234030_5254560`，1 轮）：设备全 `cpu`，`P=1113.38`、
加载期峰值 3305.50，物理余量约 4845 MiB；但 **RSS 峰值 11.35 GiB、系统可用最低 8.53 GiB、
swap 增长 0.762 MiB**——S0 系列首次出现非零 swap（仍在 4 MiB 容忍内），16G 主机上的全 CPU
降级已触碰换页边缘；talker 16.87 s、RTF 约 6.2。

S0 静态放置的核心量化结论（C512）：

1. **最坏时刻由 TTS 权重加载过 GPU 决定，与 TTS 最终放哪无关**：K24 下 T0/T3/T4 的加载期
   峰值分别为 7619.41 / 7589.41 / 7611.05，最坏物理余量 531.59 / 561.59 / 539.95 MiB，几乎相同。
2. **TTS 挪到 CPU 后 GPU 推理期增量从 2284 降到 8 MiB**，但 talker 从 ~5.6 s 变 15.35 s（RTF 5.5）。
3. **全 CPU 降级（K0:T4）在 16G 主机上已触发换页**且 RTF 6.2——“全 CPU 不可用”同时有显存
   之外的内存与延迟论据。
4. 以上均为容量/可行性口径（`performance_timing: diagnostic_stage_time_only`），不是端到端流水线收益。

### K0:T4 五轮正式矩阵负面边界收案

`20260907_235535_4085006`（2/5 后 fail-fast）：r1 swap 增长 0.76 MiB（容忍内通过），
r2 从 0.76 MiB 跳到 **54.07 MiB（增长 53.3 MiB，超 4 MiB 容忍约 13 倍）**，`swap_growth_within_tolerance`
失败停止。全程 RSS 11.37 GiB、系统可用最低 8.52 GiB。这是累积性主机内存压力：单轮 smoke 通过
不代表正式矩阵可复现。与 9B K6/K4+Forge 的处理一致，不换环境重试、不硬闯；`.wslconfig` 20G
仅作归因诊断用，不是论文环境。

**S0 最终形态**：正式 5 轮 ×4 配置（K24:T0、K24:T3、K0:T0、K24:T4）全部通过；K0:T4 全 CPU
降级以负面边界收案——“全 CPU 不可用”同时具备延迟（RTF 6.2）、主机内存（RSS 11.4 GiB 贴边、
swap 累积失控）与不可复现三重论据，直接支撑弹性驻留的必要性。

### S0 多批次合并冻结（2026-09-08）

复算入口新增 `colocated_batch_merge.py`（批次级硬校验：正式矩阵、整体通过、provenance 干净、
候选跨批次不重复、每候选 5 轮）与 `--colocated-summaries` 多批次模式；首轮合并即拦截了
K0:T0/K24:T4 初跑批次（当时工作树 `dirty=True`，系 eec05d8 漏提文档所致），重跑后干净通过。
另修复 `h_tts` 摘要键：同一 (context, tts) 出现多个 llm 档时后写会覆盖先写，现改为
`T0_at_K0`/`T0_at_K24` 形式，四候选完整保留。

产物 `c3_analysis/small_9b_s0_colocated_20260908_v1.json`：20 条记录、4 候选 × 5 轮，
源提交 `e0f1779` + `3196a95`（均干净），`G_safe=915.015625 MiB`（正残差 168.90 来自 K24:T3），
四个候选全部 `frozen_real_colocated`：

| 候选 | P_max (MiB) | 物理余量下界 | 安全余量下界 | 正残差 |
|---|---:|---:|---:|---:|
| C512_K0_T0 | 3335.11 | 4815.89 | 4854.89 | 0 |
| C512_K24_T0 | 7619.41 | 531.59 | 568.89 | 0 |
| C512_K24_T3 | 7483.41 | 667.59 | 704.89 | 168.90 |
| C512_K24_T4 | 5362.93 | 2788.07 | 2844.89 | 0 |

注意：物理余量按同步 `P_colocated` 给出；T0/T3/T4 在 K24 下的加载期峰值（7619/7589/7611）才是
最坏时刻，人工裁决取更严值。`H_D` 仍 pending；S0 完整 Go/No-Go 仍缺端到端流水线证据。

## 组合按两条任务路径展开

| 阶段 | 组合 | 先回答的问题 |
|---|---|---|
| 基础语音路径 | LLM + TTS | LLM_GPU/TTS_CPU、LLM_CPU/TTS_GPU、混合 LLM/TTS_GPU 的容量、主机内存和端到端可用性 |
| 视觉理解路径 | LLM + TTS + V | V 编码时 TTS 的设备/队列安排；V 释放后恢复语音路径，或在可行配置下保留 V |
| 图像生成路径 | LLM + TTS + Diffusion | 生成期间 LLM/TTS 如何降级、排队及恢复，GPU 与主机内存门同时满足 |
| 条件性扩展 | LLM + TTS + V + Diffusion | 仅在已到达的实际任务需要重叠且容量筛选允许时尝试，不作为必须跑完的最后一级 |

V 是 VL 前端组件，编码后仍依赖 LLM 回答；Diffusion 属于另一条生成路径。
三个/四个组件参与同一工作流，不等于全部同时驻留 GPU，更不等于全部同时提交 kernel。
比较要分别记录组件驻留、计算重叠及 CPU/GPU 放置。

当前 GPU 共存先测试兼容性，速度保留为未管理计算份额的基线；正式 SM 平衡在原生 Ubuntu 能力门通过后开展。
TTS 内部非流式的限制不会由 SM 分配自动消失。CPU TTS 成对端点和共享物理核预算仍需独立安排，不能从 T0/T3 GPU 数据代替推断。

Diffusion 的 16G 主机内存负面边界继续保留。筛选脚本仅给 GPU 数值诊断，不生成可执行的 Diffusion 重跑计划，
也不会因换了 small 文件就把旧失败门解除；四组件状态先筛选，不强行测明显放不下的组合。

## 大版模型删除与版本管理

用户明确要求删除后，已核对并只删除 WSL 文件
`/home/leslie/mvp_core/models/llm/Qwen3.5-9B-q4_k_m.gguf`，大小 5,974,828,256 B（约 5.56 GiB），
删除前完整 SHA 为 `423fb3e5fe2c98df616159b7cd1bd04765f67667f5f8712a81e8f63f4e7ee1ef`。
删除后确认 small `comparison_1a35ab89/model.gguf`（5,627,044,704 B）和
`Qwen3.5-9B-f16.mmproj`（918,165,888 B）仍存在且大小不变；没有清理目录或删除实验结果。
这是释放 WSL 文件系统空间，不等同于运行期内存减少，也不保证宿主 VHDX 文件立即缩小。
历史 large 参数/比较脚本保留用于复现，但现在直接运行 Large 或双版本对比会因文件不存在而停止，需先恢复相同哈希文件。

GitHub 目标已通过仓库 API 确认为 `hakituo/EdgeSched` 私有仓库。
本轮整理此前未提交的 CPU 画像、9B 对比、视觉/完整曲线记录、论文生命周期修订及新筛选入口，
不纳入模型、原始实验产物或临时测试输出。Git 提交不改变旧实验 manifest 的源提交与快照身份。

## 验证

`tests/elastic_residency`：400 passed、5 skipped（明确要求 WSL/CUDA/MPS 的真实测试未启用）。
新增筛选测试覆盖联合视觉不重复计费、保留 Diffusion 主机门、观察基线及 Forge 来源字段兼容。
所有修改的 Python 文件 Ruff 通过且不超过 300 行；PowerShell 视觉与完整曲线入口静态测试通过。
源码范围扫描未发现模型/二进制产物或所检查的凭据模式。没有运行新模型实验。
