# 弹性驻留运行结果目录

small 9B 离线筛选 `20260907_204516_653987` 已复算一致：T0/K20、T3/K30 仅为容量候选，长上下文余量不足 16 MiB。下一步用户运行 `scripts/experiments/run_small_9b_tts_compat.ps1`，C512/K18 配 T0/T3 各 1 轮；加载期另需验证，不冻结阈值，见 [验收与命令](SMALL_9B_NEXT_COMBINATIONS_20260907.md)。

small 9B 视觉生命周期五轮 `20260907_171635` 已 5/5、swap 0；加载/预热 0.481 s、编码 0.147 s、释放 0.011 s，见 [阶段画像](SMALL_9B_VISION_PROFILE_20260907.md)。这是独立进程画像，保温/提前释放策略仍待验证。

完整曲线批次 `20260907_143917` 已离线逐条核验，288/288、swap 0，见 [16 档曲线与 KV 验收](SMALL_9B_CURVES_20260907.md)。

> **2026-09-07 9B 版本选定**：后续主模型为 small `1a35ab89…` + F16 projector `e36da9b9…`。纯文本对比 12/12、small 视觉 Answer 1/1 通过，固定图形内容描述完整；专用视觉启动器默认 Small，Large 可显式复测。新版本 C2 90/90、C3 150/150、KV 48/48 已通过，4B 对照与旧哈希证据保留。单图不是完整质量评测或 TTS 共置结论。详见 [选型与验收](NINE_B_MODEL_COMPARISON.md)。

**盘点日期**：2026-08-30  
**工作产物根目录**：`experiment_results/elastic_residency/`  
**用途**：区分当前候选证据、历史停止记录、已替代结果和 debug/smoke，防止跨测量后端或跨口径混用。

## 1. 目录规则

```text
experiment_results/elastic_residency/
├── a1/ ... kv_calibration/       # 当前仍在使用的批次
├── c3_analysis/                  # 当前容量筛选输出
└── _archive/
    ├── superseded/               # 完整但已被更新版本/后端替代
    ├── stopped/                  # 失败或中断，不得续跑或并入完整曲线
    └── debug_smoke/              # 只证明调试链路或单轮 smoke
```

- 归档只改变目录位置，不改写、压缩或删除原始 JSON、JSONL、WAV、CSV 和 manifest。
- 当前批次保持原路径，避免影响接下来的分析脚本和文档引用。
- `_archive` 中的记录只用于追溯，不能自动晋升为论文证据。
- 工作产物根目录被独立仓库 `.gitignore` 排除，不提交到 Git；正文证据仍须按人工确认流程晋升到 `paper/experiment_data/`。

## 2. 当前批次

| 阶段 | 当前路径 | 状态与用途 |
|---|---|---|
| small 9B 五轮视觉生命周期 | `vision_endpoint/v1_vision_9b_small_profile_20260907_171635/` | 5/5、swap 0、逐轮原始记录核验通过。V 加载/预热 0.481 s、编码 0.147 s、释放 0.011 s；整卡峰值 7462.38 MiB，释放读数减少 1148 MiB。独立进程画像，非策略对照，见 [报告](SMALL_9B_VISION_PROFILE_20260907.md) |
| small 9B C2 | `c2/c2_9b_small_20260907_143917/` | 90/90 全通过，swap 0、固定 small SHA、共享源码快照；已逐条核验，见 [完整验收](SMALL_9B_CURVES_20260907.md) |
| small 9B C3 | `c3/c3_9b_small_20260907_143917/` | 150/150 全通过，swap 0、固定 small SHA、共享源码快照；已逐条核验，见 [完整验收](SMALL_9B_CURVES_20260907.md) |
| small 9B KV | `kv_calibration/kv_9b_small_20260907_143917/` | 48/48 全通过，swap 0、固定 small SHA、共享源码快照；已逐条核验，见 [完整验收](SMALL_9B_CURVES_20260907.md) |
| small 9B 视觉 Answer | `vision_endpoint/v1_vision_9b_small_answer_20260907_142945/` | 1/1、34 项门禁通过、swap 0；固定图五项内容描述完整。TTFT 0.4251 s、decode 58.738 tokens/s、峰值 7501.02 MiB（含 48 MiB GPU KV），n_ctx=1536。纯单轮回答验收；选定 small 为后续版本，新完整曲线待建，见 [详情](NINE_B_MODEL_COMPARISON.md) |
| 两份 9B 选型对比 | `model_comparison/compare_9b_20260907_023703/` | 12/12、swap 0，六组配对 prompt 一致，原始 child/iterations/源码快照离线复核与 summary 一致。small FULL 显存峰值少 90 MiB、RSS 峰值少 331.45 MiB、decode +2.05%；纯文本三轮诊断，small 视觉未验收、默认模型未切换，见 [详细对比](NINE_B_MODEL_COMPARISON.md) |
| 9B 视觉重启后复核 | `vision_endpoint/v1_vision_9b_profile_20260907_012723/` | 5/5、swap 0；与 20260906_204939 参数/实验代码一致。decode mean −0.72%、TTFT mean −3.37%；整卡基线及峰值均减少 86.30 MiB，峰值减基线不变。未见旧批严重性能污染证据，分批保留，详见 [重启对照](VISION_9B_PROBE_20260906.md) |
| 9B 回答补测 | `vision_endpoint/v1_vision_9b_answer_20260906_204651/` | 1/1 技术通过，1024 tokens / n_ctx=1536；主要图形正确、漏黑边，9 个 EOS 后仍固定生成；不是质量全通过或 128-token 性能数据 |
| 9B 五轮视觉画像 | `vision_endpoint/v1_vision_9b_profile_20260906_204939/` | 5/5、swap 0；峰值 7542.47 MiB（已含 24 MiB GPU KV），TTFT mean 0.5128 s、decode mean 55.7562 tokens/s；只证明独立资源画像，下一步先离线筛容量，见复核记录 |
| 9B FULL 文本/视觉端点 | `vision_endpoint/v1_vision_9b_probe_20260906_201923/` | 用户实跑 2/2 通过；196 视觉 tokens、嵌入 4096 维、swap 0、整卡采样峰值 7542.47 MiB；128-token 输出截断，仅单轮链路证据，详见 [复核记录](VISION_9B_PROBE_20260906.md) |
| A1 | `a1/a1_smoke_20260829_verified_v7/` | 当前实验基础设施硬门证据 |
| Audio | `audio_compat/audio_compat_20260829*/` | WSLg/Windows 播放兼容性记录 |
| B1 | `b1/b1_tts_components_20260829_verified_v2/` | T0--T3 单文本组件放置 smoke |
| B2（当前口径） | `b2/b2_tts_t0_20260917/`（状态残留序列为 `b2_tts_t0_20260917_state/`） | 当前接入后端（加速 CUDA Graph 整模型 GPU，放置 T0）的句长画像：20/50/100 字 × 5 轮 + 状态残留序列 + 主环境 ASR 回转核对；记录 RTF、波形就绪、加载与图准备成本、准备/执行阶段内存峰值、输出长度 |
| 旧后端 B2（已删除） | 原 `b2/b2_tts_matrix_20260829_verified_v1/`、`b2/b2_tts_pareto_matrix_20260831_direct_nvml_v1|v2/` | 2026-09-17 按「B2 只指当前接入后端」删除；其 T0/T3 条件式 Pareto 与全部 RTF 数值随产物一并退役，不得被引用为当前依据；生成器 `tts_component_offload_matrix.py` 已于同日删除 |
| C1 | `c1/c1_llm_residency_512_20260829_verified_v1/` | K0/FULL 端点 smoke |
| 9B C2 | `c2/c2_llm_residency_curve_20260829_direct_nvml_v2/` | Qwen3.5-9B 90/90 粗驻留曲线；迁移后作为模型特定压力基线 |
| 9B C3 | `c3/c3_llm_critical_refinement_20260829_direct_nvml_v2/` | Qwen3.5-9B 150/150 临界细化曲线；迁移后作为模型特定压力基线 |
| 9B C3 analysis | `c3_analysis/c3_h_direct_nvml_20260831_v1.json` | 9B 完整 16 档 + B2 direct v2 + Forge direct v1 容量筛选；不得复用为 4B 阈值 |
| 9B C3 LLM+TTS freeze | `c3_analysis/c3_h_tts_colocated_20260831_v1.json` | 复用 9B 正式共置 30 条离线复算；正残差 max=0，冻结 9B `G_safe=501.282 MiB` 与 H_TTS；9B H_D 未冻结 |
| 4B C2 | `c2/c2_llm_residency_curve_20260901_4b_direct_nvml_v1/` | Qwen3.5-4B 90/90 粗驻留曲线，真实 GGUF/direct NVML/swap 0；artifact 保存短 commit `a35d4f5`，已唯一解析到完整提交，晋升前须附 provenance 审计记录 |
| 4B C3 | `c3/c3_llm_critical_refinement_20260901_4b_direct_nvml_v1/` | Qwen3.5-4B 150/150 临界细化曲线；与 4B C2 合并后每个 context 恰好 16 档；同样保留短 commit 审计说明 |
| 4B KV | `kv_calibration/c3_kv_cache_calibration_20260901_4b_direct_nvml_v1/` | 48/48 正式 KV 容量与语义校准，完整 Git commit、direct NVML、swap 0；640/2176/4224 状态 token 对应 24/72/136 MiB KV buffer |
| 4B TEXT_ONLY TTS screen | `c3_analysis/c3_h_4b_tts_direct_nvml_20260901_v1.json` | 4B 完整 16 档 + B2 direct v2 离线容量筛选；T0/T3 在 512/2048/4096 下均为 FULL 可行候选；未含真实 4B 共置残差，不冻结 H_TTS，图像 H_D pending |
| 4B LLM+TTS smoke | `llm_tts_colocated/llm_tts_colocated_c4096_full_t0_4b_smoke_20260901_v1/` | C4096/FULL+T0 1/1 技术通过，真实后端/direct NVML/swap 0/完整 provenance；`P_colocated=6406.699 MiB`、物理余量 1744.301 MiB、正残差 0。该批次运行时尚未把 WAV SHA-256 写入 JSON，仅保留为正式矩阵前 smoke；正式矩阵使用新增哈希门禁的新 commit/run id |
| 4B LLM+TTS formal | `llm_tts_colocated/llm_tts_colocated_4b_text_only_20260904_direct_nvml_v1/` | 6 候选×5 轮（512/2048/4096 × FULL × T0/T3），30/30、0 failure/fallback；真实后端/direct NVML/swap 增长 0/窗口内新鲜度 218–324 ms（<750 门禁）/30 个 WAV 大小与 SHA-256 全通过；manifest provenance 完整（commit `dc3974bf`，dirty=false）。已由 `analyze_colocated_results.py` 离线复算冻结 4B 文本口径 G_safe/H_TTS |
| 4B LLM+TTS freeze | `c3_analysis/c3_h_4b_tts_colocated_20260906_v1.json` | 4B 文本共置离线复算：30/30 硬验、正残差 max=0，G_safe=501.282 MiB 校准后维持预设值并 `frozen_for_llm_tts`；H_TTS 在 512/2048/4096×T0/T3 全部 `frozen_real_colocated`（各 5 轮）；H_D 由 20260906 图像侧批次以负面边界证据收案（见下两行），该 JSON 内的 H_D 字段维持 pending 语义不再复算 |
| 4B VISION_WARM 画像 | `vision_endpoint/v1_vision_endpoint_4b_warm_profile_20260906_direct_nvml_v1/` | FULL × VISION_WARM_GPU × 5 轮，5/5、0 failure/fallback；projector warm GPU 增量 mean 761.2 / min 710 / max 838 MiB（含 warmup 按 1472×1472 预留的 223.30 MiB，非 448×448 编码成本），release 增量 −888 MiB、RSS +约 55.8 MB；视觉 token 196→14 序列位置，KV roundtrip 167 token 成功；供 4B 视觉画像引用，不冻结任何 H |
| Forge（9B screen 用） | `forge/forge_b1_reprofile_20260831_direct_nvml_v1/` | direct NVML 5/5 校准，全部门禁通过，`M_D=5484 MiB`；已由 20260906 同日复测替代为 4B screen 输入 |
| 4B Forge 同日复测 | `forge/forge_b1_reprofile_20260906_direct_nvml_v1/` | direct NVML 5/5，全部门禁通过；同日口径 `M_D=5273.21 MiB`（idle 高水位 655.61、常驻基线 668.82、b=1 延迟 mean 6.207 s）；当前 4B screen 的 Forge 输入 |
| 4B screen（Forge 口径） | `c3_analysis/c3_h_4b_tts_direct_nvml_20260906_v1.json` | 4B 16 档 + B2 direct v2 + Forge 20260906 的容量筛选；H_TTS 维持 FULL 候选，FULL 对 H_D 因 GPU 容量不可行（LLM 预算 2222 MiB），H_D 候选为 512/K16、2048/K14、4096/K14 |
| 4B LLM+Forge 正式矩阵（16G 停止） | `llm_forge_colocated/llm_forge_colocated_4b_20260906_direct_nvml_v1/` | 基于干净 `a39778f` 的正式矩阵，首个 K16@C512 唯一失败门为 swap 增长 405.15 MiB（容忍 4 MiB），失败即停（1/15）；GPU 容量、新鲜度、KV、PNG、provenance 全部有效；与 9B K6/K4 同模式负面证据 |
| 4B LLM+Forge 20G 归因诊断 | `llm_forge_colocated/llm_forge_colocated_4b_20260906_mem20g_diag_v1/` | 同代码、同候选、同参数，仅 `.wslconfig memory=20GB` 不同：K16/K14/K14 三候选各 1 轮全部通过，swap 增长 0.0–0.2 MiB，GPU 峰值 8122/8151 MiB；证明 16G 失败根因为主机内存容量，而非 K 档固有换页或 WSL2 特有行为。归因诊断证据，不进论文正文（论文环境为 16G） |
| 9B KV | `kv_calibration/c3_kv_cache_calibration_20260830_direct_nvml_v1/` | Qwen3.5-9B 48/48 KV 容量与语义证据 |
| 9B LLM+TTS smoke | `llm_tts_colocated/llm_tts_colocated_c4096_k20_t0_smoke_20260831_v1/` | C4096/K20+T0 1/1 技术通过，0 failure/fallback；正残差 0，只作 9B 正式 artifact smoke |
| 9B LLM+TTS formal | `llm_tts_colocated/llm_tts_colocated_candidates_20260831_direct_nvml_v1/` | 6 候选×5 轮，30/30、0 failure/fallback；真实后端/direct NVML/新鲜度/swap/provenance 全通过；原始运行版容量字段有 idle 重复计数，测量无需重跑，以离线复算为准；不得直接代表 4B+视觉口径 |
| 9B LLM+Forge K6 smoke | `llm_forge_colocated/llm_forge_colocated_c512_k6_smoke_20260831_v1/` | 1/1 真实运行；后端、PNG、direct NVML、新鲜度、KV、GPU 容量和 provenance 有效，swap 增长约 290.44 MiB，整体失败 |
| 9B LLM+Forge K4 fallback | `llm_forge_colocated/llm_forge_colocated_c512_k4_fallback_smoke_20260831_v1/` | 1/1 探索性真实运行；只有 swap 门失败，增长约 2070.32 MiB，证明继续降 GPU 层会恶化 DDR 压力；不继续 K2 |

2026-08-30 整理后，当前区约为 342.62 MiB、736 个文件。大文件主要是逐轮 child JSON 和高频时间线，不是论文聚合表。

## 3. 已归档批次

### 3.1 `superseded/`

- `a1/a1_smoke_20260828_verified/`
- `a1/a1_smoke_20260828_verified_v2/` 至 `v6/`
- `b1/b1_tts_components_20260829_verified_v1/`
- `c2/c2_llm_residency_curve_20260829_verified_v3/`（90/90 旧 CLI 后端完整基线）

### 3.2 `stopped/`

- `c1/c1_k0_debug_20260829_v1/`
- `c3/c3_llm_critical_refinement_20260829_verified_v1/`（1/150）
- `c3/c3_llm_critical_refinement_20260829_verified_v2/`（94/150）
- `c2/c2_llm_residency_curve_20260829_verified_v1/`（22/90）
- `c2/c2_llm_residency_curve_20260829_verified_v2/`（31/90）
- `c2/c2_llm_residency_curve_20260829_direct_nvml_v1/`（64/90）
- `forge/forge_b1_reprofile_20260829_verified_v1/`（旧 swap 门禁失败）

### 3.3 `debug_smoke/`

- `b1/b1_t0_debug_20260829_v1/`、`v2/`
- `b2/b2_t0_matrix_debug_20260829_v1/`
- `c1/c1_full_debug_20260829_v1/`、`c1_k0_debug_20260829_v2/`
- `c2/c2_gpu_deadline_debug_20260829_v1/`
- `c2/c2_observe_policy_debug_20260829_v1/`
- `c2/c2_orchestrator_debug_20260829_v1/`
- `c3/c2_nvml_overrun_recovery_smoke_20260829_v1/`
- `c3/c3_direct_nvml_smoke_20260829_v2/`
- `c3/c3_gpu_sampler_card_only_smoke_20260829_v1/`、`v2/`
- `kv_calibration/kv_calibration_smoke_20260830_verified_v1/`

归档区共 28 个批次，约 408.81 MiB、876 个文件。

## 4. 历史归类：9B 产物在 4B 迁移后（2026-09-01，主模型选型已更新）

主实验模型迁移到 `Qwen3.5-4B-Q4_K_M` + `mmproj-BF16.gguf` 后，9B 产物按下列三类处理。
判定标准是**该结果是否依赖 LLM 模型身份**：不依赖的直接沿用，依赖的转为压力/负面/
方法学证据，禁止改名为 4B 结论。

### 4.1 A 类：与 LLM 无关，直接沿用

| 产物 | 说明 |
|---|---|
| `b1/b1_tts_components_20260829_verified_v2/` | TTS 组件发现，纯 TTS 侧 |
| `b2/b2_tts_t0_20260917/` | 当前接入后端（加速整模型 GPU，T0）画像，**不含 LLM**；旧后端 T0–T4 三批产物已于 2026-09-17 删除 |
| `forge/forge_b1_reprofile_20260831_direct_nvml_v1/` | Forge 是独立服务，`M_D=5484 MiB` 与 LLM 无关 |
| `a1/a1_smoke_20260829_verified_v7/` | 实验基础设施硬门 |
| `sm_capability/sm_a_20260830_direct_nvml_v1/` | 环境能力结论（MPS binary 不存在 → SM-A NO_GO） |

这三项（B2 + Forge + 方法学）即是容量筛选中与模型无关的输入，4B 阶段不需要重跑。

### 4.2 B 类：转为压力 / 负面 / 方法学证据，不得改名为 4B 结论

| 产物 | 迁移后的正确用法 |
|---|---|
| `c2/c2_llm_residency_curve_20260829_direct_nvml_v2/`（90/90） | **大模型压力基线**：证明该驻留画像方法在更大模型上同样可跑通 |
| `c3/c3_llm_critical_refinement_20260829_direct_nvml_v2/`（150/150） | 同上；16 档合并后构成 9B 完整曲线 |
| `kv_calibration/c3_kv_cache_calibration_20260830_direct_nvml_v1/`（48/48） | KV 语义纠正（`70.263/118.292/182.331 MiB`，不是 1–2 GiB）由 9B 批次完成，方法学结论继续有效 |
| `c3_analysis/c3_h_tts_colocated_20260831_v1.json` | 冻结的是**9B 模型口径**的 `G_safe=501.282 MiB` 与 `H_TTS`，作为尺度对照保留 |
| `llm_forge_colocated/llm_forge_colocated_c512_k6_smoke_20260831_v1/`、`..._k4_fallback_.../` | **最有价值的负面证据**：证明驻留筛选必须联合 GPU 与主机内存，只看显存会选错档位；该结论不依赖模型 |

K4 把更多权重退回 DDR 后换页从约 290.44 MiB 恶化到约 2070.32 MiB，是论文可直接引用的
方法论贡献，换模型不会失效。若后续同时完成 9B 与 4B 的同口径驻留画像，可形成**跨模型
尺度对比**（同一方法、同一后端、同一门禁），比只报单一模型更强。

### 4.3 C 类：作废，禁止沿用

- 9B 的具体显存 / RSS / KV / TTFT / prefill / decode 数字；
- 9B 的 `H_D`（本来就没冻结，两次 swap 门失败后正式矩阵已停止）；
- 9B 的模型哈希、层数等 artifact 身份——4B 已独立核验（`qwen35.block_count=32`）；
- 任何把 `mmproj` 文件大小加到 9B 曲线上冒充测量的做法。

### 4.4 入口默认值变更

`llm_residency_curve.py`、`llm_kv_cache_calibration.py`、`llm_tts_colocated.py`、
`llm_forge_colocated.py` 的 `--model` 默认值已由已删除的 `Qwen3.5-9B-Q4_K_M.gguf`
改为 `Qwen3.5-4B-Q4_K_M.gguf`。历史 9B 产物不受影响（它们记录了当次使用的模型路径
与哈希）；但**所有显式引用旧路径的旧命令都不能再跑**，4B 运行请直接省略 `--model`
或显式传新路径。

## 5. 当前禁止混用的口径

1. C2/C3 最终性能曲线只使用 `direct_nvml_v2` 两个批次。
2. KV 容量只使用 `c3_kv_cache_calibration_20260830_direct_nvml_v1`。
3. B2 direct v2 与 Forge direct NVML v1 是当前同后端隔离校准；B2 v1 与 Forge 2026-08-29 v2 只作旧口径基线。9B LLM+TTS 正式共置已冻结该模型口径的 `G_safe/H_TTS`；9B `H_D` 因 swap 门失败不冻结，4B+视觉口径须重新筛选和实测。
4. `_archive/stopped` 中的部分记录即使单轮 `passed=true`，整个矩阵仍不完整，禁止拼接。
5. `timeline.jsonl` 和 `*_child.json` 是原始可追溯证据；清理磁盘前应整体压缩归档，不能只保留 summary 后删除原始轮次。

## 6. 下一批产物

旧 9B C3 H_D 路径已经在两个 smoke 后按 swap 门停止；现有确切 artifact 已列在上表，不再创建 9B 正式 H_D 目录。

B2、Forge 隔离校准、9B 完整 16 档筛选和 LLM+TTS 正式共置均已完成。9B 的 512/K6 与探索性 512/K4+Forge 都因真实 swap 增长失败，正式 15 条矩阵已停止，`H_D` 未冻结。后续主模型改为 9B+真实视觉，FULL 文本/视觉端点已 2/2 通过；4B 作为跨尺度对照。旧 9B 文本与新视觉产物仍须按哈希、环境和驻留状态分开，不能混为一条曲线。

此前计划的 4B 多模态端点门与 VISION_WARM 画像已完成，具体结果见下文。其 artifact 身份、真实 projector 执行与资源证据保持独立，不手工把 `mmproj` 文件大小加进旧曲线或复用 9B 的 H 值。

4B 文本共置矩阵已冻结：`llm_tts_colocated_4b_text_only_20260904_direct_nvml_v1`（30/30）+ 离线复算 `c3_h_4b_tts_colocated_20260906_v1.json`，G_safe=501.282 MiB 校准维持、H_TTS 全部冻结。4B VISION_WARM 多轮画像已完成（`v1_vision_endpoint_4b_warm_profile_20260906_direct_nvml_v1`，5/5）。

图像侧 `H_D` 已按 16G 论文环境以负面边界证据收案：Forge 同日复测（`M_D=5273.21 MiB`）注入 4B screen 后 FULL 因 GPU 容量不可行、H_D 候选为 K16/K14/K14；正式矩阵首个 K16@C512 即触发真实 swap（405.15 MiB）失败即停，与 9B K6/K4 同模式；20G 归因诊断（同代码 3/3 通过、swap 0.0–0.2 MiB）证明根因为 16G 主机内存容量。4B `H_D` 不冻结，与 9B 构成跨尺度负面证据：8GB GPU + 16GB 主机下 LLM+Diffusion 无任何可共置档位（FULL 被 GPU 容量排除、K 档被主机内存排除）；该结论直接支撑图像期采用迁移/逐出而非共驻留的设计，`.wslconfig` 的 `memory=20GB` 诊断配置须恢复默认。

CPU 核数入口现已实现，用户完成 R_sys=2 曲线后，下一步由用户执行 BGE 候选缓存补测（见第 7 节）；完整 R_sys 敏感性及正式并发验收仍待完成。SM-A 仍等待原生 Ubuntu，SM-B/C/D/E 维持暂停。


## 7. 辅助 CPU 核数结果与缓存口径修正（2026-09-06）

- 用户执行 `auxiliary_core_budget/d1_auxiliary_manual_v1/`：R_sys=2，40/40 次、240 个事件；离线复算通过，0 failure、0 swap growth，线程绑定及租约归还通过。
- JSON RAG 1 核 p95=2.35 ms；Whisper 1 核 p95 RTF=0.691，均满足冻结 SLA。BGE 全量重编码 1/2/4 核 p95=1073/667/696 ms，均未达 150 ms；4 核均值改善但尾延迟未稳定改善。
- 旧 BGE 每次重编码 query+32 候选并 padding=128；其 infeasible 不外推到候选已缓存的检索。新增 `bge_cached`，候选准备另计、请求只编码 query 并排序，维持 padding=128 和 SLA=150 ms；静态与假后端验证已通过，真实补测尚未执行。
- 此前助手启动的 `d1_auxiliary_rsys_curve_20260906_v1` 已按用户要求中止，保留 STOPPED_BY_USER 标记，不能拼接为完整四档 R_sys 证据。后续真实模型实验均由用户执行。
- 源码职责、手动命令及指标边界见 [辅助核数画像](AUXILIARY_CORE_BUDGET.md)。
