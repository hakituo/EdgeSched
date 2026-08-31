# 弹性驻留运行结果目录

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
| A1 | `a1/a1_smoke_20260829_verified_v7/` | 当前实验基础设施硬门证据 |
| Audio | `audio_compat/audio_compat_20260829*/` | WSLg/Windows 播放兼容性记录 |
| B1 | `b1/b1_tts_components_20260829_verified_v2/` | T0--T3 单文本组件放置 smoke |
| B2 | `b2/b2_tts_matrix_20260829_verified_v1/` | 旧后端 60/60 TTS 基线；只保留 T0/T3 候选关系，等待 direct NVML 校准 |
| B2 direct v1 | `b2/b2_tts_pareto_matrix_20260831_direct_nvml_v1/` | T0/T3 已完成 30/30、0 failure/fallback、WAV/hash 完整；缺运行期 swap/同步查询耗时且全树 provenance 不干净，仅作工程校准，等待 v2 |
| B2 direct v2 | `b2/b2_tts_pareto_matrix_20260831_direct_nvml_v2/` | 基于干净 `0261e4da` 完成 30/30 技术核验；100 个 child 生命周期 GPU 快照均 direct NVML/无 CLI，最大新鲜度上界 17.582 ms，运行期 swap 0；尚未晋升论文正文数据 |
| C1 | `c1/c1_llm_residency_512_20260829_verified_v1/` | K0/FULL 端点 smoke |
| 9B C2 | `c2/c2_llm_residency_curve_20260829_direct_nvml_v2/` | Qwen3.5-9B 90/90 粗驻留曲线；迁移后作为模型特定压力基线 |
| 9B C3 | `c3/c3_llm_critical_refinement_20260829_direct_nvml_v2/` | Qwen3.5-9B 150/150 临界细化曲线；迁移后作为模型特定压力基线 |
| 9B C3 analysis | `c3_analysis/c3_h_direct_nvml_20260831_v1.json` | 9B 完整 16 档 + B2 direct v2 + Forge direct v1 容量筛选；不得复用为 4B 阈值 |
| 9B C3 LLM+TTS freeze | `c3_analysis/c3_h_tts_colocated_20260831_v1.json` | 复用 9B 正式共置 30 条离线复算；正残差 max=0，冻结 9B `G_safe=501.282 MiB` 与 H_TTS；9B H_D 未冻结 |
| Forge | `forge/forge_b1_reprofile_20260831_direct_nvml_v1/` | 当前 direct NVML 5/5 校准，全部门禁通过，`M_D=5484 MiB` |
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

## 4. 当前禁止混用的口径

1. C2/C3 最终性能曲线只使用 `direct_nvml_v2` 两个批次。
2. KV 容量只使用 `c3_kv_cache_calibration_20260830_direct_nvml_v1`。
3. B2 direct v2 与 Forge direct NVML v1 是当前同后端隔离校准；B2 v1 与 Forge 2026-08-29 v2 只作旧口径基线。9B LLM+TTS 正式共置已冻结该模型口径的 `G_safe/H_TTS`；9B `H_D` 因 swap 门失败不冻结，4B+视觉口径须重新筛选和实测。
4. `_archive/stopped` 中的部分记录即使单轮 `passed=true`，整个矩阵仍不完整，禁止拼接。
5. `timeline.jsonl` 和 `*_child.json` 是原始可追溯证据；清理磁盘前应整体压缩归档，不能只保留 summary 后删除原始轮次。

## 5. 下一批产物

旧 9B C3 H_D 路径已经在两个 smoke 后按 swap 门停止；现有确切 artifact 已列在上表，不再创建 9B 正式 H_D 目录。

B2、Forge 隔离校准、9B 完整 16 档筛选和 LLM+TTS 正式共置均已完成。9B 的 512/K6 与探索性 512/K4+Forge 都因真实 swap 增长失败，正式 15 条矩阵已停止，`H_D` 未冻结。主实验模型迁移到 Qwen3.5-4B-Q4_K_M + 对应 BF16/F16 `mmproj`；9B 产物保留为模型特定压力与负面证据，不能混入 4B 曲线。

下一批不是继续运行旧 9B 命令，而是建立 4B 多模态迁移门：固定主 GGUF/`mmproj` 的来源与哈希，真实执行固定图片，测量 projector GPU/RSS/warm/释放、图片 token、KV、direct NVML 和 swap，并先完成文本/视觉 K0/FULL 端点 smoke。入口、run-id 和正式目录命名尚未实现与冻结；实现前不写占位命令，禁止手工把 `mmproj` 文件大小加进旧曲线或复用 9B 的 H 值。
