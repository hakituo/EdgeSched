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
| C2 | `c2/c2_llm_residency_curve_20260829_direct_nvml_v2/` | 当前 90/90 粗驻留曲线 |
| C3 | `c3/c3_llm_critical_refinement_20260829_direct_nvml_v2/` | 当前 150/150 临界细化曲线 |
| C3 analysis | `c3_analysis/c3_h_direct_nvml_20260831_v1.json` | 完整 16 档 + B2 direct v2 + Forge direct v1 容量筛选；保留为正式共置前输入 |
| C3 LLM+TTS freeze | `c3_analysis/c3_h_tts_colocated_20260831_v1.json` | 复用正式共置 30 条离线复算；正残差 max=0，`G_safe=501.282 MiB` 与 H_TTS 已冻结，H_D pending LLM+Forge |
| Forge | `forge/forge_b1_reprofile_20260831_direct_nvml_v1/` | 当前 direct NVML 5/5 校准，全部门禁通过，`M_D=5484 MiB` |
| KV | `kv_calibration/c3_kv_cache_calibration_20260830_direct_nvml_v1/` | 当前 48/48 KV 容量与语义证据 |
| LLM+TTS smoke | `llm_tts_colocated/llm_tts_colocated_c4096_k20_t0_smoke_20260831_v1/` | C4096/K20+T0 1/1 技术通过，0 failure/fallback；正残差 0，只作正式 artifact smoke |
| LLM+TTS formal | `llm_tts_colocated/llm_tts_colocated_candidates_20260831_direct_nvml_v1/` | 6 候选×5 轮，30/30、0 failure/fallback；真实后端/direct NVML/新鲜度/swap/provenance 全通过；原始运行版容量字段有 idle 重复计数，测量无需重跑，以离线复算为准 |

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
3. B2 direct v2 与 Forge direct NVML v1 是当前同后端隔离校准；B2 v1 与 Forge 2026-08-29 v2 只作旧口径基线。LLM+TTS 正式共置已冻结 `G_safe/H_TTS`；`H_D` 仍须真实 LLM+Forge 共置。
4. `_archive/stopped` 中的部分记录即使单轮 `passed=true`，整个矩阵仍不完整，禁止拼接。
5. `timeline.jsonl` 和 `*_child.json` 是原始可追溯证据；清理磁盘前应整体压缩归档，不能只保留 summary 后删除原始轮次。

## 5. 下一批产物

下一批真实结果写入（C3 H_D 收尾）：

```text
experiment_results/elastic_residency/llm_forge_colocated/
  llm_forge_colocated_c512_k6_smoke_<date>_v1/
  llm_forge_colocated_c512_k4_fallback_smoke_<date>_v1/
  llm_forge_colocated_h_d_<date>_direct_nvml_v1/
```

B2、Forge 隔离校准、完整 16 档筛选和 LLM+TTS 正式共置均已完成。512/K6+Forge 单轮 smoke 已形成有效真实运行证据，但因 swap 增长约 290.44 MiB 判失败；正式 15 条已按门禁停止。下一步仅用显式探索开关运行 screen 已画像的 512/K4，入口仍从 screen/Forge summary 读取 `M_L/M_D/G_safe/B_idle`，不允许手填另一口径，也不得把降档结果直接写成冻结 H_D。

Forge 服务必须在独立 `forge-cu128` 终端保持开启；实验命令使用 `mvp_core-cu128`：

```bash
cd /home/leslie/mvp_core

/home/leslie/.venvs/mvp_core-cu128/bin/python \
  experiments/elastic_residency/llm_forge_colocated.py \
  --run-id llm_forge_colocated_c512_k4_fallback_smoke_<date>_v1 \
  --candidates 512:K4 --allow-exploratory-fallback \
  --runs 1 --warmup 1 \
  --max-gpu-sample-age-ms 750 \
  --max-swap-growth-mb 4 \
  --output-root experiment_results/elastic_residency/llm_forge_colocated \
  --source-git-commit <clean_commit> \
  --source-git-dirty false \
  --source-git-status-entry-count 0
```

运行后必须检查 manifest、summary、iterations、failures、真实 PNG、child log、真实 LLM+Forge 后端、direct NVML、周期与同步快照新鲜度、swap、KV 状态、容量裁决和完整 Git provenance；不能只凭最后一行宣布完成。smoke 未通过前不得运行正式 15 条，也不得冻结 H_D。
