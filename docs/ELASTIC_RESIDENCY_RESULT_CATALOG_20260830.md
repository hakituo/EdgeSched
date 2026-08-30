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
| C1 | `c1/c1_llm_residency_512_20260829_verified_v1/` | K0/FULL 端点 smoke |
| C2 | `c2/c2_llm_residency_curve_20260829_direct_nvml_v2/` | 当前 90/90 粗驻留曲线 |
| C3 | `c3/c3_llm_critical_refinement_20260829_direct_nvml_v2/` | 当前 150/150 临界细化曲线 |
| C3 analysis | `c3_analysis/*.json` | 旧 TTS/Forge 口径下的候选筛选；阈值尚未冻结 |
| Forge | `forge/forge_b1_reprofile_20260829_verified_v2/` | 旧整卡采样基线；等待 direct NVML 同后端校准 |
| KV | `kv_calibration/c3_kv_cache_calibration_20260830_direct_nvml_v1/` | 当前 48/48 KV 容量与语义证据 |

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
3. B2 v1 和 Forge v2 当前只作为旧后端基线；冻结 `G_safe/H_TTS/H_D` 前必须完成 direct NVML 同后端校准。
4. `_archive/stopped` 中的部分记录即使单轮 `passed=true`，整个矩阵仍不完整，禁止拼接。
5. `timeline.jsonl` 和 `*_child.json` 是原始可追溯证据；清理磁盘前应整体压缩归档，不能只保留 summary 后删除原始轮次。

## 5. 下一批产物

下一批正式结果写入（即 C3 收尾第 1 步，下一项正式实验）：

```text
experiment_results/elastic_residency/b2/
  b2_tts_pareto_matrix_20260831_direct_nvml_v1/
```

它只测已由 B2 旧矩阵保留下来的 T0/T3，固定 20/50/100 字、每配置每句长 5 轮，共 30 条。完成并核验后，再进行 Forge direct NVML 校准和真实共置候选筛选。

调用模板（WSL2，`mvp_core-cu128` venv；运行前将 `<clean_commit>` 替换为当时 Windows 干净 HEAD）：

```bash
cd /home/leslie/mvp_core

/home/leslie/.venvs/mvp_core-cu128/bin/python \
  experiments/elastic_residency/tts_component_offload_matrix.py \
  --run-id b2_tts_pareto_matrix_20260831_direct_nvml_v1 \
  --configs T0 T3 \
  --rounds 5 \
  --output-root experiment_results/elastic_residency/b2 \
  --source-git-commit <clean_commit> \
  --source-git-dirty false \
  --source-git-status-entry-count 0
```

该批次只解决 C3 的同后端 TTS 容量口径；SM-A 的原生 Linux 恢复属于独立门禁，不混入本批次。
