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
| C3 analysis | `c3_analysis/c3_h_direct_nvml_20260831_v1.json` | 完整 16 档 + B2 direct v2 + Forge direct v1 当前筛选；colocated 残差 pending，阈值尚未冻结 |
| Forge | `forge/forge_b1_reprofile_20260831_direct_nvml_v1/` | 当前 direct NVML 5/5 校准，全部门禁通过，`M_D=5484 MiB` |
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
3. B2 direct v2 与 Forge direct NVML v1 是当前同后端隔离校准；B2 v1 与 Forge 2026-08-29 v2 只作旧口径基线。冻结 `G_safe/H_TTS/H_D` 前仍须完成真实共置残差。
4. `_archive/stopped` 中的部分记录即使单轮 `passed=true`，整个矩阵仍不完整，禁止拼接。
5. `timeline.jsonl` 和 `*_child.json` 是原始可追溯证据；清理磁盘前应整体压缩归档，不能只保留 summary 后删除原始轮次。

## 5. 下一批产物

下一批真实结果写入（C3 收尾共置 smoke）：

```text
experiment_results/elastic_residency/llm_tts_colocated/
  llm_tts_colocated_c4096_k20_t0_smoke_20260831_v1/
```

B2 direct v2、Forge direct NVML v1 与完整 16 档筛选均已完成技术核验。当前先以 C4096/K20+T0 验证正式共置 artifact、运行期门禁和 T0 残差；它是 4096 context 下的 T0 筛选边界，按隔离曲线和当前 `G_safe` 仍有约 429.6 MiB 预测余量。该 smoke 只证明链路与单组合容量，不替代每候选 5 轮正式矩阵。

调用模板（WSL2，`mvp_core-cu128` venv；无需启动 Forge 服务；运行前以当时干净 Windows HEAD 替换 `<clean_commit>`）：

```bash
cd /home/leslie/mvp_core

/home/leslie/.venvs/mvp_core-cu128/bin/python \
  experiments/elastic_residency/llm_tts_colocated.py \
  --run-id llm_tts_colocated_c4096_k20_t0_smoke_20260831_v1 \
  --llm-configs K20 \
  --tts-configs T0 \
  --contexts 4096 \
  --runs 1 \
  --safety-margin-mb 501.2822265625 \
  --tts-isolated-increment-mb T0=3255.03515625 \
  --max-gpu-sample-age-ms 750 \
  --max-swap-growth-mb 4 \
  --output-root experiment_results/elastic_residency/llm_tts_colocated \
  --source-git-commit <clean_commit> \
  --source-git-dirty false \
  --source-git-status-entry-count 0
```

运行后必须检查 manifest、summary、iterations、failures、WAV、direct NVML、周期与同步快照新鲜度、swap 增长、真实 LLM/TTS 后端、KV 状态和干净 Git provenance；不能只凭最后一行宣布完成。
