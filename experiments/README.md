# EdgeSched 实验指南

本目录包含实验框架、论文主实验、语义验证、压力测试和图表生成脚本。实验的第一原则是区分 Mock、真实 LLM、真实 Forge 和论文正文数据。

## 目录索引

| 目录 | 内容 |
|---|---|
| `framework/` | 路径、配置、统计、随机种子、环境采集、Mock/真实适配器 |
| `migration/` | EdgeSched 非对称迁移、对称迁移和消融策略 |
| `baselines/` | cold start、LRU thrashing、manual switching |
| `admission/` | 准入策略和动态任务画像对比 |
| `recovery/` | 崩溃恢复主实验 |
| `semantic_validation/` | 取消语义、StateBackplane 恢复语义 |
| `stress/` | 大规模与真实混合负载压力 |
| `workload/` | 真实交互 workload 和 trace |
| `phase_aware/` | 迁移避免、策略对比和 N 阶段 workflow |
| `overhead/` | 实现开销、综合实验、论文消融套件 |
| `charts/` | JSON 到 PNG/PDF/CSV/LaTeX 图表 |
| `scaling/` | Forge batch scaling 特性曲线 (T(b)/吞吐/加速比) |
| `tts/` | Qwen3-TTS Base 独立 CPU/GPU smoke 与 10 块 prompt 复用验收 |
| `elastic_residency/` | 弹性驻留实验契约、物理核租约、资源与调度 jitter 采样；后续承载 T0–T3、LLM 驻留和动态核心预算 |
| `elastic_residency/sm/` | GPU 计算份额（SM-A）子包：数据契约、MPS 生命周期、客户端探针、探针编排与 Green Context 只读探测 |

## 推荐入口

### 1. 无 Forge 实验

```bash
python experiments/run_no_forge_experiments.py --iterations 5
python experiments/run_no_forge_experiments.py \
  --experiments lru_thrashing,crash_recovery --iterations 5
```

集合包含：`lru_thrashing`、`crash_recovery`、`stress_test`、`baseline_comparison`。

### 2. Forge 实验

```bash
python experiments/run_forge_experiments.py --iterations 5
python experiments/run_forge_experiments.py --experiments migration --iterations 5
```

集合包含：`edgesched_migration`、`symmetric_migration`、`fast_down_naive_up`、`naive_down_optimized_up`、`real_workload`、`realistic_stress`。

### 3. 统一论文编排

```bash
python experiments/run_all_experiments.py --experiments all --iterations 5
python experiments/run_all_experiments.py --experiments core --iterations 5
```

`run_all_experiments.py` 的 `all` 当前包含迁移、cold start、LRU、恢复、压力、实现开销和真实 workload；Forge 不可用时不要使用 `--skip-forge-check` 把回退结果当作真实结果。

### 4. 性能套件

```bash
python experiments/run_performance_suite.py --workload mock
python experiments/run_performance_suite.py --workload real
```

Mock 用于验证异步链路、统计字段、背压和回归；Real 才用于真实吞吐、尾延迟、显存峰值和 trace。运行时产物默认写入 `experiment_results/`。

### 5. Phase-Aware

```bash
python experiments/phase_aware/exp1_migration_avoidance.py
python experiments/phase_aware/exp2_scheduling_comparison.py
python experiments/phase_aware/exp3_n_phase_workflow.py
```

`exp1_real_migration_avoidance.py` 涉及真实迁移，必须在 WSL2 模型环境单独核对，不要与纯策略模拟混合报告。

### 6. 独立 TTS

```bash
python experiments/tts/smoke_qwen3_tts_base.py --device cuda:0 --ref-text '准确转写'
python experiments/tts/smoke_qwen3_tts_base.py --device cpu --cpu-threads 8 --x-vector-only
python experiments/tts/profile_qwen3_tts.py --device cuda:0 --x-vector-only
python experiments/tts/profile_qwen3_tts_gpu_sentence_lengths.py \
  --device cuda:0 --x-vector-only --rounds 3 \
  --git-root /mnt/d/AI/xiaoyou-core/legacy/mvp_core
```

默认 ICL 必须提供参考音频准确转写；`--x-vector-only` 仅用于显式消融。smoke 产物位于 `experiment_results/tts/smoke/<run_id>/`，句长画像位于 `experiment_results/tts/profiling/gpu_sentence_length/<run_id>/`。指标使用 `first_chunk_ready`，不声称第一音频采样延迟。这里验证独立句子级 TTS，不证明 LLM→TTS 流水线或 Diffusion 共置性能。

### 7. 弹性驻留 A1 基础设施

```bash
python -m experiments.elastic_residency.smoke_a1 \
  --run-id <new_run_id> --duration 2.0 \
  --source-git-commit <windows_source_commit> \
  --source-git-dirty false \
  --source-git-status-entry-count 0
```

A1 产物位于 `experiment_results/elastic_residency/a1/<run_id>/`，包含冻结配置、环境 manifest、单轮原始 JSONL、事件、统一资源时间线、JSON/CSV 汇总和失败记录。WSL2 运行副本按规则不带 `.git`，因此正式运行必须把 Windows 源仓库 commit/dirty 快照显式传入；干净源码还必须声明状态项为 0。dirty 源码只有同时提供 `--source-git-snapshot <archive_or_patch>`（可选再传声明哈希）并把快照复制进运行目录后，provenance 才完整。首个 CPU 样本会标为无效并从聚合排除，资源时间线开始前会等待首个成功 GPU 样本；GPU 可用性/新鲜度、资源与事件循环 jitter、swap、affinity 和核心归还均属于 `passed` 硬门槛。该 smoke 不运行 TTS/LLM，因此不能作为后续性能目标或无 underrun 的证据。

当前通过硬门槛的 A1 记录为 `a1_smoke_20260829_verified_v7`，源 commit 为 `ae68608`。它只用于验收实验基础设施。

### 8. 弹性驻留 B1 TTS 组件 smoke

```bash
python -m experiments.elastic_residency.tts_component_offload \
  --run-id <new_run_id> --configs T0 T1 T2 T3 \
  --source-git-commit <clean_commit> \
  --source-git-dirty false --source-git-status-entry-count 0
```

每个配置在独立子进程中加载真实 Qwen3-TTS，记录 talker、speech codec、speaker encoder 和 prompt 的实际设备，分别测量 tokenize、talker、code transfer、codec decode，并保存整卡/进程估计/PyTorch 显存、CPU/RSS 和非静音 WAV 检查。若 WSL 驱动把进程显存报告为 `[N/A]`，结果必须保留 unavailable 原因，只能把相对空闲整卡基线的增量标为 estimate。当前通过记录为 `b1_tts_components_20260829_verified_v2`，源 commit 为 `38f4d81`。它是单文本冷态可行性 smoke，不代替 B2 多轮矩阵。

### 9. 弹性驻留 B2 TTS 正式矩阵

```bash
python -m experiments.elastic_residency.tts_component_offload_matrix \
  --run-id <new_run_id> --configs T0 T3 --rounds 5 \
  --max-gpu-sample-freshness-ms 750 \
  --swap-sample-interval 0.05 --max-swap-growth-mb 0 \
  --source-git-commit <clean_commit> \
  --source-git-dirty false --source-git-status-entry-count 0
```

B2 固定使用恰好 20/50/100 字文本，每配置独立加载和预热一次，然后运行 15 个测量请求。`tts_runtime_evidence.py` 在整个配置子进程生命周期内按固定 deadline 保存 swap 原始时间线；每个显存点记录同步 direct NVML 查询的开始、结束与耗时，并以查询耗时作为样本新鲜度保守上界。任一 swap 增长、查询上界超过 750 ms、CLI/fallback 或原有 waveform/device 门禁失败，都会使 iteration、child 和 summary 同时失败并写入 `failures.jsonl`。

旧 60 条 `b2_tts_matrix_20260829_verified_v1` 只保留 T0/T3 候选关系。T0/T3 direct NVML v1 `b2_tts_pareto_matrix_20260831_direct_nvml_v1` 缺运行期 swap/查询耗时字段且全树 provenance 不干净，只作为工程校准；v2 `b2_tts_pareto_matrix_20260831_direct_nvml_v2` 已基于干净 `0261e4da` 完成 30/30 技术核验，0 failure/fallback，但尚未人工晋升论文正文数据。全部既有 B2 记录的 RTF 均大于 1，因此不能证明无 underrun。

### 10. 弹性驻留 C1/C2 LLM 驻留曲线

```bash
python -m experiments.elastic_residency.llm_residency_curve \
  --run-id <new_run_id> --configs K0 FULL --runs 1 \
  --context-tokens 512 --generation-tokens 128 \
  --source-git-commit <clean_commit> \
  --source-git-dirty false --source-git-status-entry-count 0
```

入口从 GGUF 元数据读取实际 block 数；`K<number>` 映射到明确的 `n_gpu_layers`，`FULL` 单独使用 llama.cpp 完整 offload 语义。每配置在独立子进程中固定 14 个物理核，构造精确 token 数 prompt，手工分离 prefill/decode，记录每个 token gap、阶段资源指标、KV save/restore 和卸载。推理期 GPU 新鲜度与采样 jitter 是硬门槛；KV save/restore 的短时尖峰单独记录为阶段成本和质量告警。

C1 权威记录 `c1_llm_residency_512_20260829_verified_v1` 基于干净 commit `234d2df`，K0/FULL 均为 19/19 项通过。它只有 512-token、每端点 1 轮，是 profiler 与端点 smoke，不是 C2 正式曲线或论文性能证据；详细结果见 [C1 LLM 驻留端点 smoke](../docs/LLM_RESIDENCY_C1_20260829.md)。

C2 使用一个不可覆盖目录承载完整矩阵，并在每个子轮后立即追加 checkpoint：

```bash
python -m experiments.elastic_residency.llm_residency_curve \
  --run-id <new_c2_run_id> \
  --output-root experiment_results/elastic_residency/c2 \
  --configs K0 K8 K16 K24 K32 FULL \
  --contexts 512 2048 4096 --runs 5 \
  --generation-tokens 128 --jitter-gate-mode observe \
  --source-git-commit <clean_commit> \
  --source-git-dirty false --source-git-status-entry-count 0
```

配置在每个轮次内确定性轮转，降低固定顺序热漂移。意外中断时，只能用完全相同的参数、源码 commit、Python/平台和模型 manifest 加 `--resume`；已有 `(context, config, run)` 会跳过，矩阵外或重复记录会拒绝。子轮超时、进程失败或测量有效性验收失败默认立即停止，总汇总失败时命令返回非零。C2 的 jitter 阈值是待测性能目标而非证据有效性门，因此正式矩阵必须显式使用 `observe`：超标轮次写入 `performance_target_checks` 并保留在完整曲线中，不会被隐藏或改写为通过。若 `failures.jsonl` 非空，该 run 即使后来补齐也不能成为通过记录，修复后必须换新 run id。

正式 GPU sampler 通过 ctypes 直接调用 NVML，不启动 `nvidia-smi` 子进程，也不做静默 CLI fallback；manifest、summary 和每轮验收必须记录 `gpu_telemetry_backend=direct_nvml` 与 `gpu_cli_subprocess_used=false`。GPU 数据最大年龄 750 ms 仍是测量有效性硬门。周期探针若越过下一采样 tick，完成后必须立即补采，禁止继续等待固定网格而人为放大陈旧窗口；`gpu_deadline_recovery_count` 记录这种恢复。驻留曲线的显存查询必须位于辅助 utilization/power/temperature 查询之后，使样本时间戳紧邻显存读数。WSL 若无法返回数值化的 compute-process 显存，必须保留 unavailable 原因；`gpu_card_delta_from_baseline_mb` 只表示隔离运行下的整卡差值，不得称为精确进程显存。TTS 另以 PyTorch allocated/reserved 峰值作为框架内主证据。
更早的 B2/Forge/C2/C3 结果使用旧采样口径，只保留为工程基线和链路证据。C2 direct v2（90 条）、C3 direct v2（150 条）、B2 direct v2（30 条）与 Forge direct NVML v1（5 条）已完成同后端技术核验；下一步只用这些当前批次重新筛选候选，不能跨后端拼接为论文阈值。
冻结的 10 档临界矩阵（K2/K4/K6/K10/K12/K14/K18/K20/K28/K30、3 上下文、5 轮、128 生成 token、observe）标记为 `c3_critical_refinement`；建议显式使用 `--output-root experiment_results/elastic_residency/c3`，避免与 C1 smoke 产物混放。有效性门失败时控制台会直接打印失败检查名。

### 10.1 Forge b=1 direct NVML 同后端校准

Forge 服务需先在独立的 `forge-cu128` 环境启动；校准入口在 `mvp_core-cu128` 中调用 HTTP API，并用 direct NVML 周期采集整卡高水位：

```bash
python -m experiments.elastic_residency.forge_b1_reprofile \
  --run-id forge_b1_reprofile_20260831_direct_nvml_v1 \
  --iterations 5 --warmup 1 \
  --max-gpu-sample-age-ms 750 --max-swap-growth-mb 4 \
  --source-git-commit <clean_commit> \
  --source-git-dirty false --source-git-status-entry-count 0
```

`forge_runtime_evidence.py` 对参与 `M_D` 计算的 idle baseline、resident baseline 和每个正式 measurement 窗口分别检查 GPU 样本年龄；任一窗口无有效年龄样本或超过 750 ms，都会使 `summary.passed=false` 并写入 `failures.jsonl`。direct NVML 后端和样本数通过但样本过期，仍视为无效批次。旧 `forge_b1_reprofile_20260829_verified_v2` 只保留为旧口径工程基线。

正式记录 `forge_b1_reprofile_20260831_direct_nvml_v1` 已完成 5/5，`summary.passed=true`，`M_D=5484 MiB`。完成后可关闭 Forge 服务；下一步是纯离线的完整 16 档候选筛选：

```bash
python experiments/elastic_residency/analyze_results.py \
  --c2-summary experiment_results/elastic_residency/c2/c2_llm_residency_curve_20260829_direct_nvml_v2/summary.json \
  --c3-summary experiment_results/elastic_residency/c3/c3_llm_critical_refinement_20260829_direct_nvml_v2/summary.json \
  --expected-llm-config-count 16 \
  --b2-summary experiment_results/elastic_residency/b2/b2_tts_pareto_matrix_20260831_direct_nvml_v2/summary.json \
  --forge-summary experiment_results/elastic_residency/forge/forge_b1_reprofile_20260831_direct_nvml_v1/summary.json \
  --tts-configs T0,T3 \
  --output experiment_results/elastic_residency/c3_analysis/c3_h_direct_nvml_20260831_v1.json
```

`analysis_input_merge.py` 要求 C2/C3 均通过、context 集合一致、档位互斥，并在每个 context 合并为恰好 16 档；不满足时命令失败且不产生可用筛选结果。该输出仍是共置 smoke 的候选清单，不是冻结阈值。

### 11. KV Cache 容量校准（独立入口，不影响 C2/C3 曲线）

`llm_residency_curve.py` v3 起在推理计时窗口之外增补 `full_state_serialized_bytes`、`kv_sequence_state_bytes`、`kv_position_min/max`、`kv_tokens_used`、`kv_gpu_buffer_mb`、`kv_cpu_buffer_mb`、`compute_gpu_buffer_mb`、`compute_cpu_buffer_mb` 等字段，语义由 `experiments/elastic_residency/kv_measurement.py` 的 `kv_measurement_semantics` 固定；历史 C2/C3 记录不含这些字段但仍然有效，不需要重跑。`kv_state_bytes` 保留为兼容字段，其语义是含 logits、embedding 与 KV memory 的完整序列化状态，不是纯 GPU KV 显存。

KV buffer 容量由独立校准入口提供（16 configs × 3 contexts × 1 run = 48 条确定性容量数据，KV 分配不依赖随机性因此无需多轮）：

```bash
python -m experiments.elastic_residency.llm_kv_cache_calibration \
  --run-id <new_run_id> \
  --output-root experiment_results/elastic_residency/kv_calibration \
  --source-git-commit <clean_commit> \
  --source-git-dirty false --source-git-status-entry-count 0
```

入口复用 C2 的模型初始化参数，`n_ctx` 与填充状态均为 `context + generation`（默认 generation=128），对应 C2/C3 完成 128-token 生成后的状态；同时复用 CPU affinity、direct NVML 与 artifact contract。入口不做性能计时，KV 查询全部在 eval 之后。每条记录验证层数、状态 token 数、KV position、序列状态与完整状态大于 0、buffer 日志捕获、direct NVML、CLI=false、swap=0、资源归还与 Git provenance。

正式记录 `c3_kv_cache_calibration_20260830_direct_nvml_v1` 基于干净 commit `820ee0fbcbbf671a791d31786bf171bf2a241d1a` 完成 48/48 条、0 failure。`n_ctx/KV tokens=640/2176/4224` 时，预分配 KV buffer 总量为 `24/72/136 MiB`，FULL 的 GPU KV 部分同为 `24/72/136 MiB`；完整序列化状态为 `70.263/118.292/182.331 MiB`。该结果只补齐容量语义，不把单轮确定性校准当作性能重复实验。

### 12. LLM+TTS 真实共置显存测量

`experiments/elastic_residency/llm_tts_colocated.py` 在同一进程加载真实 GGUF LLM（填充 context 后记录整卡基线 `B_LLM`）与 Qwen3-TTS（T0/T3 放置，分阶段 `torch.cuda.synchronize()` + reset peak），TTS 执行窗口内由 direct NVML 时间线和阶段快照取整卡峰值 `P_colocated`，并计算 `tts_colocated_increment_mb = P_colocated - B_LLM` 与 `colocated_positive_residual_mb = max(0, colocated_increment - isolated_increment)`（隔离增量由 `--tts-isolated-increment-mb CONFIG=value` 注入，缺失时保持 None，禁止估计）。`G_safe` 已包含 idle GPU baseline high water，因此容量裁决使用 `(P_colocated - B_idle) + G_safe <= gpu_total`，并另以 `P_colocated <= gpu_total` 检查物理峰值；直接对绝对峰值再加完整 `G_safe` 会重复计算 idle。`M_L(k, context)` 已包含该 LLM 配置的 KV buffer，容量计算不得再叠加一次 KV。共置候选由 C2+C3 新曲线分析后确定，先 smoke 后小矩阵：

```bash
python -m experiments.elastic_residency.llm_tts_colocated \
  --run-id <new_run_id> \
  --llm-configs <K...> --tts-configs T0 T3 --contexts 2048 --runs 5 \
  --max-gpu-sample-age-ms 750 --max-swap-growth-mb 4 \
  --tts-isolated-increment-mb T0=<isolated_mb> T3=<isolated_mb> \
  --source-git-commit <clean_commit> \
  --source-git-dirty false --source-git-status-entry-count 0
```

共置入口对 TTS 执行窗口的周期 direct NVML 样本和参与 `B_LLM/P_colocated` 的同步容量快照分别执行 750 ms 新鲜度硬门；swap 按全生命周期最大值相对起点的增长计算，默认容忍 4 MiB 的 WSL2 系统噪声，数百 MiB 级真实换页仍会失败。任何验收检查失败都会写入 `failures.jsonl`，不能只看 summary 最后一行。

正式 6 候选×5 轮矩阵 `llm_tts_colocated_candidates_20260831_direct_nvml_v1` 已完成 30/30、0 failure/fallback。原始测量有效，但其运行时版本把绝对 `P_colocated` 再加完整 `G_safe`，对 15 条 FULL+T3 记录产生了 idle 重复计数的旧判定。无需重跑模型；`analyze_colocated_results.py` 复用原始 `B_idle/B_LLM/P_colocated` 离线重算，输出 `c3_h_tts_colocated_20260831_v1.json`：六个候选全部容量可行，colocated 正残差最大值为 0，`G_safe=501.282 MiB`，按 context 的 `H_TTS` 已冻结。`H_D` 仍须真实 LLM+Forge 共置，当前不得冻结。

筛选后的正式矩阵使用 `--candidates CONTEXT:LLM:TTS ...` 显式列出候选；该参数覆盖 `--llm-configs/--tts-configs/--contexts` 的笛卡尔积，并按给定顺序执行每候选的重复轮次。最紧候选应放首位配合默认 fail-fast。正式 manifest 同时保存 GGUF 哈希、完整 TTS 模型树哈希及逐文件清单、参考音频哈希，不能只用 `config.json` 代替模型 provenance。

### 12.1 LLM+Forge 真实共置与 H_D 冻结

`llm_forge_colocated.py` 强制读取完整 C3 screen 与同 run 的 Forge direct NVML summary，只接受 512/K6、2048/K4、4096/K4 三个筛选边界。父进程先通过真实 Forge API warmup；每轮独立 child 在 Forge 常驻时加载 GGUF、填充 `context+128` KV 状态，再执行固定 1024×1024、8 steps、Euler、b=1 请求。每轮保存真实 PNG、`B_Forge/B_LLM/P_colocated`、KV、direct NVML 时间线、同步快照新鲜度、swap、affinity、资源归还和失败记录。Forge 终端必须在整个 smoke/矩阵期间保持运行。

Forge 服务终端：

```bash
cd /home/leslie/mvp_core/stable-diffusion-webui-forge-main
source /home/leslie/.venvs/forge-cu128/bin/activate
python launch.py --api --listen --port 7860 \
  --skip-version-check --skip-python-version-check
```

先运行最紧的 512/K6 单组合 smoke：

```bash
cd /home/leslie/mvp_core
/home/leslie/.venvs/mvp_core-cu128/bin/python \
  experiments/elastic_residency/llm_forge_colocated.py \
  --run-id llm_forge_colocated_c512_k6_smoke_<date>_v1 \
  --candidates 512:K6 --runs 1 --warmup 1 \
  --max-gpu-sample-age-ms 750 --max-swap-growth-mb 4 \
  --output-root experiment_results/elastic_residency/llm_forge_colocated \
  --source-git-commit <clean_commit> \
  --source-git-dirty false --source-git-status-entry-count 0
```

smoke 的 manifest、summary、iteration、failure、PNG、child log、direct NVML、新鲜度、swap、KV、真实后端和 provenance 全部通过后，才运行 3 候选×5 轮正式矩阵；按预测安全余量从紧到松排列并默认 fail-fast：

```bash
/home/leslie/.venvs/mvp_core-cu128/bin/python \
  experiments/elastic_residency/llm_forge_colocated.py \
  --run-id llm_forge_colocated_h_d_<date>_direct_nvml_v1 \
  --candidates 512:K6 4096:K4 2048:K4 --runs 5 --warmup 1 \
  --max-gpu-sample-age-ms 750 --max-swap-growth-mb 4 \
  --output-root experiment_results/elastic_residency/llm_forge_colocated \
  --source-git-commit <clean_commit> \
  --source-git-dirty false --source-git-status-entry-count 0
```

正式 15 条通过后，`analyze_forge_colocated_results.py` 合并已有 TTS freeze 与 Forge 共置正残差，重新检查 H_TTS 安全余量并冻结最终 `G_safe/H_TTS/H_D`。该分析仍写入 `experiment_results`，不会自动晋升 `paper/experiment_data/01_paper_main/`。

WSL2 音频优先使用 WSLg Pulse；若当前会话无法枚举输出设备，可自动或显式回退到 Windows WASAPI：

```bash
python -m experiments.elastic_residency.audio_playback_probe \
  --wav <pcm16_wav> --backend auto --output-json <summary.json>
python -m experiments.elastic_residency.audio_playback_probe \
  --wav <pcm16_wav> --backend windows --output-json <summary.json>
```

探针记录源/输出采样率、设备、host API、callback 数、状态、underflow 和最大 callback gap；不支持的 24 kHz 单声道会显式协商到设备格式。静音 smoke 只验证输出链路，正式 underrun 结论必须来自真实 TTS chunk 和并发工作流。

### 13. 弹性驻留 SM-A：GPU 计算份额能力探针

对应论文指南 P1-C / SM-A。当前状态：**真实 WSL2 probe 已完成，结论为 NO_GO**。
正式记录 `sm_a_20260830_direct_nvml_v1` 基于干净 commit
`ca7a7ff0832342fc76f261b6b7d8d720b5a116fa`：普通 CUDA 基线成功，识别到
RTX 5070 Laptop 的 36 个物理 SM，direct NVML 与 Git provenance 均完整，0 failure、
0 fallback、0 cleanup failure；但 WSL2 中 `nvidia-cuda-mps-control` 不存在，
四个 percentage 均按 `UNSUPPORTED` 记录。因此停止 SM-B/SM-C，不把普通 CUDA
共享写成受控 SM 分配，也不自动转向 Green Context 生产实现。

```bash
python -m experiments.elastic_residency.sm.sm_capability \
  --run-id <new_run_id> \
  --percentages 100 75 50 25 \
  --output-root experiment_results/elastic_residency/sm_capability \
  --source-git-commit <clean_commit> \
  --source-git-dirty false --source-git-status-entry-count 0
```

模块位于 `experiments/elastic_residency/sm/`：

| 模块 | 职责 |
|---|---|
| `sm_capability.py` | SM-A 探针入口与编排 |
| `sm_profiles.py` / `sm_share_profile.py` / `sm_partition_rules.py` | 数据契约、profile 与独占 SM 预算规则 |
| `mps_commands.py` | 控制命令原语：stdin 编码、`CommandSpec`/`CommandResult`、解析、PID 文件与存活 |
| `mps_runtime.py` / `mps_daemon.py` / `mps_queries.py` / `mps_shutdown.py` / `mps_artifacts.py` | 启动、就绪判定、查询、关闭阶段身份核对、日志归档与清理 |
| `sm_client_probe.py` / `cuda_exec_affinity.py` / `sm_client_workload.py` | 独立进程客户端、CUDA Driver 只读查询与短 CUDA workload |
| `sm_client_process.py` / `sm_client_runner.py` | 子进程执行与逐 percentage 编排 |
| `sm_probe_env.py` / `sm_probe_records.py` / `sm_probe_verdict.py` | 环境采集、记录构造与 Go/No-Go 判定 |
| `green_context_types.py` / `green_context_probe.py` | Green Context 符号表/结构体自检与只读能力探测 |

产物位于 `experiment_results/elastic_residency/sm_capability/<run_id>/`，包含 `run_config.json`、`manifest.json`、`iterations.jsonl`、`events.jsonl`、`summary.json`、`summary.csv`、`failures.jsonl`，以及 `client_stdio/`（每个客户端子进程的 stdout/stderr/JSON）与 `mps_logs/`（清理前归档的 daemon 日志）。

执行顺序与口径（`nvidia-cuda-mps-control -d` 只拉起 control daemon，MPS server 由首个 CUDA client
按需启动，因此**不能在启动阶段等待 server 出现**）：

1. 普通 CUDA 不受控共享基线（显式移除 MPS 环境变量），得到 `physical_total_sms`；
2. `nvidia-cuda-mps-control` 可用性检查；
3. 每个 percentage：启动 control daemon → 等待私有 pipe 目录中的
   `nvidia-cuda-mps-control.pid`（controller 就绪证据）→ 启动**全新客户端子进程**；
4. 客户端保持 context 存活期间，父进程轮询 `get_client_list` / `get_server_list` 证明客户端真的连上 MPS；
5. 客户端退出后经 **stdin** 发送 `quit`：
   - controller 仍在时继续用控制协议确认 server 清空；
   - controller PID/PID 文件消失后**不再调用 `get_server_list`**，改为核对退出前
     记录的 server 进程身份（PID + starttime/executable）是否仍存活；
   - 只有 controller 与所有已观察 server 都确认消失才 `reached_quiescent=True`；
     进程身份无法确认时 fail-closed 并保留私有目录与结构化原因；
6. 清理本次 run 的私有目录；
7. 输出结构化 report 与 Go/No-Go 判定。

控制协议必须走 interactive stdin：

```text
subprocess.run([mps_binary], input="get_server_list\n", ...)
```

不能执行 `nvidia-cuda-mps-control get_server_list`；仍然禁止 `shell=True`。
`CommandResult` 分别记录 `argv`、`stdin_command`、`returncode`、stdout/stderr 与超时/错误。
可选 `--mps-start-server` 会经 stdin 发送 `start_server -uid <当前 uid>`（默认由首个 client 按需拉起，
保持同一用户语义）。

必须遵守的边界：

- 物理 SM 与 context 可见 SM 分开：基线给出 `physical_total_sms`，MPS 客户端的
  `multi_processor_count` 只记为 `context_visible_sm_count`（可能被 active-thread percentage 限制），
  所有 `requested_sm_count` 都用基线物理总数计算，MPS 客户端属性不得覆盖物理总量。
- `effective_sm_count` 只接受客户端 CUDA context 通过 `cuCtxGetExecAffinity` 得到的实测值。
  该 API 的真实签名是 `cuCtxGetExecAffinity(CUexecAffinityParam*, CUexecAffinityType)`，
  第二个参数是整数枚举 `CU_EXEC_AFFINITY_TYPE_SM_COUNT = 1`，不是元素个数指针；
  `CUexecAffinityParam` 的 `type`/`numSms` 偏移由 C ABI 保证，联合体整体大小未与 CUDA 头文件核对，
  结果中保留 `layout.union_total_size_verified = false`。查询失败时保持 `None`，禁止用 0 或比例推算冒充。
  返回的原始 return code 只是驱动响应证据，**不能**在没有 WSL 实测前写成环境能力结论。
- 第一版使用 `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`，能力只能标记为 `MPS_CAP`（客户端执行资源上限），**不是**独占 SM 分区，不参与“SM 数之和 <= 物理 SM”的准入判断。
- 只有 `MPS_STATIC` / `GREEN_CONTEXT` 且验证成功时才按空间分区处理；当前没有可用的静态分区后端。
- 后端不存在、daemon 启动失败或份额无法验证时输出 `UNSUPPORTED`，绝不静默退回普通 CUDA；`summary.fallback_count` 必须为 0。
- 显存限制与 SM 限制分别建模。MPS 下整卡 direct NVML 不是单个客户端的精确显存，客户端显存另记 PyTorch `allocated/reserved`。
- 正式 GPU 遥测复用 `DirectNvmlProbe`，`manifest.gpu.gpu_cli_subprocess_used` 恒为 `false`，不启动 `nvidia-smi`。

只读的 Green Context 能力探测单独提供入口（默认结论为 `UNSUPPORTED`，不创建任何 Green Context）：

```bash
python -m experiments.elastic_residency.sm.green_context_probe --output-json <path>
```

它校验的 stream 符号是 `cuGreenCtxStreamCreate`（不存在 `cuStreamCreateWithGreenCtx`）；
`CUdevResource` 布局未与 CUDA 头文件核对，设备资源查询只记为
`device_sm_count_unverified` 并保持 `UNSUPPORTED / layout_unverified`，不崩溃也不宣称支持。

单元测试（不依赖 GPU）与 WSL GPU 集成入口分开：

```bash
python -m pytest tests/elastic_residency/test_sm_profiles.py \
  tests/elastic_residency/test_sm_capability.py \
  tests/elastic_residency/test_sm_probe_records.py \
  tests/elastic_residency/test_mps_commands.py \
  tests/elastic_residency/test_mps_runtime.py \
  tests/elastic_residency/test_mps_daemon.py \
  tests/elastic_residency/test_mps_daemon_cleanup.py \
  tests/elastic_residency/test_cuda_exec_affinity.py \
  tests/elastic_residency/test_green_context_probe.py -q

# 仅在 WSL2 + CUDA + MPS 环境显式开启
EDGESCHED_RUN_WSL_GPU_TESTS=1 python -m pytest \
  tests/elastic_residency/test_sm_capability_wsl_gpu.py -q
```

WSL 集成测试不允许“实现自己造成的 No-Go”通过。MPS binary 存在时会同时检查：

- `summary["unsupported_reasons"]`；
- 每项 `summary["sm_evidence"][*]["failure"]`（**环境白名单以外的任何非空失败都判失败**，
  不只匹配已知 marker）；
- 每条受控记录的 `daemon.cleanup_succeeded`；
- `daemon.post_quit_evidence.reached_quiescent`；
- `daemon.residual_artifacts` 必须**显式存在**且严格等于 `[]`；
- 受控客户端必须**显式包含** `sm_measurement.driver_query`，且
  `requested_affinity_type == 1`、`cu_query_returncode` 为整数或 `None`。

环境能力白名单（统一用于聚合原因与逐项 failure）：

```text
mps_binary_unavailable
effective_sm_count_not_queryable_from_client_context
effective_sm_count_not_verified_for_at_least_two_percentages
cuCtxGetExecAffinity        # 驱动返回码证据本身
```

也就是说，合法的 `cuCtxGetExecAffinity` 不可用不会被误判成实现缺陷；
而 `client_status`、客户端 PID 未观察到、JSON 解析错误、affinity 查询路径异常、
清理未闭合、缺失 `residual_artifacts` / `driver_query` 都会让集成测试失败。

硬门判定逻辑位于 `tests/elastic_residency/sm_wsl_gates.py`，并配有纯单元测试
`tests/elastic_residency/test_sm_wsl_gates.py`，覆盖未知 failure、
缺失 `driver_query`、缺失 `residual_artifacts`、清理未闭合等场景均会触发 `AssertionError`。
若临时目录权限受限，给 pytest 指定仓库内可写的 `--basetemp`，不要把权限错误误报成测试失败。

### 13.1 SM-A 的 provenance 与运行流程

**`git archive HEAD` 不能表示 dirty 工作区**：它只打包已提交的 HEAD，不包含工作树里
modified 与 untracked 的内容。用它当作 dirty 源码快照会得到错误的 provenance。

状态项数量必须实时读取，不能硬编码：

```powershell
git -C D:\AI\xiaoyou-core\legacy\mvp_core status --porcelain
(git -C D:\AI\xiaoyou-core\legacy\mvp_core status --porcelain | Measure-Object -Line).Lines
```

论文正式实验推荐流程：

1. 完成修复与测试；
2. Windows 侧形成一个干净 commit（本目录是独立仓库，提交/推送只在 Windows 侧进行）；
3. 确认 `git status --porcelain` 输出为空（0 项）；
4. 用 `scripts/sync_to_wsl2_safe.ps1 -SkipGit` 同步到 WSL2；
5. 使用**真实完整的 HEAD** 与 `--source-git-dirty false --source-git-status-entry-count 0` 运行。

如果坚持 dirty 探索运行，快照必须包含**实际工作树**的 modified 与 untracked 文件，例如：

```powershell
$repo = 'D:\AI\xiaoyou-core\legacy\mvp_core'
git -C $repo diff HEAD --binary > D:\AI\edgesched_wip.patch
git -C $repo ls-files --others --exclude-standard > D:\AI\edgesched_untracked.txt
# 按 untracked 清单打包，再与 patch 一起复制进运行目录
```

该结果只能用于探索与排障，**不得直接晋升为论文正式证据**；正式证据必须在干净 commit 上重跑。

## 环境变量

高频变量包括：

- `XIAOYOU_LLM_MODEL_PATH`：论文实验 GGUF 路径。
- `XIAOYOU_BENCH_LLM_PATH`：benchmark 适配器 GGUF 路径。
- `XIAOYOU_BENCH_N_CTX`、`XIAOYOU_BENCH_N_GPU_LAYERS`：LLM 运行参数。
- `XIAOYOU_FORGE_BASE_URL`：Forge API，通常是本机/WSL2 的 7860 端口。
- `XIAOYOU_USE_CPP_SCHEDULER`、`XIAOYOU_USE_CPP_FOR_LLM`、`XIAOYOU_USE_CPP_FOR_IMAGE`：C++ 路径开关。
- `XIAOYOU_ALLOW_CPP_LLM_WORKER`：显式允许实验性 C++ LLM worker。
- `XIAOYOU_ALLOW_MOCK_FORGE`：只用于受控调试；启用后结果不得晋升为真实 Forge 证据。

完整环境和路径见 [项目上下文](../docs/PROJECT_CONTEXT.md#7-环境与同步边界)。

## 结果与论文数据

```text
experiment_results/                      当前工作区运行产物
experiment_results/elastic_residency/    当前弹性驻留批次；历史批次位于 _archive/
paper/experiment_data/03_raw_iterations/ 原始迭代
paper/experiment_data/02_auxiliary/      辅助数据
paper/experiment_data/01_paper_main/     新版正文数据入口；当前尚未晋升数据
paper/rewrite/                            新版正文工作区
paper/archive/pre_rewrite_20260830/      旧论文、图表、工具和数据快照
```

弹性驻留结果的当前/已替代/停止/debug 分类见 [运行结果目录](../docs/ELASTIC_RESIDENCY_RESULT_CATALOG_20260830.md)。`_archive` 只改变历史工作产物的位置，不删除原始时间线；分析和论文数据晋升不得自动扫描归档批次。

正文数据晋升必须满足：真实后端未回退、参数冻结、多次迭代完整、manifest 存在、聚合口径一致、图表重建成功、正文引用可追溯。不要从 `paper/archive/` 复制旧数字覆盖新版正文。

## 快速验证

```powershell
..\..\venv_core\Scripts\python.exe -m pytest \
  tests\test_chat_service.py tests\test_phase_aware_quick.py -q

..\..\venv_core\Scripts\python.exe tests\verify_paper_experiments.py
```

真实实验请在 `/home/leslie/mvp_core` 中激活 `/home/leslie/.venvs/mvp_core-cu128` 后运行，并保存完整日志和 manifest。
