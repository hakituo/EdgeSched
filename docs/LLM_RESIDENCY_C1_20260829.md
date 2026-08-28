# C1 LLM 驻留端点 smoke（2026-08-29）

## 结论

C1 已通过，可以进入 C2 完整驻留曲线。权威记录为：

```text
experiment_results/elastic_residency/c1/
  c1_llm_residency_512_20260829_verified_v1/
```

该记录基于已推送的干净 commit `234d2df686998fbb82f51068f6b55ee886cf0205`，使用 WSL2 中真实 `llama-cpp-python 0.3.35` 和 `Qwen3.5-9B-Q4_K_M.gguf`。GGUF SHA-256 为 `de10bfef1d344e454439e66c19922af7ca3b00636da3fe21adae91ac7bd8e0a3`，实际读取到 `qwen35.block_count=32`。

K0 和 FULL 均通过 19/19 项硬门槛，0 failure、0 fallback；prompt 恰好 512 tokens，固定生成 128 tokens，127 个 token gap 完整，KV state 保存后恢复到恰好 640 tokens，结束后租约全部释放且 16 个物理核全部归还。

## 冻结配置

- LLM 工作预算：14 个物理核，对应逻辑 CPU `0–27`；
- 系统/资源/GPU 采样预算：2 个物理核，对应逻辑 CPU `28–31`；
- `n_ctx=640`，prompt 512 tokens，固定生成 128 tokens；
- `K0`：`n_gpu_layers=0`、`offload_kqv=false`、`op_offload=false`；
- `FULL`：`n_gpu_layers=-1`、`offload_kqv=true`、`op_offload=true`；
- `flash_attn=true`，每个配置在独立子进程中加载、推理、保存/恢复 KV 并卸载；
- 资源采样 50 ms，GPU 探针 250 ms；开始时间线前等待首个 GPU 样本。

## 结果

| 指标 | K0 | FULL |
|---|---:|---:|
| 模型加载 | 2.566 s | 2.585 s |
| TTFT | 10.659 s | 0.326 s |
| Prefill | 48.05 tok/s | 1593.58 tok/s |
| Decode | 2.77 tok/s | 51.03 tok/s |
| Token gap p50 / p95 / max | 359.72 / 425.52 / 484.53 ms | 18.94 / 23.26 / 32.31 ms |
| KV save / restore | 0.616 / 1.498 s | 0.600 / 1.336 s |
| KV state | 73,676,082 B | 73,676,082 B |
| GPU 占用增量估计 | 127 MiB | 5543 MiB |
| 推理后 RSS | 6.253 GiB | 1.139 GiB |
| 推理期 GPU 数据最大年龄 | 560.46 ms | 366.03 ms |
| 推理期采样 jitter p95 / max | 0.811 / 26.959 ms | 0.251 / 0.287 ms |

FULL 相对 K0 的 TTFT 缩短 96.94%，prefill 为 33.16 倍，decode 为 18.45 倍，token-gap p95 降低 94.53%；代价是 GPU 占用增量估计增加 5416 MiB，同时推理后 RSS 减少约 5.11 GiB。

## 验收与边界

硬门槛只作用于真实推理区间：GPU 数据最大年龄不超过 750 ms，资源采样 jitter p95/max 不超过 5/30 ms。K0 的推理期最大 jitter 为 26.959 ms，接近但未超过门槛；C2 必须观察它在 5 次重复中的稳定性。

KV save/restore 会造成比推理更大的短时采样抖动：本轮 K0 KV-save jitter p95/max 为 56.265/72.872 ms，FULL KV-save 为 34.966/36.514 ms，FULL KV-restore 最大值为 77.661 ms。它们被保留为阶段成本和质量告警，没有混入推理期硬门槛，也不能被解释为“系统全程无尖峰”。

WSL NVIDIA 驱动对 compute process 显存返回 `[N/A]`。产物将精确进程显存保留为 `null`，并把相对加载前整卡基线的差值明确标为 estimate，没有把不可用值伪造成 0。

本轮只有 512-token、每端点 1 次独立运行，仅证明 C1 profiler、两端点配置、精确 token 计数、阶段指标、KV 保存恢复和资源门禁可工作。它不提供 `k={0,8,16,24,32,FULL}` 的正式曲线，不计算 `H_TTS/H_D`，也不进入 `paper/experiment_data/01_paper_main/`。C2 仍需完成 3 个上下文 × 6 个档位 × 5 轮、同日 Forge batch=1 复测和临界层数补测。
