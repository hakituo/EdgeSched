# Qwen3-TTS Base GPU 句长画像（2026-08-28）

## 结论

在当前 RTX 5070 Laptop GPU、CUDA 12.8 PyTorch、SDPA、Qwen3-TTS Base 0.6B 与 x-vector 音色消融配置下，45/45 条真实中文语音均成功且顺序正确，但 **没有任何一条达到 RTF < 1**。

因此当前落入第三种结论：GPU 也无法满足持续实时生成约束。下一步应优先研究真正的音频内部流式、生成后端或模型优化；暂不直接铺开 CPU 核心扩展组合。CPU 后续仍可作为 Diffusion 占用 GPU 时的降级路径进行特性测量，但不能指望分核解决当前实时率问题。

## 实验口径

- 模型：`Qwen3-TTS-12Hz-0.6B-Base`
- 后端：`real_qwen3_tts`，禁止 fallback
- 设备：NVIDIA GeForce RTX 5070 Laptop GPU
- 驱动：591.74
- attention：SDPA，未安装 flash-attn
- 音色模式：`x_vector_only_ablation`；参考音频准确转写未知，因此不作为默认 ICL 音色质量证据
- 文本：短/中/长各 5 条不同中文文本，实际长度分别为 17–20、43–47、91–103 字
- 运行：模型和 prompt 只加载一次，1 条预热不计入统计，之后 3 轮共 45 条
- 顺序：三档在各轮中轮换，减小温度和时间漂移
- `first_chunk_ready_s`：请求入队到完整句子 WAV 结果在主进程可用的时间；不是第一音频采样延迟
- 显存：`torch.cuda.max_memory_allocated()` 的单 chunk 峰值，不等于整卡总占用

运行时 Git 基线为 `f0c0a39abbaef642fc0c1d19a434141c365574fa`，manifest 同时记录 dirty 状态与状态摘要哈希。WSL 内存、swap、GPU 型号和驱动均保存在原始 manifest。

## 汇总结果

| 句长 | 样本 | 平均字符 | 合成时间 mean / p95 | 音频时长 mean | RTF mean / p95 | First chunk ready mean / p95 | 峰值显存 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 短 | 15 | 18.4 | 6.68 / 8.13 s | 4.01 s | 1.666 / 1.861 | 6.69 / 8.14 s | 2252.9 MiB |
| 中 | 15 | 45.2 | 15.39 / 20.46 s | 8.84 s | 1.737 / 2.300 | 15.41 / 20.49 s | 2405.8 MiB |
| 长 | 15 | 97.2 | 35.74 / 40.24 s | 19.22 s | 1.860 / 2.129 | 35.77 / 40.27 s | 2761.8 MiB |

完整性检查：

- 成功：45/45
- RTF < 1：0/45
- 输出乱序：0
- fallback：0
- Worker epoch：1
- 模型加载次数：1
- prompt 构建次数：1

## GPU 画像函数

在本次 17–103 字观测范围内，对每条原始记录做简单线性拟合：

```text
T_hat_GPU(L) = max(0, -0.624 + 0.3711 * L) 秒
R² = 0.953

M_hat_GPU(L) = 2122.86 + 5.7031 * L MiB
R² = 0.990
```

这里 `L` 是包含标点的中文文本字符数。函数只适用于当前模型、SDPA、x-vector 配置和观测区间，不能外推到 ICL、其他后端或超长文本。

RTF 的长度拟合为：

```text
RTF_hat_GPU(L) = 1.624 + 0.002422 * L
R² = 0.137
```

RTF 拟合度很低，说明它不仅取决于输入字符数，还受到生成音频长度、文本内容和运行抖动影响。调度器不应只用该直线预测 RTF；现阶段应优先使用三档经验分布及 p95 约束。

## 证据位置

```text
experiment_results/tts/profiling/gpu_sentence_length/
  gpu_sentence_length_xvector_r3_20260828/
    run_config.json
    manifest.json
    iterations.jsonl
    summary.csv
    profile_models.json
    gpu_sentence_length_profile.png
    audio/*.wav
    worker.log
```

这些是工作区真实实验产物，尚未晋升到 `paper/experiment_data/01_paper_main/`。
