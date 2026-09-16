# Qwen3-TTS Base 独立子系统

本模块实现 **句子级流式** TTS：文本 token 被增量断句后，完整句子依次进入单个 Qwen3-TTS Base Worker。`qwen-tts==0.1.1` 的 `non_streaming_mode=False` 不会返回真正的增量音频包，因此当前实现不宣称模型内部音频流式，也不引用论文中的 97ms 作为本系统 TTFA。完成句子从入队到 WAV 结果可用的指标统一命名为 `first_chunk_ready_s`，不是第一音频采样延迟。

## 当前边界

本里程碑包含：

- `VoiceCloneProfile` 音色画像及参考音频哈希校验；
- `spawn` 子进程中的真实 Base 模型加载和 prompt 缓存；
- `deque + asyncio.Condition` 有界队列、背压、取消和迟到结果丢弃；
- 跨 token 的 `<think>`、`[GEN_IMG:...]` 过滤和确定性断句；
- 输出重排检查、Worker 状态机、设备重启式切换指标；
- CPU/GPU 独立 smoke 和连续 10 块 prompt 复用验收。

本里程碑尚不连接 LLM、播放器、Diffusion、ResourceManager 或 Phase-Aware。`phase_model.py` 和 `resource_manager.py` 中既有 TTS 静态资源值仍是未校准占位，不应进入论文决策。

## 结构与职责

| 路径 | 职责 |
|---|---|
| `domain/models/tts.py` | 音色画像、chunk 请求/结果、输出模式 |
| `services/tts/sentence_splitter.py` | 增量过滤和确定性断句 |
| `services/tts/queue.py` | 可原子取消的有界队列 |
| `services/tts/qwen3_tts_worker.py` | 主进程代理和 spawn 模型子进程 |
| `services/tts/pipeline.py` | 生产、派发、取消、迟到丢弃和有序输出 |
| `data/adapters/qwen3_tts_adapter.py` | 生产适配器门面 |
| `experiments/tts/` | 真实 Base smoke 与 10 块画像实验 |
| `tests/tts/` | 数据、断句、队列、协议、取消和顺序验证 |

生产 Adapter 只接受 `backend_kind="real_qwen3_tts"`。Mock 位于 `data/adapters/mock_tts_adapter.py` 和实验框架的 `mock_tasks.py`，不得冒充真实结果。

## 音色画像规则

默认使用 ICL：

```text
x_vector_only_mode = False
ref_text = 参考音频的准确逐字转写
```

Worker 每次启动只加载一次模型，并对每个注册画像调用一次 `create_voice_clone_prompt()`。后续所有 chunk 复用缓存 prompt。`model_load_count`、`voice_prompt_build_count` 和 `worker_epoch` 同时进入 health 与每块结果。

`--x-vector-only` 是显式消融，可以在不知道参考音频准确转写时用于链路验证，但不能和默认 ICL 的音色质量结果混写。Base 不支持 CustomVoice 的 `emotion/instruct`，非空 emotion 会抛出 `UnsupportedFeatureError`。

CPU profiling 必须同时设置线程数和物理核亲和性。当前 WSL 拓扑是 16 个物理核、32 个逻辑 CPU，首线程代表为 `0,2,4,...,30`；Worker health 和每条结果都会回报实际 affinity，不能只凭 `cpu_threads` 声称完成分核。

## WSL2 真实运行

环境：`/home/leslie/.venvs/mvp_core-cu128`
模型：`/home/leslie/mvp_core/models/Qwen3-TTS-12Hz-0.6B-Base`
参考音频：`/home/leslie/mvp_core/models/tts_refs/ref_calm.wav`

准确转写已知时，运行默认 ICL GPU smoke：

```bash
/home/leslie/.venvs/mvp_core-cu128/bin/python \
  experiments/tts/smoke_qwen3_tts_base.py \
  --device cuda:0 \
  --ref-text '这里填写参考音频的准确转写'
```

不知道准确转写时，只能明确运行 x-vector 消融：

```bash
/home/leslie/.venvs/mvp_core-cu128/bin/python \
  experiments/tts/smoke_qwen3_tts_base.py \
  --device cuda:0 --x-vector-only
```

CPU smoke 与连续 10 块验收：

```bash
/home/leslie/.venvs/mvp_core-cu128/bin/python \
  experiments/tts/smoke_qwen3_tts_base.py \
  --device cpu --cpu-threads 8 --x-vector-only

/home/leslie/.venvs/mvp_core-cu128/bin/python \
  experiments/tts/profile_qwen3_tts.py \
  --device cuda:0 --x-vector-only
```

GPU 句长 profiling：

```bash
/home/leslie/.venvs/mvp_core-cu128/bin/python \
  experiments/tts/profile_qwen3_tts_gpu_sentence_lengths.py \
  --device cuda:0 --x-vector-only --rounds 3 \
  --git-root /mnt/d/AI/xiaoyou-core/legacy/mvp_core
```

2026-08-28 的完整结果与决策见 [GPU 句长画像](TTS_GPU_SENTENCE_LENGTH_PROFILE_20260828.md)：45 条样本全部 RTF > 1，因此当前应先研究真正音频流式或后端优化，不直接铺开 CPU 核心组合。**2026-09-17 该判断已兑现**：换推理路径（CUDA Graph）后 RTF 降到 0.32，见下节。

## 推理后端对照（CUDA Graph，2026-09-17）

原始 RTF > 1 的主因不是模型能力，而是每步约 125–156 ms 的解码开销（`process_cpu_percent` 恒 ~100%，符合「每步约 500 次 CUDA kernel 启动」的特征）。换用 MIT 项目 `faster-qwen3-tts`（静态 KV cache + `torch.cuda.CUDAGraph` 捕获）后，**同一模型、同一参考音频、同一 B2 文本**（x-vector、非流式、`do_sample=False`、eager 注意力）：

| 文本 | 上游动态 RTF（min–max） | CUDA Graph RTF（min–max） | 加速比 |
|---|---:|---:|---:|
| 20 字（4.72 s 音频） | 1.7754（1.6215–1.8833） | 0.3168（0.3165–0.3174） | 5.6× |
| 50 字（10.64 s 音频） | 1.7257（1.6534–1.7840） | 0.3067（0.3058–0.3073） | 5.6× |
| 100 字（18.88 s 音频） | 1.7622（1.6612–1.8345） | 0.3032（0.3030–0.3034） | 5.8× |

上表来自 `tts_backend_both_eager_20260917_072516`：**同一隔离环境、同一进程先后测量**，因此可直接比较。
早期跨环境批次（`cudagraph_only_20260917_v2`，报 9.5×/7.2×/8.1×）**已作废只作旁证**——同环境下上游 eager 为
1.72–1.78，与 B2 的 sdpa 基线（1.60–1.92）基本一致，说明注意力实现影响很小，差异来自环境与包版本
（原环境 python3.10 + 上游 `qwen-tts 0.1.1` + transformers 4.57.3）。

入口：

```bash
# 两条路径同进程先后测量（推荐；用脚本入口自动同步与落 provenance）
./scripts/experiments/run_tts_backend_compare.ps1 -Mode both        # Windows 侧发起
# 或直接跑
/home/leslie/qwen3tts-cugraph-probe/faster-qwen3-tts/.venv/bin/python -u \
  -m experiments.tts.profile_qwen3_tts_cudagraph --paths upstream_dynamic cuda_graph_static
```

产物：`experiment_results/tts/cudagraph/<run_id>/{manifest.json,chunks.jsonl,audio/}`，含模型树哈希、参考音频哈希、git provenance 与逐条 PCM SHA。

**边界（必须随数字一起报告）**

- 加速路径依赖 `qwen-tts-hf 0.1.1.post1` + `transformers>=5.15.1` + cu128 torchaudio，**不能装进 `mvp_core-cu128`**（上游 `qwen-tts 0.1.1` 会在图捕获时报 `StaticLayer.lazy_initialization()` 签名不匹配；默认 torchaudio wheel 是 CUDA 13 构建；transformers 5.17 会让 `MimiConfig` 缺 `rope_theta`），因此使用仓库外隔离环境 `/home/leslie/qwen3tts-cugraph-probe/`。
- 静态 KV cache 与上游走不同 SDPA kernel，**不是位精确**。同环境实测：上游动态路径 3/3 轮 PCM SHA 完全一致、样本数恒为 113280 / 255360 / 453120；CUDA Graph 路径 3/3 轮 SHA 全不同、样本数 111360–128640 / 249600–261120 / 443520–491520（最大差约 10%）。即加速路径即使 `do_sample=False` 也不确定，**输出长度也会变**。
- 尚未接入 `s0_pipeline`：接入需要替换 `SentenceSynthesizer` 后端并把 `s0_protocol_v1` 升到 v2（含音频一致性口径），属冻结口径变更，须先评估再动手。

每次真实运行写入：

```text
experiment_results/tts/smoke/<run_id>/
  manifest.json
  chunks.jsonl
  audio/
  worker.log
```

`manifest.json` 记录模型文件清单及目录哈希、参考音频哈希、语言、线程/亲和、dtype、attention backend、设备、采样参数和真实后端标识。`chunks.jsonl` 只记录 WAV 路径、音频哈希与指标，不包含音频字节。

## 验证

Windows 静态/Mock：

```powershell
..\..\venv_core\Scripts\python.exe -m pytest tests\tts -q
..\..\venv_core\Scripts\python.exe -m ruff check --no-cache domain\models services\tts data\adapters\qwen3_tts_adapter.py experiments\tts tests\tts
```

Mock 通过只证明进程协议、队列和取消语义。性能结论必须来自 WSL2 的 `backend_kind="real_qwen3_tts"` 产物。
