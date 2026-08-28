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

2026-08-28 的完整结果与决策见 [GPU 句长画像](TTS_GPU_SENTENCE_LENGTH_PROFILE_20260828.md)：45 条样本全部 RTF > 1，因此当前应先研究真正音频流式或后端优化，不直接铺开 CPU 核心组合。

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
