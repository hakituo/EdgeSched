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
模型：`/home/leslie/xiaoyou-core/models/Qwen3-TTS-12Hz-0.6B-Base`（2026-09-25 起脚本默认值；与 mvp_core 下旧副本经 `diff -rq` 核验完全一致，仍可用 `XIAOYOU_BENCH_TTS_PATH` 覆盖）
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

实测每步约 125–156 ms 的解码开销、且 `process_cpu_percent` 恒 ~100%；上游项目文档称每步约 500 次 CUDA kernel 启动，本仓库的 CPU 占用数据**与该解释一致，但不足以单独完成原因归因**（未做内核级 profile）。换用 MIT 项目 `faster-qwen3-tts`（静态 KV cache + `torch.cuda.CUDAGraph` 捕获）后，**同一模型、同一参考音频、同一 B2 文本**（x-vector、非流式、`do_sample=False`、eager 注意力）：

| 文本 | 上游动态 RTF（min–max） | CUDA Graph RTF（min–max） | 加速比 |
|---|---:|---:|---:|
| 20 字（4.72 s 音频） | 1.7754（1.6215–1.8833） | 0.3168（0.3165–0.3174） | 5.6× |
| 50 字（10.64 s 音频） | 1.7257（1.6534–1.7840） | 0.3067（0.3058–0.3073） | 5.6× |
| 100 字（18.88 s 音频） | 1.7622（1.6612–1.8345） | 0.3032（0.3030–0.3034） | 5.8× |

上表来自 `tts_backend_both_eager_20260917_072516`：**同一隔离环境、同一进程先后测量**，因此可直接比较。
早期跨环境批次（`cudagraph_only_20260917_v2`，报 9.5×/7.2×/8.1×）**已作废，产物已删除**——同环境下上游 eager 为
1.72–1.78，与 B2 的 sdpa 基线（1.60–1.92）基本一致，说明注意力实现影响很小，差异来自环境与包版本
（原环境 python3.10 + 上游 `qwen-tts 0.1.1` + transformers 4.57.3）。

入口：

```bash
# 两条路径同进程先后测量（推荐；用脚本入口自动同步与落 provenance）
./scripts/experiments/run_tts_backend_compare.ps1 -Mode both        # Windows 侧发起
# 或直接跑
/home/leslie/.venvs/mvp_core-cu128/bin/python -u \
  -m experiments.tts.profile_qwen3_tts_cudagraph --paths upstream_dynamic cuda_graph_static
```

产物：`experiment_results/tts/cudagraph/<run_id>/{manifest.json,chunks.jsonl,audio/}`，含模型树哈希、参考音频哈希、git provenance 与逐条 PCM SHA。

**边界（必须随数字一起报告）**

- 加速路径依赖 `qwen-tts-hf 0.1.1.post1` + `transformers>=5.15.1` + cu128 torchaudio，**不能装进旧的 `mvp_core-cu128`**（上游 `qwen-tts 0.1.1` 会在图捕获时报 `StaticLayer.lazy_initialization()` 签名不匹配；默认 torchaudio wheel 是 CUDA 13 构建；transformers 5.17 会让 `MimiConfig` 缺 `rope_theta`），因此使用仓库外隔离环境 `/home/leslie/qwen3tts-cugraph-probe/faster-qwen3-tts/.venv`。
- **2026-09-17 起主环境路径已扶正**：`/home/leslie/.venvs/mvp_core-cu128` 改为指向该隔离环境的软链；同日**遗留环境 `mvp_core-cu128_legacy` 已删除**（释放约 9G），它是唯一装了上游 `qwen_tts` 的环境。因此脚本里写 `/home/leslie/.venvs/mvp_core-cu128/bin/python` 就是在加速环境里跑（Python 3.12 / torch 2.11 / transformers 5.15.1）；换环境只需改这条软链。需要上游 `qwen_tts` 的旧后端脚本（`upstream_dynamic`、旧 B2 矩阵、worker 画像、应用 `services/tts/qwen3_tts_worker.py`）重跑需自建环境并用 `-Python` 指定。注意 `cpp_scheduler` 的 WSL2 扩展是按旧 py3.10 构建的，若要用需在新主环境重编。
- **2026-09-25 环境再次变更（当前有效）**：系统迁移后加速隔离环境 `/home/leslie/qwen3tts-cugraph-probe/` 已变为空 venv（transformers 5.x 历史栈）；主环境软链改指 `/home/leslie/xiaoyou-core/.venv`（Python 3.13 / torch 2.12+cu130 / transformers 4.57.3），其中 `faster-qwen3-tts 0.2.6` 的依赖声明是 `transformers>=4.57`、`qwen-tts>=0.1.1`，**已在 4.57.3 下实测通过（不需要 transformers 5.x，也不要升级——见下条"为什么不能升 5.x"）**：predictor/talker 两个 CUDA Graph 均捕获成功，经 mvp 集成入口 `build_synth("cuda_graph_static")` 合成有效音频（峰值 0.436）；单卡 RTX 5070 / bf16 / x-vector / eager 口径下**稳态 RTF 0.3136（合成 1.73 s / 音频 5.52 s）与历史隔离环境的 0.303–0.317 一致**；首个请求有 6 s 级一次性图相关开销（历史口径单列 graph_capture_s）。S0 v2 / `cuda_graph_static` 可在现环境直接重跑，历史数值现在有了现环境复现证据；`qwen3tts-cugraph-probe` 目录保留作历史回退路径，代码里 `DEFAULT_FASTER_PATH` 仅在环境没装包时回退到它。
- **为什么不能在共享 venv 升级 transformers 5.x**：`qwen-tts 0.1.1` 精确钉死 `transformers==4.57.3`，且该 venv 是 xiaoyou-core 主项目的运行环境；升 5.x 必须同时把 qwen-tts 换成 `qwen-tts-hf`（faster-qwen3-tts 0.4.0 起的默认依赖），会破坏上游路径和主项目其他模块。0.2.6 是唯一同时兼容「CUDA Graph 加速 + 4.57.3 + 上游 qwen-tts 同环境共存」的版本线（0.3.x/0.4.0 已转向 qwen-tts-hf/transformers 5），所以主环境**刻意保持 faster-qwen3-tts==0.2.6，不要顺手升级**。
- 静态 KV cache 与上游走不同 kernel 路径，**不是位精确**。同环境实测：上游动态路径 3/3 轮 PCM SHA 完全一致、样本数恒为 113280 / 255360 / 453120；加速后端 3/3 轮 SHA 全不同，样本数 111360/122880/128640、249600/261120、443520/472320/491520，相对上游同一文本**最大偏差 +13.6%（20 字）、最小 −2.3%**。**这只证明「不位精确」，既不等于质量不合格，也不等于统计结果不可复现**；长度差异需另行核查（重复同句、交替长短文本、检查状态重置是否完整），并同时报音频时长、合成耗时与文本/听感一致性，不能只看 SHA。
- **2026-09-17 已接入 `s0_pipeline`**（此前为「尚未接入」）：新增 `--tts-backend {cuda_graph_static,upstream_dynamic}`，默认 `cuda_graph_static`（当前接入后端）；加速路径复用 `AcceleratedSentenceSynthesizer`（只覆盖 `_generate`），协议升到 `s0_protocol_v2`，正式配置由 4 档降为 2 档（`K24:T0`、`K0:T0`），`K24:T3`/`K24:T4` 因后端无分部件放置接口而排除。隔离环境已补装 CUDA 编译的 `llama-cpp-python==0.3.35`（`llama_supports_gpu_offload()=True`）与 `sounddevice==0.5.5`，S0 为保持 LLM 与 TTS 同进程共置默认跑在该环境。音频一致性口径见 `experiments/tts/check_tts_output_correctness.py`（主环境 ASR 回转核对）。

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
