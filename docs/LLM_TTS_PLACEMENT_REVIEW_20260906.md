# LLM / TTS 反向放置与 GPU 共存验收复核（2026-09-06）

本轮只读取既有结果，不运行模型。用户确认当前 GPU 共存阶段以兼容性为目标，
SM 管理可用后再研究计算份额与流水线平衡；同时保留 LLM_GPU + TTS_CPU 和 LLM_CPU + TTS_GPU 两种静态基线。

## 已定位的 TTS 真实证据

下表路径相对 `experiment_results/tts/`，数字读取 manifest/chunks.jsonl/iterations.jsonl。

| 证据 | 配置与结果 | 边界 |
|---|---|---|
| `smoke/cpu8_xvector_base_worker_20260828/` | CPU 8 线程，同一句话合成 25.0142 s、音频 3.52 s、RTF 7.1063，1 条完成 | manifest 的 cpu_affinity=[]；不能登记为 8 个物理核隔离曲线，也不是多核数矩阵 |
| `smoke/gpu_xvector_base_worker_20260828/` | GPU，同一句话合成 5.3589 s、音频 2.96 s、RTF 1.8104，1 条完成 | 与 CPU 输出音频时长不同；两条单次 smoke 不是严格性能统计 |
| `smoke/gpu10_xvector_prompt_cache_20260828/` | GPU 连续 10 块，mean RTF 1.7075；模型与 voice prompt 各构建一次 | 句子级生成/复用验证，不是真正音频内部流式 |
| `profiling/gpu_sentence_length/gpu_sentence_length_xvector_r3_20260828/` | 45/45 成功，短/中/长各 15 条，RTF<1 为 0/45 | 16C/32T 可用 affinity，不能据此声称 CPU 吃满或做过核数扩展 |

CPU/GPU smoke 均为 Qwen3-TTS-12Hz-0.6B-Base、SDPA、x_vector_only_ablation，模型 tree SHA-256
`c26faabad4b9171c9184f36d769609a5f631e41fcd825ec6ef52028b5356b196`。

| GPU 句长 | 合成时间 mean（s） | 音频时长 mean（s） | RTF mean |
|---|---:|---:|---:|
| 短 | 6.6787 | 4.0053 | 1.6657 |
| 中 | 15.3921 | 8.8373 | 1.7373 |
| 长 | 35.7351 | 19.2160 | 1.8597 |

以上是按输入句长测量、记录生成音频秒数，不是独立控制音频秒数的矩阵。
在当前 Windows 落盘 `experiment_results/tts`、实验归档及论文归档检索中，未定位到 TTS CPU 1/2/4/8/16 物理核扩展曲线；
这不证明其他目录从未跑过。现有明确的多核曲线主要为辅助 BGE/Whisper，不能当作 TTS 曲线。
T0–T3 是 TTS 组件的设备放置，T3 仍有 GPU talker，也不等于完整 TTS_CPU。

## 两种静态放置的现有线索

旧 9B C2：`experiment_results/elastic_residency/c2/c2_llm_residency_curve_20260829_direct_nvml_v2/`。
逐 context、逐配置各 5 轮；K0 设置 n_gpu_layers=0、offload_kqv=false、op_offload=false，LLM 预算 14 物理核/28 线程。
512/2048/4096 context 下，K0 decode mean 分别 4.4572/4.4376/4.3639 tokens/s，
FULL 分别 56.3526/55.6561/55.2438 tokens/s；K0 TTFT mean 8.5765/35.8923/73.4526 s。

**身份限制**：旧 9B GGUF SHA-256 为 `de10bfef1d344e454439e66c19922af7ca3b00636da3fe21adae91ac7bd8e0a3`，
当前为 `423fb3e5fe2c98df616159b7cd1bd04765f67667f5f8712a81e8f63f4e7ee1ef`。
旧曲线只能提供模型特定线索，不能直接冻结当前 9B+视觉的 CPU 性能或放置结论。

- LLM_GPU + TTS_CPU：文本生成快，但现有 CPU TTS smoke 很慢，可能成为语音瓶颈。
- LLM_CPU + TTS_GPU：GPU 语音生成较快，但 CPU LLM 的首句与后续文本供应可能成为瓶颈。
- 尚未定位到这两个方案在当前 9B artifact 上、相同句子切分/CPU 总预算/播放协议下的成对端到端实测；不能靠独立曲线相加宣布优胜。

## 当前实验顺序与结论边界

1. 先复用已有隔离画像做容量与身份核对，GPU 共存只做必要的兼容性验收：真实双后端、有效输出、容量与 swap、状态/资源释放。区分共驻留与同时执行，前者不能代替并发兼容性。
2. 保存兼容性运行的耗时与干扰信息，但不把普通 CUDA 共享速度当作 SM 优化后的最终性能；也不因无 SM 的速度不佳就否定所有 SM 配置。
3. 两个反向静态放置是 S0 必须比较的基线；只补当前 artifact/核心预算/端到端链路缺口，不盲目重跑已有 GPU 句长矩阵。
4. 原生 Linux SM-A 通过后才比较受控 SM 与普通共享/反向放置。SM 可能减少争抢、平衡阶段，但有分区代价，收益待证，不能预设一定更优。
5. 当前独占 GPU TTS 的 RTF 已大于 1；SM 分配本身不使后端具备音频内部流式接口，也不保证持续实时。有限回复的伪流式仍须记录预缓冲、TTFA、实际播放 callback 与 underrun。

兼容性门和性能 Go/No-Go 分开：本阶段通过不代表伪流式完成或弹性调度收益已成立。
