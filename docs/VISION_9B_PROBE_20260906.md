# 9B + 视觉端点复核与主模型口径（2026-09-06）

用户执行 `scripts/experiments/run_vision_9b_smoke.ps1` 后，已只读核验
`experiment_results/elastic_residency/vision_endpoint/v1_vision_9b_probe_20260906_201923/`。
本记录不启动实验、不改写原始产物、不晋升论文数据。

## 决策与证据边界

**当前状态（用户完成 Answer / Profile 后）**：9B+视觉基础端点和五轮资源画像已完成。
完整回答已覆盖三种主要图形及白底，遗漏黑色边框；仅主要内容识别通过，不是全部细节质量验收。
下一步先离线筛容量与阶段切换候选，再决定是否有必要编写或运行共置补测；不得重复要求运行本组基础画像。

后续主实验目标使用 **Qwen3.5-9B-q4_k_m.gguf + Qwen3.5-9B-f16.mmproj**，统一执行文本与图片理解。
4B 保留为跨尺度对照及容量边界证据。该决定取代 2026-08-31 的 4B 主模型选型。
现有通用实验 CLI 的 4B 默认值仍是历史行为，后续 9B 命令必须显式传入模型及 projector；
当前可用的 9B 专用手动入口为 `scripts/experiments/run_vision_9b_smoke.ps1`。

2026-09-06 端点复核后，按用户要求将两份模型复制到 WSL 原生目录
`/home/leslie/mvp_core/models/llm/`，文件名不变，源与目标 SHA-256 均与下文一致。
D: 源文件与现有 4B 文件保留。手动入口已改为读取这两个 WSL 路径，并在启动前检查文件存在；
`-PrintOnly` 路径检查通过，未启动新的推理或实验。下文旧端点的 D: 存储身份保持不变。

本次仅证明 FULL 档的文本对照与实际视觉执行端点可用。两种状态各 1 轮，不能作为 p95 性能结论、
多轮稳定性、长上下文、完整图片描述质量、视觉与 TTS 共置或弹性调度 Go 的证明。
历史 9B TEXT_ONLY 曲线及 H_TTS 保持原模型、原哈希和原环境口径，复用前须核对身份；
不能直接作为当前 9B VISION_WARM 的阈值，也不能从 4B projector 画像外推。
16G 下已停止的 LLM+Forge 试验继续保留负面边界性质。

## 实际运行身份

- WSL2 Linux，Python 3.10.19；推理环境 `mvp_core-cu128`。
- 模型从 `/mnt/d/AI/xiaoyou-core/models/llm/` 读取，本次未迁入 WSL 原生模型目录；加载时间包含该存储路径影响。
- 主 GGUF：5974828256 B，SHA-256 `423fb3e5fe2c98df616159b7cd1bd04765f67667f5f8712a81e8f63f4e7ee1ef`。
- projector：918165888 B，SHA-256 `e36da9b9891e13a172b25028ad18b4fc63bc4d681cf391a44b2a9a3b51bc797a`。
- 主模型 32 层、embedding 4096；projector projection 4096，27 个视觉块；双方缺 basename 时门禁要求非空名称、base-model 名称及来源 URL 全部一致。
- FULL，`TEXT_ONLY_CHAT / VISION_WARM_GPU`，各 1 轮；`context_tokens=512`、`generation_tokens=128`、实际 `n_ctx=640`。纯文本对照实际 prompt 为 20 tokens，视觉轮为 23 个文本 tokens 加 196 个视觉 tokens，不能称为填满 512-token 的负载。
- 14 个物理核给 LLM，2 个物理核保留系统，seed 42；`split_encode`，固定 448×448 图片。
- 源提交 `39b8f0e44867dbe42e28928b925d13f95085d81d`，dirty=true，21 个状态条目；源码快照 SHA-256 `ab6ac26d06cd572fd9c972c4bf3212041157b00b9d4ef85a2b1ceb8bb9e4e2a5`。

## 只读核验结果

- summary 通过，2/2，failure 0；iterations 与 summary.records 一致；child 除父进程追加的执行序号外与记录一致。
- 源码快照与输出文本哈希复算一致；全部 acceptance checks 通过，direct NVML、采样新鲜度、affinity、KV 往返、租约归还均通过；两组 swap 全程为 0。
- 真实图片编码返回非空嵌入，日志形状 `[4096, 196, 1]`；196 个视觉 tokens 推进 14 个 M-RoPE 位置，图片 chunk 返回码 0。
- 输出正确识别白底、左上蓝色正方形及中央红圆；固定生成 128 tokens 后截断，尚未描述右下绿方块及黑边框，不能登记为完整语义验收。

| 单轮指标 | TEXT_ONLY_CHAT | VISION_WARM_GPU |
|---|---:|---:|
| TTFT（s，按当前入口计时） | 0.3969 | 0.9156 |
| decode（tokens/s） | 52.8441 | 54.3286 |
| 采样整卡显存峰值（MiB） | 6394.4727 | 7542.4727 |
| swap 峰值（MiB） | 0 | 0 |

视觉轮 projector 加载 8.2081 s，编码 0.2910 s，视觉嵌入 decode 0.2222 s，释放 0.01085 s。
TTFT 不含前面的主模型与 projector 加载。projector warm 整卡增量 1068 MiB；整卡峰值相对 8151 MiB 总量剩余 608.53 MiB，包含桌面等基线，非精确进程显存，也不是冻结后的安全余量。
`memory_after_inference` 在 projector 释放后采集，不能拿其 6396.47 MiB 当视觉共驻峰值。
9B 日志中的 warmup CUDA compute buffer 为 248.10 MiB；原始 `vision_measurement_semantics` 内仍有历史 4B 的示例数字（644.26/223.30 MiB），这些说明字符串不作为本次测量值。

下节保留 Answer / Profile 的运行协议，现已由用户执行完成；最新结果与下一步决策见末节。

## WSL 原生路径复跑与下一轮协议

用户复跑 `v1_vision_9b_probe_20260906_204035` 已通过 2/2，模型路径确为
`/home/leslie/mvp_core/models/llm/`，模型哈希与前次一致；源码快照、输出哈希及 parent/child 记录核对通过。
视觉轮主模型加载 2.4020 s、projector 加载 0.5561 s、TTFT 0.5161 s、decode 55.6232 tokens/s，
采样整卡峰值 7542.4727 MiB、swap 0；仍为 196 视觉 tokens，输出仍在 128 tokens 截断。
不同存储路径两次试跑没有控制页缓存状态，不把时间差当作存储因果或正式迁移收益。

手动入口现支持 `-Stage Smoke / Answer / Profile`，默认 Smoke 保留原来的两单元协议：

| Stage | 状态 | 轮数 | 固定生成预算 | 用途 |
|---|---|---:|---:|---|
| Smoke | TEXT_ONLY_CHAT + VISION_WARM_GPU | 各 1 | 128 | 原端点协议 |
| Answer | VISION_WARM_GPU | 1 | 1024 | 为完整图片回答提供更长预算，需复核实际内容 |
| Profile | VISION_WARM_GPU | 5 | 128 | 保持原生成长度的多轮资源/编码画像 |

默认自动创建带阶段名的新 run id，不覆盖旧结果；两个模型固定 SHA-256 门禁。
各阶段均使用 context_tokens=512，因此 Answer 的 n_ctx=1536，Profile 的 n_ctx=640；
生成预算与 KV 容量不同，禁止把 Answer 混入 Profile 性能统计。
沿用现有贪心固定长度生成，遇 EOS 仍继续；完整回答以第一个终止标记前的有效内容为准，
核对白底、黑边、左上蓝方块、中央红圆、右下绿方块及有无矛盾。1024 token 只是预算，不能保证回答完成。
Profile 每轮独立子进程、重新加载并 warm projector，之后编码固定图片；5 轮 seed 为 42–46，
不能称为同一常驻会话的连续请求，也不是预热过的端到端聊天或 TTS 共置验收。
执行顺序为先 Answer 再 Profile；两项现已完成，不需重复。失败则保留结果停止，不放宽门禁。

静态验证：`tests/elastic_residency/test_vision_9b_launcher.ps1` 覆盖三阶段参数、跨目录启动、
原生路径、哈希/快照参数、默认兼容和非法参数拒绝，通过；测试禁止 WSL 调用。
本轮只改手动编排脚本与记录，未启动模型、同步代码或运行真实实验。

## Answer 与五轮 Profile 完成记录

产物根目录仍为 `experiment_results/elastic_residency/vision_endpoint/`，均读取 WSL 原生 9B 模型和 F16 projector，哈希与前述身份一致。

| 批次 | 结果 | 可用结论与限制 |
|---|---|---|
| `v1_vision_9b_answer_20260906_204651` | 1/1 技术通过，0 failure，swap 0，GPU 采样峰值 7568.4727 MiB | 1024 tokens / n_ctx=1536；主要图形、颜色、位置及白底正确，但漏黑边；EOS 计数 9，后续重复与固定预算忽略 EOS 一致，不能当作自然回复长度或正常结束语义 |
| `v1_vision_9b_profile_20260906_204939` | 5/5 全通过，0 failure，swap 全程 0 | FULL / VISION_WARM_GPU，128 tokens / n_ctx=640；每轮均 196 视觉 tokens、非空嵌入，基础多轮资源画像完成，不是长上下文、TTS 共置或调度收益证据 |

两批 iterations 与 summary.records 一致，child 仅少父进程追加的执行序号；全部 acceptance checks 为真，
输出文本和源码快照 SHA-256 已复算一致。源码提交均为 `39b8f0e44867dbe42e28928b925d13f95085d81d`、dirty=true，实际代码身份由各自快照保存：

- Answer：`5926229b6d722a377800427775216b5d077e20d7d279292419c9d193a0de597f`。
- Profile：`5926229b6d722a377800427775216b5d077e20d7d279292419c9d193a0de597f`。

五轮统计只使用 Profile，不混入 Answer 或 D: 读取的试跑：

| 指标 | mean | min–max |
|---|---:|---:|
| 主模型加载（s） | 2.4667 | 2.2866–3.0780 |
| projector 加载（s） | 0.4287 | 0.3695–0.5695 |
| projector 编码（s） | 0.1447 | 0.1242–0.1720 |
| TTFT，不含模型加载（s） | 0.5128 | 0.4721–0.5876 |
| decode（tokens/s） | 55.7562 | 54.9718–56.1664 |
| projector warm 整卡增量（MiB） | 1126.0 | 1126.0–1126.0 |
| 采样整卡峰值（MiB） | 7542.4727 | 7542.4727–7542.4727 |

## KV 与容量判断：先筛选，再决定是否补测

本次峰值已包含 `n_ctx=640` 的预分配 GPU KV（24 MiB），不能重复加一次。
compute_gpu_buffer 为 501.77 MiB，是另一个已计入实测峰值的缓冲概念；序列化状态字节数不等于 GPU KV。
实际短 prompt 不代表长对话，不能把本次测量外推到 2048/4096 或更长上下文。
8151 MiB 总量减峰值只剩 608.53 MiB，这是未扣安全余量的物理余量，不能直接作为 TTS 可用预算。
不宣布 FULL+视觉+GPU TTS 可行，也不据单次独立组件加和冻结新的 H_TTS。

下一步先用已有数据核对模型哈希、上下文、idle 与安全余量口径，排除明显容量不成立的候选。
只在筛选后仍存在合理候选时，准备少量用户手动补测，比较以下阶段方案与完整迁移基线：

1. 图像编码/嵌入进入语言上下文后，释放 projector、保留语言状态继续生成，再安排 TTS。
2. 必要时减少 LLM 的 GPU 驻留层数，联合评估主机内存、swap、转换成本及端到端延迟。
3. Diffusion 生图沿用 16G 下的迁移/逐出设计；现有负面边界不因视觉端点通过而解除。

第 1 项是待验证方案：当前入口在生成结束后才释放 projector，尚未证明“prefill 后立即释放再继续 decode”的完整执行与恢复契约。
若这些方案不能比整模型迁移或阶段串行更划算，就收缩弹性调度主张；不能为了制造共置收益继续盲跑矩阵。
尚未实现/测量新的阶段切换或视觉 TTS 共置，未修改原始 JSON、未晋升正文数据、未自动运行新实验。

## 重启前后对照（2026-09-07）

用户反馈重启前桌面卡顿、重启后恢复，并手动运行 `v1_vision_9b_profile_20260907_012723`。
与 `v1_vision_9b_profile_20260906_204939` 比较，两批均为 5/5 全通过，0 failure、swap 全程 0，
每轮 196 视觉 tokens、24 MiB GPU KV；所有输出文本哈希一致（此事实不作为视觉输出确定性的通用门禁）。
两批 run_config 除 run_id 外完全一致，主模型/projector/图片哈希、采样间隔、核数和生成预算均相同。
两份源码压缩快照各 136 个普通文件，逐文件哈希比对仅 `experiments/README.md` 不同，实验代码一致。
新批次快照 SHA-256 为 `b3ba6f5f9c88b3461b1aa85b8d0108528dce61d537ab76bbc9edf53062680c91`，复算通过；
iterations/summary/child 和输出哈希同样核验通过。重启时间与桌面卡顿来自用户陈述，实验未采集 Windows 桌面延迟。

| 指标（时间与速度为五轮 mean） | 重启前 | 重启后 | 变化 |
|---|---:|---:|---:|
| 主模型加载（s） | 2.4667 | 2.5027 | +1.46% |
| projector 加载（s） | 0.4287 | 0.4215 | −1.68% |
| projector 编码（s） | 0.1447 | 0.1313 | −9.27%（约 13.4 ms） |
| TTFT，不含加载（s） | 0.5128 | 0.4955 | −3.37% |
| decode（tokens/s） | 55.7562 | 55.3543 | −0.72% |
| 每轮 token-gap p95 的均值（ms） | 19.3563 | 19.8990 | +2.80% |
| 整卡加载前基线（MiB，各轮相同） | 767.3750 | 681.0703 | −86.3047 |
| 整卡采样峰值（MiB，各轮相同） | 7542.4727 | 7456.1680 | −86.3047 |
| 峰值减基线（MiB） | 6775.0977 | 6775.0977 | 0 |
| projector warm 整卡增量（MiB） | 1126.0 | 1126.0 | 0 |

两组 TTFT 范围分别 0.4721–0.5876 s / 0.4325–0.6148 s，decode 范围分别
54.9718–56.1664 / 54.9272–55.7741 tokens/s，存在明显重叠。采样 GPU 温度最高值为 75°C / 76°C，
WSL 系统可用内存最低值均约 13.06 GiB；不能把这些读数当作 Windows 整机内存或桌面流畅度测量。

当前没有证据表明旧五轮画像受到严重性能污染，不因用户感知桌面卡顿而作废旧批次。
峰值下降完全对应基线下降，不写成模型显存优化或重启释放了模型内存；缺少进程归因，不能定位减少的是哪个后台应用。
时间变化幅度与方向并不一致，不能据每组五轮、单次前后对照宣称重启带来因果性能提升或统计等效。
两批分别保留为重启前后环境对照，不混作单一冻结曲线，也不据此要求重跑全部 TTS/LLM 历史实验。
