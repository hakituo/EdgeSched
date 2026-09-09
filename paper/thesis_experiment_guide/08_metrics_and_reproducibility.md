# 统一指标、统计与复现要求

> 导航：[总索引](README.md) · 兼容入口：[THESIS_EXPERIMENT_GUIDE_DRAFT.md](../THESIS_EXPERIMENT_GUIDE_DRAFT.md)

## 13. 统一指标

### 13.1 LLM

- TTFT；
- prefill tokens/s；
- decode tokens/s；
- 最大和 p95 token gap；
- 迁移窗口内最大请求延迟；
- 上下文一致性。

### 13.2 TTS

当前非增量 TTS 的 chunk 是按冻结分句规则形成的句级合成任务及其完整输出波形，
不是句内增量音频块；以下合成时长、deadline 和 chunk 样本数均采用此单位。
长句切分与尾句 flush 规则以 E2 协议为准，不因使用 chunk 一词而宣称支持增量音频输出。

- 首句文本就绪延迟、首句完整波形就绪延迟；
- 首次实际播放延迟（设备消费首样本，不是写入 WAV 或提交播放队列）；
- queue wait；
- chunk synthesis time；
- 音频时长；
- RTF mean / p95；
- chunk deadline miss、实际 gap / underrun 次数、累计时长与最大时长；
- 完整语音回复时间（请求到达至最后样本实际播放结束）。

当前非增量 TTS 的 `sentence-ready latency` 定义为 `t_first_waveform_ready - t_request`。
若保留 TTFA 字段，必须附带终点语义；不得将波形就绪延迟直接冒充实际播放延迟。
首次实际播放延迟为 `t_first_device_sample - t_request`，仅在设备时间戳及其时钟映射可验证时报告；
若仅有软件 callback 或入队时间，应注明代理边界，真实播放验收仍 pending。
计时从请求到达开始，包含适用的排队和加载；重叠阶段使用真实关键路径，不能把阶段时长机械相加。

设 `t_play_start` 是首样本实际播放时间，`d_j` 是第 j 句最终播放波形的样本数除以采样率。
第 j 句（j ≥ 2）的无断流目标开始时刻固定为：

\[
D_j=t_{play\_start}+\sum_{m<j}d_m.
\]

音频必须及时就绪并送达设备；如设备需要供给提前量 `epsilon_feed`，则检查
`t_ready_j <= D_j - epsilon_feed`，同时记录实际提交和消费结果。提前量由采集/播放路径校准后冻结，
不在正式结果中调整。发生断流后仍保留原 `D_j`，不能移动后续 deadline 抹掉累计延迟。

实际 underrun 指首播后至回复完成前，因供给不足导致的缺音；波形内自然静音、明确的会话间空闲
和首播前等待不计入。连续缺音区间的事件合并规则、采样分辨率、检测盲区和设备 underrun 状态
须记录；不能仅凭波形 ready 时间线推断设备零 underrun。供给 deadline miss 与设备 underrun 分开报告。
预缓冲等待计入实际播放延迟，有限回复预缓冲后零 underrun 不证明后端持续实时；RTF 仍独立报告。

### 13.3 Image

- 准入到资源准备开始的时间；
- 迁移/释放完成时间；
- 第一张图片完成时间；
- 任务执行时间；
- 队列等待；
- batch throughput；
- Diffusion slowdown。

### 13.4 系统

- 峰值 GPU VRAM；
- GPU 物理 SM 总数、请求比例、实际有效 SM 数与分区后端；
- GPU utilization、功耗、温度，以及可获得时的显存带宽压力代理；
- 峰值 RSS 和 WSL 可用内存；
- 运行期 swap start/max/end/growth 与原始采样时间线；
- CPU 利用率和亲和配置；
- 迁移次数、方向和时间；
- 驻留档位变化；
- OOM、失败和回退次数；
- 调度决策开销；
- SM profile 创建/销毁或 worker 重启成本；
- 预测值与真实值误差。

“Restore Gap=0”只能在采样分辨率和指标定义明确时使用。不得将异步预加载时间等同于用户完全无感知。

---

## 14. 统计与复现要求

### 14.1 运行次数

- 每个配置独立预热。
- 每个正式配置至少 5 次完整独立运行。
- 用于 p95 的事件或 chunk 至少 30 个，优先 50 个以上。
- 请求级与 chunk 级样本量分别计算；五次请求不支持用其 chunk 数冒充请求级 p95 样本。保留运行级分组，不能将同轮 chunk 视为独立实验重复。
- 少量样本只报告原始值、均值和范围，不滥用 p99。

### 14.2 顺序控制

- 使用随机顺序或正反交替顺序降低温度漂移偏差。
- 冷启动与稳态实验分开。
- 不同 batch shape 分别预热。
- 在运行前后记录温度、功耗或至少保留足够冷却时间。

### 14.3 固定输入

正式实验固定并记录：

- prompts；
- 输出 token 上限；
- TTS 文本和分句结果；
- 声音配置；
- 图像 seed、分辨率、steps、sampler、CFG；
- context/KV 长度；
- 核心掩码与线程变量。

### 14.4 证据链

每个正文数字必须能够定位到：

```text
paper figure/table
  -> aggregated JSON
  -> raw iterations
  -> experiment script
  -> manifest
  -> code/model/environment hash
```

manifest 至少记录：

- 是否真实 LLM/TTS/Forge；
- 模型文件和哈希；
- CUDA、驱动和主要库版本；
- WSL2 内存与 swap；
- CPU 核心和亲和配置；
- SM 分区后端、请求份额、实际有效 SM 数、MPS server/client 状态与 GPU UUID；
- 运行参数与随机种子；
- 代码提交哈希；
- 是否发生 fallback。

TTS 的阶段性显存点若采用同步 direct NVML 查询，不套用后台缓存的 sample-age 定义；必须记录
查询开始/结束 monotonic timestamp、查询耗时，并把查询耗时作为样本新鲜度保守上界。
原始 child artifact 同时保留全子进程 swap 时间线，iteration/summary 必须携带相同门禁结果。

Forge 的 b=1 容量校准使用周期 direct NVML 时间线，因为 `M_D` 依赖完整生成窗口的整卡高水位。
因此必须对实际参与差值计算的 idle baseline、resident baseline 与每个 measurement 窗口逐窗保存
最大样本年龄；任一窗口无有效年龄样本或超过 750 ms，即使后端名和样本总数正确也必须拒绝整批结果，
并在 `failures.jsonl` 保留失败门禁名称。

---
