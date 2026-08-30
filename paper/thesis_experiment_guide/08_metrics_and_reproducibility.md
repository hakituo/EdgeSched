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

- 第一段音频 TTFA；
- queue wait；
- chunk synthesis time；
- 音频时长；
- RTF mean / p95；
- gap / underrun count；
- 完整语音回复时间。

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

---
