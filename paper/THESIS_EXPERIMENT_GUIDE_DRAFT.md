# EdgeSched 论文与实验指南（兼容入口）

原单文件指南已经拆分到 [thesis_experiment_guide/README.md](thesis_experiment_guide/README.md)。

## 新会话最小读取集

1. [指南索引](thesis_experiment_guide/README.md)
2. [Go/No-Go 与当前任务状态](thesis_experiment_guide/10_gates_and_current_status.md)
3. 当前任务对应的一个主题模块
4. 运行实验时再读：
   - `docs/ELASTIC_RESIDENCY_EXPERIMENT_EXECUTION_PLAN_20260828.md`
   - `docs/ELASTIC_RESIDENCY_RESULT_CATALOG_20260830.md`

## 当前一句话状态

C2+C3 direct NVML 的 240 条 LLM 曲线与 48 条 KV 校准已完成；当前先补 T0/T3、Forge 的 direct NVML 同后端校准和 LLM+TTS 共置残差，再进入 CPU 核数实验。SM-A 在 WSL2 中暂定 NO_GO，等待原生 Ubuntu 恢复验证。

本文件只保留兼容导航，不再复制研究设计和实验状态，避免多个长文档继续漂移。
