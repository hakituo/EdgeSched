# 新版论文正文工作区

这里承载按照 `../thesis_experiment_guide/README.md` 重写的论文正文。当前处于实验定标阶段，尚未创建正式 LaTeX 主稿。

## 开始写作的门槛

- C3 的 T0/T3、Forge direct NVML 校准和真实共置残差完成；
- 按 context 冻结 `G_safe/H_TTS/H_D`；
- D 阶段 CPU 核心预算和 `R_sys` 结果完成；
- E 阶段真实 LLM→TTS 重叠形成 Go/No-Go 结论；
- 计划写入正文的结果已人工晋升到 `../experiment_data/`。

在这些门槛之前，可以完善问题定义、方法框架和 Related Work，但不能提前写入未经验证的性能结论。

旧稿位于 `../archive/pre_rewrite_20260830/`，只作结构和措辞参考。
