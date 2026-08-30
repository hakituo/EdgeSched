# EdgeSched 新版论文骨架

本骨架服从 `../THESIS_EXPERIMENT_GUIDE_DRAFT.md`，最终章节名可在证据冻结后调整。

1. **Abstract**：问题、方法、真实系统边界和主要实测结果。
2. **Introduction**：单机多模态助手中的显存与 CPU 竞争、现有二元迁移的不足、贡献。
3. **Background and Motivation**：LLM 分层驻留、TTS 组件放置、Diffusion burst、WSL2/消费级 GPU 约束。
4. **Problem Formulation**：工作流状态、PlacementPlan、容量约束、交互约束和回退语义。
5. **System Design**：画像、准入、弹性驻留、CPU 核心预算、迟滞和故障回退。
6. **Implementation**：Python 控制面、llama.cpp、Qwen3-TTS、Forge、direct NVML、证据契约；明确 C++ 边界。
7. **Evaluation**：单模型画像、真实共置、无图像重叠、图像期间共存、策略比较、消融与稳健性。
8. **Discussion and Limitations**：TTS RTF、WSL 驱动限制、非通用 CUDA 分页、单用户单机边界和 No-Go 条件。
9. **Related Work**：边缘推理、模型迁移、异构调度、多模态流水线。
10. **Conclusion**：仅总结由正式实验支持的结论。
