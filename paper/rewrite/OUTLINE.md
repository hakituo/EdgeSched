# EdgeSched 新版论文骨架

本骨架服从 `../thesis_experiment_guide/README.md` 及其主题模块，最终章节名可在证据冻结后调整。

1. **Abstract**：问题、方法、真实系统边界和主要实测结果。
2. **Introduction**：单机多模态助手中的显存与 CPU 竞争、现有二元迁移的不足、贡献。
3. **Background and Motivation**：LLM 分层驻留、按需视觉编码组件、TTS 组件放置、Diffusion burst、WSL2/消费级 GPU 约束；区分视觉理解与图像生成。
4. **Problem Formulation**：工作流状态、PlacementPlan、容量约束、交互约束和回退语义。
5. **System Design**：画像、准入、弹性驻留、CPU 核心预算、迟滞和故障回退。
6. **Implementation**：Python 控制面、llama.cpp、Qwen3-TTS、Forge、direct NVML、证据契约；明确 C++ 边界。
7. **Evaluation**：文本与视觉组件分阶段画像、容量边界与迁移调度、目标组合兼容性、GPU/CPU 放置比较；视觉常驻/请求后释放/队列保温对照，完整计入加载与切换。SM 性能比较在能力门通过后开展；提前释放单列待实现与验证。
8. **Discussion and Limitations**：TTS RTF、WSL 驱动限制、非通用 CUDA 分页、单用户单机边界和 No-Go 条件。
9. **Related Work**：边缘推理、模型迁移、异构调度、多模态流水线。
10. **Conclusion**：仅总结由正式实验支持的结论。
