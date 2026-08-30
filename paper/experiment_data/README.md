# 新版论文实验数据

> **重置日期**：2026-08-30
> **状态**：尚未晋升新版正文数据。

旧论文数据已完整保存在 `../archive/pre_rewrite_20260830/experiment_data/`。当前目录只接收按照 `THESIS_EXPERIMENT_GUIDE_DRAFT.md` 新口径完成、人工核对并明确晋升的数据。

```text
experiment_data/
├── 01_paper_main/     # 新版正文直接使用的聚合数据
├── 02_auxiliary/      # 辅助、敏感性和补充数据
└── 03_raw_iterations/ # 与聚合结果对应的原始迭代和 manifest
```

晋升前必须确认：真实后端、完整矩阵、失败记录、模型哈希、环境、随机种子、CPU affinity、direct NVML 口径和 Git provenance。`experiment_results/` 中存在文件不等于已经晋升。
