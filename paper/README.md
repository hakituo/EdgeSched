# EdgeSched 论文重写工作区

当前论文准备按照 `THESIS_EXPERIMENT_GUIDE_DRAFT.md` 重新撰写。旧英文稿、旧章节、旧图表、旧正文数据和历史报告已经归档，不再把它们当成新版论文真源。

## 当前入口

| 路径 | 作用 |
|---|---|
| `THESIS_EXPERIMENT_GUIDE_DRAFT.md` | 新版研究问题、实验顺序、证据口径和完成标准的规划真源 |
| `rewrite/README.md` | 新正文工作区状态与写作门槛 |
| `rewrite/OUTLINE.md` | 按新版研究路线整理的章节骨架 |
| `references.bib` | 可复用参考文献库；正式写作前仍需逐条核验 |
| `experiment_data/01_paper_main/` | 新版正文数据区；当前未晋升任何新数据 |
| `experiment_data/02_auxiliary/` | 新版辅助数据区 |
| `experiment_data/03_raw_iterations/` | 新版晋升后的原始迭代证据区 |
| `archive/pre_rewrite_20260830/` | 重写前的论文、图表、工具与实验数据完整快照 |
| `archive/legacy_materials_before_rewrite/` | 更早的历史报告和旧产物 |

## 当前状态

- 新版论文还没有可编译正文；旧 `main_en.tex` 及 `sections_en/` 已归档。
- 新版 `01_paper_main/` 为空，不能把 `experiment_results/` 中的探索数据直接写进正文。
- 当前优先完成 C3 收尾、D/E 阶段真实实验和 Go/No-Go 判断，再冻结 Evaluation 数据和论文主张。
- 旧稿可以用于理解历史背景，不能复制旧性能数字冒充当前 Qwen3.5/Qwen3-TTS/Forge 结果。

## 数据晋升

1. 在冻结的 WSL2 环境完成正式运行。
2. 核对真实后端、manifest、模型哈希、Git provenance、原始记录数和失败文件。
3. 保留完整原始轮次并重建聚合结果。
4. 人工确认后才复制到 `experiment_data/`。
5. 根据冻结数据编写新版正文、图表和表格。

归档内容保持只读。若需要复用旧 LaTeX 模板或画图代码，应复制到新版工作区后重新核对，不要直接在归档目录继续编辑。
