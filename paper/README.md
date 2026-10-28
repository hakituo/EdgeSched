# EdgeSched 论文重写工作区

当前论文准备按照 `thesis_experiment_guide/` 中按任务拆分的指南重新撰写。旧英文稿、旧章节、旧图表、旧正文数据和历史报告已经归档，不再把它们当成新版论文真源。

## 当前入口

| 路径 | 作用 |
|---|---|
| `thesis_experiment_guide/README.md` | 新版研究问题、实验顺序、证据口径和当前门禁的模块化索引 |
| `thesis_experiment_guide/11_slack_aware_scheduling_direction.md` | 当前论文主线：余量感知的驻留调度，含既有证据、V0 结果与 V0–V4 验证顺序 |
| `THESIS_EXPERIMENT_GUIDE_DRAFT.md` | 兼容入口，只负责把旧引用导航到模块化指南 |
| `rewrite/README.md` | 新正文工作区状态与写作门槛 |
| `rewrite/OUTLINE.md` | 按新版研究路线整理的章节骨架 |
| `edgesched_phase_aware_formulation.tex` | 当前 Phase-Aware 弹性驻留与联合资源调度公式；未验证配置不进入候选集 |
| `system_architecture.tex` | 当前资源流、控制流、画像证据与异构执行面架构图 |
| `references.bib` | 可复用参考文献库；正式写作前仍需逐条核验 |
| `experiment_data/01_paper_main/` | 新版正文数据区；当前未晋升任何新数据 |
| `experiment_data/02_auxiliary/` | 新版辅助数据区 |
| `experiment_data/03_raw_iterations/` | 新版晋升后的原始迭代证据区 |
| `archive/pre_rewrite_20260830/` | 重写前的论文、图表、工具与实验数据完整快照；`manuscript_variants/` 保存中文/其他版本旧稿 |
| `archive/legacy_materials_before_rewrite/` | 更早的历史报告、旧产物和 `thesis_materials/` |

## 当前状态

- 新版论文还没有可编译正文；旧 `main_en.tex` 及 `sections_en/` 已归档，其他旧版本在 `archive/pre_rewrite_20260830/manuscript_variants/`。
- 2026-10-26 起主线改为**余量感知的驻留调度**（时间维度），研究问题、证据与验证顺序见 `thesis_experiment_guide/11_slack_aware_scheduling_direction.md`；Gate B 收缩，S0 v2 不登记 Go。
- V0 共跑诊断已完成并放行（`colocation_v0_20261026_v1|v2`，28/28 通过，共跑速度恢复到隔离画像的 60%）；下一步是 V0.5（S0 v2 用 14 线程重跑）和 V1（切换代价画像），再进入 V2 离线模拟。
- SM-A 已在原生 Linux 转 GO（有效 SM 36/26/18/8），SM-B 待排。
- 方法公式与系统架构图可以先独立维护，但不得提前填入尚未完成实验的性能结论。
- 新版 `01_paper_main/` 为空，不能把 `experiment_results/` 中的探索数据直接写进正文。
- 旧稿可以用于理解历史背景，不能复制旧性能数字冒充当前 Qwen3.5/Qwen3-TTS/Forge 结果。

## 数据晋升

1. 在冻结的实验环境完成正式运行，manifest 记录主机形态、内存上限与 swap 状态。
2. 核对真实后端、manifest、模型哈希、Git provenance、原始记录数和失败文件。
3. 保留完整原始轮次并重建聚合结果。
4. 人工确认后才复制到 `experiment_data/`。
5. 根据冻结数据编写新版正文、图表和表格。

归档内容保持只读。若需要复用旧 LaTeX 模板或画图代码，应复制到新版工作区后重新核对，不要直接在归档目录继续编辑。
