# EdgeSched 论文工作区

当前论文以英文稿为主。正文、证据数据、图表和历史材料分层保存，避免把旧稿或单次实验误当成当前结论。

## 当前入口

| 路径 | 作用 |
|---|---|
| `main_en.tex` | 英文论文主文件 |
| `sections_en/` | 00 Abstract 到 07 Conclusion |
| `references.bib` | 参考文献库 |
| `figures/` | 当前论文图表（PDF/PNG） |
| `tables/` | 当前论文表格（LaTeX） |
| `experiment_data/01_paper_main/` | 正文聚合数据真源 |
| `experiment_data/02_auxiliary/` | 辅助说明数据 |
| `experiment_data/03_raw_iterations/` | 原始迭代结果 |
| `scripts/generate_figures.py` | 从 JSON 重建图表/表格入口 |
| `THESIS_EXPERIMENT_GUIDE_DRAFT.md` | 论文结构、实验设计和写作工作指南 |
| `draft/` | Markdown 章节草稿 |
| `previous/` | 历史稿件与产物，仅供追溯 |

数据文件到论文图表的映射见 `experiment_data/README.md`。项目整体证据边界见当时的 `docs/PROJECT_CONTEXT.md`。

## 编译

Windows 使用 `compile_en.bat`；WSL/Linux 使用 `latexmk_engine.sh` 或本机 LaTeX 环境。编译产生的 `.aux`、`.log`、`.out`、`.bbl`、`.blg`、`.pdf` 等是生成产物，不是论文内容真源。

## 历史说明

该说明是 2026-08-30 重写前的原始入口快照。路径已经随归档重排，不代表当前工作区结构或证据状态。
