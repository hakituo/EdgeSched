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

数据文件到论文图表的映射见 [experiment_data/README.md](experiment_data/README.md)。项目整体证据边界见 [项目上下文](../docs/PROJECT_CONTEXT.md#6-实验与论文证据链)。

## 编译

Windows：

```bat
compile_en.bat
```

WSL/Linux 可使用 `latexmk_engine.sh`，或按本机 LaTeX 环境运行 `latexmk`。编译产生的 `.aux`、`.log`、`.out`、`.bbl`、`.blg`、`.pdf` 等是生成产物，不是论文内容真源。

## 更新实验数字

1. 从冻结的 WSL2 环境重跑对应实验，确认没有 Mock 回退。
2. 保留 `03_raw_iterations/` 中每次原始结果和 manifest。
3. 聚合后人工核对，再更新 `01_paper_main/`。
4. 运行 `paper/scripts/generate_figures.py` 重建图表/表格。
5. 运行 `paper/scripts/check_data.py`、`verify_all_iters.py` 或对应验证。
6. 对照 `sections_en/06_evaluation.tex`，确认每个数字与图表可追溯。

严禁用 `previous/`、`archive_legacy_*`、Mock 结果或单次结果直接覆盖当前正文数据。

## 历史与辅助文件

- `affinity_routing_formulas.tex` 是公式推导辅助稿，不是论文主入口。
- `EdgeSched_draft.pdf`、`main_en.pdf` 是预览产物，内容以 `.tex` 源文件为准。
- `previous/` 可能包含旧目录结构、旧实验口径和旧数字，不应作为当前代码说明。

