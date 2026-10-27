# EdgeSched 文件夹归类地图

> 2026-10-26 按实际目录扫描整理。归类完成后已执行下面的第 1–6 步整理，第 7 步按建议跳过。
> 「入库」= 被 git 跟踪的文件数 / 目录内文件总数。

## 一览

| 类别 | 是什么 | 体量 |
|---|---|---|
| A 当前主线 | 弹性驻留 / S0 / SM / 视觉实验代码和测试 | 小，几 MB |
| B 早期调度器原型 | 旧论文那套调度器（迁移、准入、背板、C++） | 源码小，C++ 依赖和构建 325 MB |
| C 论文设计（现行） | 实验指南、重写骨架、公式、参考文献 | 小 |
| D 论文旧稿 | 2–5 月的旧论文和旧材料，归档里已有一份 | 约 17 MB |
| E 文档 | `docs/` 下的项目说明和实验记录 | 388 KB |
| F 运行产物 | 实验结果、日志、生成的图片和音频 | 约 1.9 GB |
| G 外部大件 | 模型、Forge、wheel | 约 9.8 GB |
| H 配置与工具 | AGENTS、CI、依赖清单、启动脚本 | 小 |

---

## A 当前主线（论文实验在用）

| 路径 | 内容 | 入库 |
|---|---|---|
| `experiments/elastic_residency/` | C1–C3 曲线、KV、共置、Forge、辅助核数、分析脚本 | 是 |
| `experiments/elastic_residency/s0_pipeline/` | S0 端到端 LLM→TTS 流水线（协议 v2） | 是 |
| `experiments/elastic_residency/sm/` | SM-A 能力探针（10-13 已 GO） | 是 |
| `experiments/elastic_residency/vision/` | 视觉端点门与画像 | 是 |
| `experiments/tts/` | TTS 后端对照、B2 画像 | 是 |
| `services/tts/` | 独立句子级 TTS 子系统 | 是 |
| `tests/elastic_residency/`、`tests/tts/` | 主线测试（10-13 记录：`tests/elastic_residency/` 531 passed） | 是 |
| `scripts/experiments/*.ps1` | Windows 启动器（9 个） | 是。主环境已经换成 Linux，这些是历史入口 |

## B 早期调度器原型（旧论文链，代码仍在）

当前论文主线基本不走这些代码，但它们是 EdgeSched 的起点，也是 `main.py` 演示服务的依赖。

| 路径 | 内容 |
|---|---|
| `main.py`、`config.py`、`__init__.py` | FastAPI 演示入口与配置 |
| `services/`（除 `tts/`） | 全局调度、准入、资源管理、状态背板、NVRAM 模拟、Phase-Aware；还有 `biological_system.py`、`bio_state_store.py`，看起来是从主应用带过来的 |
| `domain/`、`presentation/`、`shared/`、`utils/`、`memory/`、`include/` | 演示服务的分层代码（`memory/` 只有 `persistent_state.py`，和 Autarkia 的记忆系统无关） |
| `cpp_scheduler/` | C++ 执行面：源码约 0.5 MB，`deps/` 133 MB，`build/` 192 MB（56/3535 入库） |
| `experiments/{admission,baselines,migration,recovery,stress,workload,phase_aware,overhead,scaling,semantic_validation,framework,charts}` | 旧论文的实验，最后改动在 2026-08-28 |
| `experiments/run_*.py` | 旧实验编排入口 |
| `tests/test_*.py`（根层）、`tests/verify_*.py`、`tests/benchmark_*.py` | 旧调度器的测试 |

## C 论文设计（现行真源）

| 路径 | 内容 |
|---|---|
| `paper/thesis_experiment_guide/` | **研究路线真源**，01–10 模块，10 是门禁状态 |
| `paper/rewrite/` | 新版正文骨架（`OUTLINE.md`），正文尚未开写 |
| `paper/edgesched_phase_aware_formulation.tex` | 方法公式 |
| `paper/system_architecture.tex` | 架构图 |
| `paper/references.bib` | 参考文献（含 HorizonServe、VoxServe 等） |
| `paper/experiment_data/` | 正文数据入口；`01_paper_main/` 现只有 README，6 个 5–6 月旧 JSON 与 `migration/` 已于 2026-10-26 清空（归档保留副本） |
| `paper/README.md`、`paper/THESIS_EXPERIMENT_GUIDE_DRAFT.md`（18 行跳转页） | 导航 |

## D 论文旧稿（已归档，顶层副本已清理）

`paper/README.md` 说旧稿"已归档"，归档目录为 `paper/archive/pre_rewrite_20260830/`。2026-10-26 已把 9-26 入库时一并跟踪的顶层旧稿处理完：

| 路径 | 处理 |
|---|---|
| `paper/main_en.tex`、`paper/affinity_routing_formulas.*` | 与归档完全相同的副本，删除顶层重复件（归档保留） |
| `paper/sections_en/` | 8 个文件按 `diff --strip-trailing-cr` 与归档逐字一致，仅行尾不同，删除顶层副本 |
| `paper/main_cn.*`、`main_zh.tex`、`main_en_full.tex`、`main.pdf`、`sections_cn/`、`sections_zh/`、`SUMMARY.md`、`test_compile.tex`、`compile_cn.bat` | 归档中没有的旧版本，移入 `paper/archive/pre_rewrite_20260830/manuscript_variants/` |
| `paper/draft/`、`paper/previous/`、`paper/figures/`、`paper/tables/`、`paper/scripts/`、`paper/EdgeSched_draft.pdf`、`One_Page_Executive_Summary.*`、`System_Architecture_and_Performance_Report.*`、`paper/SUMMARY.md` | 归档已有等价副本，删除顶层重复件 |
| `thesis_materials/` | 移入 `paper/archive/legacy_materials_before_rewrite/thesis_materials/` |
| 根目录 `THESIS_EXPERIMENT_GUIDE_DRAFT.md`（1326 行旧完整版）、`TODO.md`、`EXPERIMENT_RERUN_PLAN.md`、`todo.txt` | 移入 `docs/archive/pre_rewrite_20260830/` |

`paper/` 顶层现只保留现行导航（`README.md`、18 行兼容入口 `THESIS_EXPERIMENT_GUIDE_DRAFT.md`）、现行设计文件（`edgesched_phase_aware_formulation.tex`、`system_architecture.tex`、`references.bib`）、`thesis_experiment_guide/`、`rewrite/`、`experiment_data/` 与 `archive/`。

## E 文档（`docs/`）

| 子类 | 文件 |
|---|---|
| 导航与记录 | `PROJECT_INDEX.md`、`PROJECT_CONTEXT.md`、`UPDATES.md`、本文件 |
| 实验计划与目录 | `ELASTIC_RESIDENCY_EXPERIMENT_EXECUTION_PLAN_20260828.md`、`ELASTIC_RESIDENCY_RESULT_CATALOG_20260830.md` |
| 分批实验验收 | `LLM_RESIDENCY_C1_20260829.md`、`TTS_COMPONENT_OFFLOAD_MATRIX_20260829.md`、`TTS_GPU_SENTENCE_LENGTH_PROFILE_20260828.md`、`VISION_9B_PROBE_20260906.md`、`NINE_B_MODEL_COMPARISON.md`、`SMALL_9B_*`（3 份）、`LLM_TTS_PLACEMENT_REVIEW_20260906.md`、`AUXILIARY_CORE_BUDGET.md` |
| 子系统说明 | `TTS_SUBSYSTEM.md` |
| 历史设计（不再是真源） | `CPP_SCHEDULER_EVALUATION_REPORT.md`（`archive/historical_design/` 里有同名同内容的一份，只差第一行编码）、`NVRAM_ARCHITECTURE_DESIGN.md`、`DIFFERENCE_ASSESSMENT_REPORT.md` |
| 归档 | `docs/archive/` |

2026-10-26 已刷新 `README.md`、`PROJECT_INDEX.md`、`PROJECT_CONTEXT.md` 的开头状态段与 `PROJECT_CONTEXT.md` 的"权威环境"表述；正文中的 WSL2/Windows 内容保留为历史记录。

## F 运行产物（不入库或只少量入库）

| 路径 | 体量 | 说明 |
|---|---|---|
| `experiment_results/` | 1.7 GB / 5222 文件 | 几乎全在 `elastic_residency/` 下，含 222 份 manifest；只有 19 个文件入库 |
| `experiment_results/_archive/` | 740 KB | 重写前的旧结果 |
| ~~根目录 `TTS/`~~ | 已删除 | `experiment_results/tts/` 的旧副本，未入库；比对确认无独有内容后删除 |
| `logs/`、`experiment_results/logs/` | 约 340 KB | 运行日志；根目录旧 `*.log` 已按原日期并入 `logs/2026/<月>/<日>/` |
| `output/images/`、`experiments/output/images/` | 149 MB 等 | Forge 生成的图片 |
| ~~根目录 `*.log`（8 个）~~ | 已归置 | 2 个空文件删除，其余 6 个移入 `logs/2026/` |
| ~~`experiments/scheduler_flight_recorder.log`~~、`experiments/debug_vl_test.jpg` | — | 日志已移入 `logs/`；`debug_vl_test.jpg` 保留 |
| `data/*.dat`、`data/*.json`、`data/app.log` | — | 状态背板和 NVRAM 测试数据（`data/adapters/` 才是源码） |
| `__pycache__/`、`.pytest_cache/` | — | 缓存 |

## G 外部大件（不入库）

| 路径 | 体量 |
|---|---|
| `models/Image/` | 6.5 GB |
| `models/Qwen3-TTS-12Hz-0.6B-Base/` | 2.4 GB |
| `models/auxiliary/` | 648 MB（BGE、Whisper） |
| `models/llm/` | 软链到 970 EVO 外挂盘 |
| `stable-diffusion-webui-forge-main/` | 79 MB |
| `artifacts/wheels/` | 141 MB（`llama_cpp_python` 的 CUDA 版 wheel） |

## H 配置与工具

| 路径 | 说明 |
|---|---|
| `AGENTS.md`、`.trae/skills/xiaoyou-mvp-core/`、`.codebuddy/rules/` | 给 AI 助手的规则 |
| `requirements.txt`、`requirements-wsl-cu128.txt` | 依赖（后者的文件名沿用历史叫法） |
| `environment.yml` | 8-28 的 conda 环境描述，现在用 uv 管理的 venv，可能已过时 |
| `check_env.sh`、`run_test.sh` | 环境检查与测试脚本 |
| `scripts/sync_to_wsl2_safe.ps1` | **已废止**（AGENTS 明令禁止再用） |
| `scripts/fix_real_workload_agg.py` | 旧数据修补脚本 |

---

## 已执行的整理动作（2026-10-26）

按确认执行第 1–6 步，第 7 步按建议跳过：

1. **删除明显的垃圾**：已删根目录空日志 `experiment_no_forge.log`、`s0_test_bp.log`，以及未入库的根目录 `TTS/`（`experiment_results/tts/` 为完整版）。删除前逐个比对，确认无独有内容。
2. **旧日志归置**：其余 6 个根目录 `*.log` 与 `experiments/scheduler_flight_recorder.log` 已移入 `logs/2026/<月>/<日>/`，按原修改日期归档；根目录和 `experiments/` 下不再散落日志。
3. **清空 `01_paper_main` 旧数据**：`realistic_stress_test.json` 经 `diff --strip-trailing-cr` 确认与归档仅行尾/末行换行差异，其余旧 JSON 与归档一致；数据文件已删除，目录现只保留 `README.md`。`02_auxiliary/`、`03_raw_iterations/` 未动。
4. **`paper/` 顶层旧稿归档**：重复副本删除（归档保留），归档中没有的旧版本移入 `manuscript_variants/`；`sections_en/` 确认仅 CRLF 差异后删除顶层副本。
5. **根目录旧计划归档**：旧完整版指南、`TODO.md`、`EXPERIMENT_RERUN_PLAN.md`、`todo.txt` 移入 `docs/archive/pre_rewrite_20260830/`；`thesis_materials/` 移入 `paper/archive/legacy_materials_before_rewrite/`。
6. **刷新导航文档**：`README.md`、`docs/PROJECT_INDEX.md`、`docs/PROJECT_CONTEXT.md` 开头状态段已更新为 2026-10-26 当前状态；`paper/THESIS_EXPERIMENT_GUIDE_DRAFT.md` 兼容入口的"一句话状态"同步更新。
7. **未执行（按建议跳过）**：早期调度器原型（B 类）未收进子目录，避免影响 import 路径和 `main.py`。

本次整理未提交 git，移动/删除只发生在工作树；第 3–5 步涉及已入库文件，提交时会留下移动和删除记录。
