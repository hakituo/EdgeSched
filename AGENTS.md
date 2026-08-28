# EdgeSched / MVP Core 项目规则（Codex 局部入口）

本文件补充父仓库 `../../AGENTS.md`，只约束 `legacy/mvp_core/`。冲突时以本文件的局部规则为准；父仓库的安全、中文注释、虚拟环境和记录规则仍然有效。

## 开始任务前

1. 先读 `docs/PROJECT_INDEX.md`，按任务类型定位文件。
2. 涉及架构、实验口径或运行环境时，再读 `docs/PROJECT_CONTEXT.md`。
3. 修改本模块前必须读 `.trae/skills/xiaoyou-mvp-core/SKILL.md`。
4. 论文数据任务同时读 `paper/experiment_data/README.md`；C++ 任务同时读 `cpp_scheduler/README.md`。

## 项目边界

- 本目录是 Xiaoyou Core 的实验/论文原型，论文对外名为 **EdgeSched**，不是主应用后端。
- Python 控制面位于 `services/`；原生执行面位于 `cpp_scheduler/`；实验入口位于 `experiments/`；论文证据位于 `paper/experiment_data/`。
- `main.py` 只是 FastAPI/WebSocket 演示入口，不代表论文全部实验的统一入口。
- `config.py` 优先适配父仓库 `config/integrated_config.py`，独立运行时才使用环境变量回退配置。

## 环境与运行

- 真实模型、CUDA、C++ 编译和论文实验优先在 WSL2 `/home/leslie/mvp_core` 运行，使用 `/home/leslie/.venvs/mvp_core-cu128`。
- Forge 使用独立环境 `/home/leslie/.venvs/forge-cu128`，服务目录为 `/home/leslie/mvp_core/stable-diffusion-webui-forge-main/`。
- Windows 侧只做编辑、静态验证和结果收集；可用父仓库 `venv_core` 运行不依赖 GPU 的验证。
- Windows 到 WSL2 只使用 `scripts/sync_to_wsl2_safe.ps1` 的增量同步语义，禁止 `--delete`、镜像覆盖或清理 WSL2 模型目录。
- 不修改虚拟环境；确需修改依赖时先与用户讨论。

## 实验与证据规则

- 明确区分 Mock、仿真、真实模型、真实 Forge 和论文正文证据；Mock 通过只证明链路和统计口径，不证明生产性能。
- `experiment_results/` 是运行产物；`paper/experiment_data/01_paper_main/` 才是当前论文正文数据源。
- 更新正文数据前必须保留原始迭代、环境、参数、随机种子和 manifest；不得用单次运行、历史归档或 `paper/previous/` 覆盖正文证据。
- 论文中的性能数字必须能反向定位到 JSON、生成脚本和运行环境。没有真实重跑证据时，只能写“设计目标”“模拟结果”或“待验证”。
- `docs/CPP_SCHEDULER_EVALUATION_REPORT.md`、`docs/DIFFERENCE_ASSESSMENT_REPORT.md`、`docs/NVRAM_ARCHITECTURE_DESIGN.md` 是历史设计材料，不是当前实现真源。

## 修改与验证

- 调度主路径：`services/task_scheduler.py`、`services/admission_controller.py`、`services/resource_manager.py`、`services/state_backplane.py`。
- C++ 桥接：`services/cpp_scheduler_engine.py`、`services/cpp_scheduler_wrapper.py`、`cpp_scheduler/bindings/python_bindings.cpp`。
- Phase-Aware：`services/phase_aware/` 与 `cpp_scheduler/core/phase_strategy.*`，Python/C++ 契约变更必须双向检查。
- 每次优化补充或更新 `tests/` 下的验证；默认先跑静态/Mock 测试，真实 GPU/Forge 测试在 WSL2 单独执行。
- 修改目录、入口、环境、数据口径或模块职责时，同步更新 `docs/PROJECT_INDEX.md`、`docs/PROJECT_CONTEXT.md` 和相关 README。
- 不把 `cpp_scheduler/build/`、模型、运行日志、临时输出或本地 mmap 数据当作源码。

