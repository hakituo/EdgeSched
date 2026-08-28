# EdgeSched（MVP Core）

EdgeSched 是 Xiaoyou Core 的本地多模态调度研究原型，目录名沿用 `mvp_core`。项目围绕单 GPU/有限显存场景中的 LLM 常驻、图像突发、CPU backfill、迁移代价、准入控制和崩溃恢复开展实现与实验。

第一次进入项目，请从以下两份文档开始：

- [项目索引](docs/PROJECT_INDEX.md)：按任务快速定位代码、实验和验证。
- [项目上下文](docs/PROJECT_CONTEXT.md)：架构、数据流、环境、证据边界和维护风险。

## 核心结构

```text
FastAPI / WebSocket / experiment
                 |
                 v
GlobalTaskScheduler (Python control plane)
  |-- AdmissionController
  |-- ResourceManager
  |-- StateBackplane (mmap)
  |-- PythonWorker / image queue
  |-- PhaseAwareScheduler
  `-- CPPSchedulerEngine
          |
          v
ResourceIsolationScheduler (C++ execution plane)
  |-- native queue / workers / monitoring
  |-- llama.cpp worker + KV cache
  `-- C++ phase-aware strategy
```

主要入口：

- `main.py`：FastAPI 演示服务，提供 `/ws` 和 `/health`。
- `services/task_scheduler.py`：Python 全局调度门面。
- `services/cpp_scheduler_engine.py`：C++ 调度器和模型迁移桥接。
- `services/phase_aware/`：Phase-Aware workflow、GPU lease、代价决策和 backfill。
- `experiments/run_all_experiments.py`：论文实验统一编排。
- `paper/main_en.tex`：当前英文论文主文件。

## 运行环境

真实模型、CUDA、C++ 构建和论文实验的推荐环境是 WSL2：

| 用途 | 路径 |
|---|---|
| 代码 | `/home/leslie/mvp_core` |
| 主虚拟环境 | `/home/leslie/.venvs/mvp_core-cu128` |
| Forge 虚拟环境 | `/home/leslie/.venvs/forge-cu128` |
| GGUF 模型 | `/home/leslie/mvp_core/models/llm` |
| 图像模型 | `/home/leslie/mvp_core/models/Image` |
| Forge | `/home/leslie/mvp_core/stable-diffusion-webui-forge-main/` |

Windows 到 WSL2 使用 `scripts/sync_to_wsl2_safe.ps1` 增量同步。该脚本不删除 WSL2 独有模型和 Forge 文件；不要改成镜像删除模式。

WSL2 的可复现实验依赖使用 `requirements-wsl-cu128.txt`。它锁定当前已验证的 PyTorch CUDA 12.8、Qwen3-TTS、Transformers 和 llama-cpp-python 版本，并强制后者从源码构建 CUDA 版。安装前还需按文件头部说明安装 SoX；通用依赖仍以 `requirements.txt` 为准。

## 演示服务

在依赖和配置已准备好的环境中：

```bash
python main.py
```

- WebSocket：`ws://localhost:8000/ws`
- Health：`http://localhost:8000/health`

`config.py` 优先复用父仓库 `config/integrated_config.py`。独立运行且导入失败时，才读取 `XIAOYOU_BENCH_*`、`XIAOYOU_USE_CPP_*` 等环境变量。模型不存在时演示服务会回退到 Mock，因此“服务启动成功”不是“真实模型路径已验证”。

## 实验

完整说明见 [实验指南](experiments/README.md)。常用入口：

```bash
# 无 Forge 的实验集合
python experiments/run_no_forge_experiments.py --iterations 5

# 需要 Forge 的迁移与真实 workload
python experiments/run_forge_experiments.py --experiments migration --iterations 5

# 全部论文实验编排
python experiments/run_all_experiments.py --experiments all --iterations 5

# Mock/Real 性能套件
python experiments/run_performance_suite.py --workload mock
```

`experiment_results/` 是运行产物；论文正文当前采用的数据位于 `paper/experiment_data/01_paper_main/`。新结果必须经过多次迭代、manifest、聚合检查和人工确认，不能自动覆盖正文数据。

## 验证

静态项目索引验证：

```powershell
..\..\venv_core\Scripts\python.exe tests\scripts\verify_project_context.py
```

不依赖真实模型的高频测试：

```powershell
..\..\venv_core\Scripts\python.exe -m pytest tests\test_chat_service.py tests\test_phase_aware_quick.py -q
```

C++ 绑定、真实 GGUF、迁移和 Forge 测试必须在对应 WSL2 环境单独运行，不能用 Mock 结果替代。

## 论文

论文入口与数据映射见 [paper/README.md](paper/README.md) 和 [实验数据说明](paper/experiment_data/README.md)。历史稿件位于 `paper/previous/`，只用于追溯，不是当前实现或当前数据真源。

## 版本管理提示

父仓库 `.gitignore` 当前忽略整个 `/legacy/`，所以本目录修改默认不会显示在父仓库 Git 状态中。需要正式纳入版本管理时，应先决定独立仓库方案或有意识地调整父仓库忽略策略。
