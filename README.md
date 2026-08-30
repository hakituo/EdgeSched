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
- `services/tts/`：独立句子级流式 Qwen3-TTS Base 子系统。
- `paper/THESIS_EXPERIMENT_GUIDE_DRAFT.md`：新版论文与实验路线真源。
- `paper/rewrite/`：新版正文工作区；旧英文稿已归档。

当前目录分层：

| 路径 | 当前语义 |
|---|---|
| `services/`, `domain/`, `data/adapters/` | Python 控制面和运行适配器 |
| `cpp_scheduler/` | C++ 执行面；`build/` 是本地生成目录，不是源码 |
| `experiments/` | 可复现实验入口 |
| `experiment_results/` | 工作产物；旧论文结果集中在 `_archive/` |
| `paper/rewrite/` | 新版论文正文工作区 |
| `paper/experiment_data/` | 新版论文证据入口，当前尚未晋升数据 |
| `paper/archive/` | 重写前论文链条和更早历史材料 |
| `docs/archive/` | 被当前项目上下文和论文总指南取代的历史文档 |

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

Windows 到 WSL2 使用 `scripts/sync_to_wsl2_safe.ps1` 增量同步。该脚本不删除 WSL2 独有模型和 Forge 文件；不要改成镜像删除模式。工作区另有未完成改动时使用 `-SkipGit`，避免默认自动提交把无关文件一起纳入。

WSL2 的可复现实验依赖使用 `requirements-wsl-cu128.txt`。它锁定当前已验证的 PyTorch CUDA 12.8、Qwen3-TTS、Transformers 和 llama-cpp-python 版本，并强制后者从源码构建 CUDA 版。安装前还需按文件头部说明安装 SoX、PortAudio 与 ALSA→Pulse 桥接；通用依赖中的 ONNX Runtime、Faster-Whisper/CTranslate2、sounddevice、blendmodes 和 Pillow 以 `requirements.txt` 的固定兼容版本为准。音频探针优先使用 WSLg Pulse，并在端点不可用时回退到 Windows WASAPI；静音链路通过不等于真实并发 TTS 已满足无 underrun。

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

# 独立 Qwen3-TTS GPU smoke（默认 ICL，需准确转写）
python experiments/tts/smoke_qwen3_tts_base.py --device cuda:0 --ref-text '准确转写'
```

TTS 的音色画像、x-vector 消融、CPU/GPU smoke 和证据边界见 [Qwen3-TTS Base 独立子系统](docs/TTS_SUBSYSTEM.md)。2026-08-28 的 [GPU 句长画像](docs/TTS_GPU_SENTENCE_LENGTH_PROFILE_20260828.md) 显示 45/45 条真实样本 RTF 均大于 1，因此当前先研究真正音频流式或后端优化，不直接铺开 CPU 核心组合。

`experiment_results/` 是运行产物；`paper/experiment_data/01_paper_main/` 是新版正文数据入口，但 2026-08-30 重置后尚未晋升任何新数据。新结果必须经过多次迭代、manifest、聚合检查和人工确认，不能自动进入正文。

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

论文入口见 [paper/README.md](paper/README.md)，新版骨架位于 [paper/rewrite/](paper/rewrite/README.md)，数据边界见 [实验数据说明](paper/experiment_data/README.md)。重写前稿件和数据位于 `paper/archive/pre_rewrite_20260830/`，只用于追溯。

## 版本管理提示

父仓库 `.gitignore` 当前忽略整个 `/legacy/`，所以本目录修改默认不会显示在父仓库 Git 状态中。需要正式纳入版本管理时，应先决定独立仓库方案或有意识地调整父仓库忽略策略。
