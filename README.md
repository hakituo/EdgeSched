# EdgeSched（MVP Core）

small 9B 离线筛选 `20260907_204516_653987` 已复算一致：T0/K20、T3/K30 仅为容量候选，长上下文余量不足 16 MiB。下一步用户运行 `scripts/experiments/run_small_9b_tts_compat.ps1`，C512/K18 配 T0/T3 各 1 轮；加载期另需验证，不冻结阈值，见 [验收与命令](docs/SMALL_9B_NEXT_COMBINATIONS_20260907.md)。

已完成 [阶段组合离线筛选](docs/SMALL_9B_NEXT_COMBINATIONS_20260907.md)：`scripts/experiments/run_small_9b_screen.ps1`。WSL 大版 9B 已按用户要求删除，小版和视觉文件保留；历史 large 复测入口需要恢复原文件。

small 9B 视觉生命周期五轮 `20260907_171635` 已 5/5、swap 0；加载/预热 0.481 s、编码 0.147 s、释放 0.011 s，见 [阶段画像](docs/SMALL_9B_VISION_PROFILE_20260907.md)。这是独立进程画像，保温/提前释放策略仍待验证。

完整曲线批次 `20260907_143917` 已离线逐条核验，288/288、swap 0，见 [16 档曲线与 KV 验收](docs/SMALL_9B_CURVES_20260907.md)。

> **2026-09-07 9B 版本选定**：后续主模型为 small `1a35ab89…` + F16 projector `e36da9b9…`。纯文本对比 12/12、small 视觉 Answer 1/1 通过，固定图形内容描述完整；专用视觉启动器默认 Small，Large 可显式复测。新版本 C2 90/90、C3 150/150、KV 48/48 已通过，4B 对照与旧哈希证据保留。单图不是完整质量评测或 TTS 共置结论。详见 [选型与验收](docs/NINE_B_MODEL_COMPARISON.md)。

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
- `paper/thesis_experiment_guide/`：按任务拆分的新版论文与实验路线真源；旧路径仅保留兼容导航。
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

**2026-09-25 起主开发环境就是本机原生 Linux（Ubuntu-24.04-E）**，编辑、运行、git 提交都在此处，不再使用 WSL2 双副本流程：

| 用途 | 路径 |
|---|---|
| 代码与 git 仓库 | `/home/leslie/mvp_core`（`.git` 在工作树内，origin = `https://github.com/hakituo/EdgeSched`） |
| 主虚拟环境 | `/home/leslie/xiaoyou-core/.venv`（共享，Python 3.13 / torch 2.12+cu130；`/home/leslie/.venvs/mvp_core-cu128` 为指向它的软链） |
| Forge 虚拟环境 | `/home/leslie/.venvs/forge-cu128`（Python 3.10 / torch 2.10+cu128） |
| GGUF 模型 | `/home/leslie/mvp_core/models/llm`（软链到 970 EVO，跑实验前确认盘已挂载） |
| 图像模型 | `/home/leslie/mvp_core/models/Image`；Forge 额外权重软链自 970 EVO |
| Forge | `/home/leslie/mvp_core/stable-diffusion-webui-forge-main/` |

提交直接在 Linux 完成，不再走 `scripts/sync_to_wsl2_safe.ps1`（该 Windows→WSL2 同步+自动提交流程已废止）：

```bash
git add <本任务相关文件>      # 显式路径，不要 git add -A
git commit -m "一句话说明这次改了什么"
git push origin main
```

提交信息禁止 `auto`/`更新`/`同步` 这类无信息文案；工作区另有其他会话的未完成改动时只暂存本任务文件。详细过程（背景/改动/验证）记入 `docs/UPDATES.md`。文本文件统一 LF（见根目录 `.gitattributes`）。

可复现实验依赖参考 `requirements-wsl-cu128.txt`（文件名保留历史命名）。它锁定当前已验证的 PyTorch CUDA 12.8、Qwen3-TTS、Transformers 和 llama-cpp-python 版本，并强制后者从源码构建 CUDA 版。安装前还需按文件头部说明安装 SoX、PortAudio 与 ALSA→Pulse 桥接；通用依赖中的 ONNX Runtime、Faster-Whisper/CTranslate2、sounddevice、blendmodes 和 Pillow 以 `requirements.txt` 的固定兼容版本为准。音频输出走本机 ALSA/PulseAudio；静音链路通过不等于真实并发 TTS 已满足无 underrun。

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

```bash
PYTHONPATH=/home/leslie python tests/scripts/verify_project_context.py
```

不依赖真实模型的高频测试：

```bash
PYTHONPATH=/home/leslie python -m pytest tests/test_chat_service.py tests/test_phase_aware_quick.py -q -o asyncio_mode=auto
```

C++ 绑定、真实 GGUF 和 Forge 测试在本机 Linux 直接运行（GPU/设备访问需在沙箱外执行），不能用 Mock 结果替代。

## 论文

论文入口见 [paper/README.md](paper/README.md)，新版骨架位于 [paper/rewrite/](paper/rewrite/README.md)，数据边界见 [实验数据说明](paper/experiment_data/README.md)。重写前稿件和数据位于 `paper/archive/pre_rewrite_20260830/`，只用于追溯。

## 版本管理提示

父仓库 `.gitignore` 当前忽略整个 `/legacy/`，所以本目录修改默认不会显示在父仓库 Git 状态中。需要正式纳入版本管理时，应先决定独立仓库方案或有意识地调整父仓库忽略策略。


辅助任务 CPU 核数画像入口已实现：`experiments/elastic_residency/auxiliary_core_budget.py`，复用物理核租约与采样器，支持 JSON RAG 1/2 核、BGE/Whisper 1/2/4 核和 `R_sys=0/1/2/4` 隔离敏感性。模块职责、运行方法和 D3 正式并发验收边界见 [辅助核数画像](docs/AUXILIARY_CORE_BUDGET.md)。


BGE 新增独立 `bge_cached` 候选缓存画像：模型加载、候选预计算、query+排序分别记录，保留原 `bge` 模式及旧结果。用户完成的 R_sys=2 曲线已 40/40 通过有效性复算，新缓存模式仅做静态/假后端验证、等待用户补测；一条 PowerShell 启动命令见 [运行说明](docs/AUXILIARY_CORE_BUDGET.md)。
