# EdgeSched（MVP Core）

**2026-10-26 当前状态**：主开发环境为本机原生 Linux `/home/leslie/mvp_core`（2026-09-25 起，编辑、运行和 git 提交均在此处）。S0 v2 正式比较 `s0_formal_20260920_223105_8510199`（`K24:T0` vs `K0:T0` 各 5 轮，10/10、0 failure）显示混合在主指标上全面占优（首句波形就绪 4.384 vs 10.740 s、应用层供给缺口 0 vs 6 段），但输出文本/句数/token 未受控、设备时钟仍 pending，**不登记 Gate B Go**，结论维持“收缩主张”，planner 实现暂停。SM-A 已于 2026-10-13 在原生 Linux 转 **GO**（`sm_a_20261013_native_linux_v6`，`cuCtxGetExecAffinity` 实测有效 SM 36/26/18/8），SM-B 可按门禁启动，SM-C/D/E 待排。H_D 仍 pending；S0 设备层与 BGE 缓存补测仍待完成。环境注意：970 EVO 当前被改作 Ventoy，`models/llm/` 与 Forge 额外权重软链断裂，权重本体保留，跑相关实验前需先恢复挂载。

9B 选型、small 曲线与视觉画像的既有证据分别见 [选型与验收](docs/NINE_B_MODEL_COMPARISON.md)、[16 档曲线与 KV 验收](docs/SMALL_9B_CURVES_20260907.md)、[视觉阶段画像](docs/SMALL_9B_VISION_PROFILE_20260907.md)。目录整理记录见 [文件夹归类地图](docs/FOLDER_MAP.md)。

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
| 主虚拟环境 | `/home/leslie/xiaoyou-core/.venv`（共享，Python 3.13 / torch 2.11+cu128；`/home/leslie/.venvs/mvp_core-cu128` 为指向它的软链） |
| Forge 虚拟环境 | `/home/leslie/.venvs/forge-cu128`（Python 3.10 / torch 2.10+cu128） |
| GGUF 模型 | `/home/leslie/mvp_core/models/llm`（软链到 970 EVO；2026-10-26 当前盘被改作 Ventoy，软链断裂，跑实验前需先恢复挂载） |
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
