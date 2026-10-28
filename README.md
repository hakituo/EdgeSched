# EdgeSched（MVP Core）

EdgeSched 是 **Autarkia**（原 Xiaoyou Core）的本地多模态调度研究原型，目录名沿用 `mvp_core`。项目围绕单 GPU / 有限显存场景中的 LLM 常驻、图像突发、CPU backfill、迁移代价、准入控制和崩溃恢复展开，实验结论要写进论文；目前仍是研究原型，论文进行中。

**2026-10-28 当前状态**：主开发环境为本机原生 Linux `/home/leslie/mvp_core`，编辑、运行和 git 提交都在此处。论文主线在 Gate B 收缩后转向**余量感知的驻留调度**（时间维度），见 [指南 11](paper/thesis_experiment_guide/11_slack_aware_scheduling_direction.md)。V0 共跑诊断已完成（`colocation_v0_20261026_v1|v2`，28/28 通过）：S0 共跑减速主因是 llama.cpp 的 28 线程超额订阅 CPU，降到 14 线程后共跑速度恢复到隔离画像的 60%，V0 放行；S0 v2 仍需用 14 线程重跑（V0.5）后才能重新判读余量。Gate B 仍不登记 Go，`paper/experiment_data/01_paper_main/` 尚未晋升数据。SM-A 已在原生 Linux 转 **GO**（`sm_a_20261013_native_linux_v6`，实测有效 SM 36/26/18/8），SM-B 待排。环境方面：9B 权重已放到本地 `models/llm_store/`；4B 与 Forge 额外权重仍依赖未挂载的 970 EVO；真实播放暂缺 PortAudio/SoX。

9B 选型、small 曲线与视觉画像的既有证据分别见 [选型与验收](docs/NINE_B_MODEL_COMPARISON.md)、[16 档曲线与 KV 验收](docs/SMALL_9B_CURVES_20260907.md)、[视觉阶段画像](docs/SMALL_9B_VISION_PROFILE_20260907.md)。目录整理记录见 [文件夹归类地图](docs/FOLDER_MAP.md)。

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
- `experiments/elastic_residency/`：当前论文主线实验（S0、V0 共跑诊断、SM、C2/C3、共置与视觉）。
- `experiments/run_all_experiments.py`：旧调度器论文实验的统一编排。
- `services/tts/`：独立句子级流式 Qwen3-TTS Base 子系统。
- `paper/thesis_experiment_guide/`：按任务拆分的新版论文与实验路线真源；当前主线见第 11 模块，旧路径仅保留兼容导航。
- `paper/rewrite/`：新版正文工作区；旧英文稿已归档。

当前目录分层：

| 路径 | 当前语义 |
|---|---|
| `services/`, `domain/`, `data/adapters/` | Python 控制面和运行适配器 |
| `cpp_scheduler/` | C++ 执行面；`build/` 是本地生成目录，不是源码 |
| `experiments/elastic_residency/` | 当前论文主线实验入口 |
| `experiments/` | 其余实验入口（旧调度器论文链仍保留） |
| `experiment_results/` | 工作产物；旧论文结果集中在 `_archive/` |
| `models/llm_store/` | 本地 9B GGUF 与 mmproj；`models/llm/` 中有软链 |
| `paper/rewrite/` | 新版论文正文工作区 |
| `paper/experiment_data/` | 新版论文证据入口，当前尚未晋升数据 |
| `paper/archive/` | 重写前论文链条和更早历史材料 |
| `docs/archive/` | 被当前项目上下文和论文总指南取代的历史文档 |

## 运行环境

**2026-09-25 起主开发环境就是本机原生 Linux（Ubuntu 24.04.5 LTS）**，编辑、运行、git 提交都在此处，不再使用 WSL2 双副本流程：

| 用途 | 路径 |
|---|---|
| 代码与 git 仓库 | `/home/leslie/mvp_core`（`.git` 在工作树内，origin = `https://github.com/hakituo/EdgeSched`） |
| 实验环境载入 | `source scripts/experiments/edgesched_env.sh`（设置 `$EDGESCHED_PYTHON`、CUDA 12 库路径与 TTS 权重路径） |
| 主虚拟环境 | `/home/leslie/Autarkia/.venv`（与父项目 Autarkia 共享，Python 3.13.15 / torch 2.11.0+cu128 / transformers 4.57.3 / numpy 2.5.2；`/home/leslie/.venvs/mvp_core-cu128` 为指向它的软链） |
| Forge 虚拟环境 | `/home/leslie/.venvs/forge-cu128`（Python 3.10 / torch 2.10+cu128） |
| 9B GGUF / mmproj | `models/llm_store/`（本地副本）；`models/llm/` 下的 9B 软链指向这里 |
| 4B GGUF / mmproj | `models/llm/` 软链到 `/media/leslie/970EVO1/xiaoyou-models/llm`；当前未挂载，跑 4B 前需恢复 |
| TTS 模型 | `/home/leslie/Autarkia/models/Qwen3-TTS-12Hz-0.6B-Base` |
| 图像模型 | `models/Image` 本地（含 SDXL Turbo）；Forge 其他 checkpoint 软链到 970 EVO，当前断裂 |
| Forge | `/home/leslie/mvp_core/stable-diffusion-webui-forge-main/` |

提交直接在 Linux 完成，不再走 `scripts/sync_to_wsl2_safe.ps1`（该 Windows→WSL2 同步+自动提交流程已废止）：

```bash
git add <本任务相关文件>      # 显式路径，不要 git add -A
git commit -m "一句话说明这次改了什么"
git push origin main
```

提交信息禁止 `auto`/`更新`/`同步` 这类无信息文案；工作区另有其他会话的未完成改动时只暂存本任务文件。详细过程（背景/改动/验证）记入 `docs/UPDATES.md`。文本文件统一 LF（见根目录 `.gitattributes`）。

运行实验前先 `source scripts/experiments/edgesched_env.sh`：它在不修改系统 ld 配置的前提下补 CUDA 12 库路径，并设置 `$EDGESCHED_PYTHON` 与 TTS 权重路径。`requirements-wsl-cu128.txt` 文件名保留历史命名，只用于补装缺失的 Python 依赖；其中 torch 固定组仅适用于独立重建的 Python 3.10 环境（如 Forge），不要用它降级共享 venv。系统层仍需按文件头安装 SoX、PortAudio 等；**当前本机缺 PortAudio/SoX，真实播放和录音链路不可用**，不涉及设备播放的离线实验不受影响。静音链路通过也不等于真实并发 TTS 已满足无 underrun。

## 演示服务

先载入实验环境，再启动：

```bash
source scripts/experiments/edgesched_env.sh
"$EDGESCHED_PYTHON" main.py
```

- WebSocket：`ws://localhost:8000/ws`
- Health：`http://localhost:8000/health`

`config.py` 优先复用父项目 Autarkia 的 `config/integrated_config.py`。独立运行且导入失败时，才读取 `XIAOYOU_BENCH_*`、`XIAOYOU_USE_CPP_*` 等环境变量（变量名沿用历史命名）。模型不存在时演示服务会回退到 Mock，因此“服务启动成功”不是“真实模型路径已验证”。

## 实验

完整说明见 [实验指南](experiments/README.md)。**当前论文主线**在 `experiments/elastic_residency/`：

- 余量感知 V0 共跑诊断：`experiments/elastic_residency/colocation/v0_parent.py`
- S0 端到端流水线：`experiments/elastic_residency/s0_pipeline/s0_parent.py`
- SM-A GPU 计算份额探针：`experiments/elastic_residency/sm/sm_capability.py`
- S0 余量离线复算：`experiments/elastic_residency/analyze_s0_slack.py`
- TTS 加速后端画像：`experiments/tts/profile_qwen3_tts_cudagraph.py`

运行前先 `source scripts/experiments/edgesched_env.sh`；各入口的具体参数以 `--help` 和对应文档为准。

旧调度器实验（旧论文链，仍可运行）：

```bash
# 无 Forge 的实验集合
"$EDGESCHED_PYTHON" experiments/run_no_forge_experiments.py --iterations 5

# 需要 Forge 的迁移与真实 workload
"$EDGESCHED_PYTHON" experiments/run_forge_experiments.py --experiments migration --iterations 5

# 全部论文实验编排
"$EDGESCHED_PYTHON" experiments/run_all_experiments.py --experiments all --iterations 5

# Mock/Real 性能套件
"$EDGESCHED_PYTHON" experiments/run_performance_suite.py --workload mock

# 独立 Qwen3-TTS GPU smoke（默认 ICL，需准确转写）
"$EDGESCHED_PYTHON" experiments/tts/smoke_qwen3_tts_base.py --device cuda:0 --ref-text '准确转写'
```

TTS 的音色画像、x-vector 消融、CPU/GPU smoke 和证据边界见 [Qwen3-TTS Base 独立子系统](docs/TTS_SUBSYSTEM.md)。2026-08-28 的 [GPU 句长画像](docs/TTS_GPU_SENTENCE_LENGTH_PROFILE_20260828.md) 中 45/45 条 RTF 均大于 1 是**旧上游后端**的口径；2026-09-17 起接入的 `cuda_graph_static` 加速后端在 B2 画像中 20/50/100 字 RTF 已降到 0.30–0.33（< 1），S0 v2 也使用该后端。真实播放仍受 PortAudio/SoX 缺失限制。

`experiment_results/` 是运行产物；`paper/experiment_data/01_paper_main/` 是新版正文数据入口，2026-10-26 已清空 5–6 月旧数据，当前仍未晋升任何新数据。新结果必须经过多次迭代、manifest、聚合检查和人工确认，不能自动进入正文。

辅助 CPU 核数画像：`experiments/elastic_residency/auxiliary_core_budget.py` 复用物理核租约与采样器，覆盖 JSON RAG 1/2 核、BGE/Whisper 1/2/4 核与 `R_sys=0/1/2/4` 隔离敏感性。`d1_auxiliary_manual_v1`（R_sys=2）已 40/40 通过有效性复算；BGE 候选缓存的真实补测 `d1_bge_cached_20260906_195348`（WSL2，15/15 有效）中 1/2/4 核 p95 为 30/17/13 ms，均满足 150 ms SLA，候选预计算 mean 为 1.22/0.70/0.52 s。以上都是隔离辅助任务画像，不等于 D3 正式 R_sys 验收，数据未晋升正文；细节与证据边界见 [辅助核数画像](docs/AUXILIARY_CORE_BUDGET.md)。

## 验证

先载入实验环境（直接调用裸 `python` 会落在没有依赖的 uv CPython 上）：

```bash
source scripts/experiments/edgesched_env.sh

# 静态项目索引验证
"$EDGESCHED_PYTHON" tests/scripts/verify_project_context.py

# 不依赖真实模型的高频测试
"$EDGESCHED_PYTHON" -m pytest tests/test_chat_service.py tests/test_phase_aware_quick.py -q -o asyncio_mode=auto
```

C++ 绑定、真实 GGUF 和 Forge 测试在本机 Linux 直接运行（GPU/设备访问需在沙箱外执行），不能用 Mock 结果替代。

## 论文

当前主线是**余量感知的驻留调度**（时间维度），研究问题、已有证据和验证顺序见 [指南 11](paper/thesis_experiment_guide/11_slack_aware_scheduling_direction.md)，模块化指南入口见 [paper/thesis_experiment_guide/README.md](paper/thesis_experiment_guide/README.md)。新正文工作区位于 [paper/rewrite/](paper/rewrite/README.md)，数据边界见 [实验数据说明](paper/experiment_data/README.md)。重写前稿件和数据位于 `paper/archive/pre_rewrite_20260830/`，只用于追溯。

## 版本管理

本目录是独立 git 仓库（origin = `https://github.com/hakituo/EdgeSched`），与父项目 Autarkia 分开管理；`.git` 在原生 Linux 工作树内，提交和推送都在这里完成。父仓库 `.gitignore` 整体忽略 `/legacy/`，所以本目录改动不会出现在父仓库 Git 状态中，这是预期行为。2026-10-27 已删除 `.github/workflows/` 下依赖本机 self-hosted runner 的双端自动同步 workflow；Windows 副本不再权威，不要在那里提交。
