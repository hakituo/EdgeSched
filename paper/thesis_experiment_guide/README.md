# EdgeSched 论文与实验指南索引

**用途**：把研究设计、实验协议和当前状态拆成可按任务选择读取的模块，避免 AI 或维护者每次加载整份长文档。

## 首读顺序

新会话默认只读：

1. 本索引；
2. [当前门禁与任务状态](10_gates_and_current_status.md)；
3. 与当前任务直接相关的一个模块；
4. 需要执行实验时，再读 `docs/ELASTIC_RESIDENCY_EXPERIMENT_EXECUTION_PLAN_20260828.md` 和 `docs/ELASTIC_RESIDENCY_RESULT_CATALOG_20260830.md`。

不要为了回答单个实验问题一次性加载全部模块。

## 模块地图

| 模块 | 何时读取 | 内容 |
|---|---|---|
| [01 研究定位、工作流与任务抽象](01_research_scope_and_workloads.md) | 改研究问题、术语或贡献边界 | 目标场景、语音/图像工作流、设备亲和性、驻留与 GPU 计算份额能力 |
| [02 系统状态与运行模式](02_state_and_runtime_modes.md) | 改状态机或决策动作 | 调度状态、候选驻留档位、S0/S1/S2 |
| [03 画像与资源约束](03_profiles_and_constraints.md) | 改 profiler、容量公式或回退约束 | LLM/TTS/Diffusion 画像、显存/CPU/SM 约束、分层目标 |
| [04 代价与规划契约](04_cost_and_plan_contract.md) | 改代价模型、转换算法、预测器或 PlacementPlan | 队列视界、转换摊销、画像插值与在线残差、迟滞、泛化边界、规划输出 |
| [05 CPU 与 GPU 分配方法](05_cpu_and_gpu_allocation.md) | 做核数或 SM 实验 | CPU 扩展曲线、最低核心预算、拓扑消融、SM/流水线平衡 |
| [06 实现差距与路线](06_implementation_and_roadmap.md) | 判断代码是否已实现 | 当前实现审计、P0–P5 路线、SM-A 能力探针 |
| [07 实验阶段与矩阵](07_experiment_stages_and_matrix.md) | 设计或审查正式实验 | 探索/冻结/正式阶段、E0–E9 |
| [08 指标与复现](08_metrics_and_reproducibility.md) | 改统计、输入或证据链 | 统一指标、轮数、顺序、固定输入、manifest |
| [09 证据与论文组织](09_evidence_and_paper_plan.md) | 写论文或决定数据能否使用 | 当前数据边界、必须重跑项、章节与图表规划、局限性 |
| [10 门禁与当前状态](10_gates_and_current_status.md) | 每个新会话必读 | Go/No-Go、完成项、暂停项、下一批实验、完成标准 |

## 当前真值摘要（2026-08-31）

- C2+C3 direct NVML 已完成 16 个 LLM 驻留档位，共 240 条性能记录。
- KV 容量与语义校准已完成 48/48 条。
- B2 已筛出 T0/T3，direct NVML v2 隔离矩阵已完成 30/30 技术核验。
- Forge b=1 direct NVML v1 已完成 5/5，当前 `M_D=5484 MiB`。
- 完整 16 档离线筛选已生成 `c3_h_direct_nvml_20260831_v1.json`；colocated 残差仍 pending，筛选值尚未冻结。
- LLM+TTS 的 C4096/K20+T0 正式 artifact smoke 已 1/1 技术通过；6 候选×5 轮正式共置矩阵和重复运行 colocated 残差尚未完成。
- SM-A 在 WSL2 中因 MPS binary 不存在暂定 NO_GO；待原生 Ubuntu 恢复验证，不伪造 SM-B/C 曲线。
- CPU 物理核拓扑、affinity 与租约基础设施已完成；正式 `auxiliary_core_budget.py` 和核数矩阵尚未实现。
- `paper/experiment_data/01_paper_main/` 当前没有已晋升的新版正文数据。

## 真源与更新规则

发生冲突时按以下顺序判断：

1. 当前源码、测试与真实运行日志；
2. 实验 `manifest.json`、`iterations.jsonl`、`summary.json`；
3. `docs/ELASTIC_RESIDENCY_RESULT_CATALOG_20260830.md`；
4. `docs/ELASTIC_RESIDENCY_EXPERIMENT_EXECUTION_PLAN_20260828.md`；
5. 本目录中的研究设计与状态摘要；
6. `paper/archive/` 和 `docs/archive/` 仅供追溯。

运行结果变化时，优先更新第 10 模块和结果目录；研究模型变化时，更新对应主题模块。不要把短期 Git HEAD、临时测试计数或尚未产生的 run id 固化为长期研究结论。
