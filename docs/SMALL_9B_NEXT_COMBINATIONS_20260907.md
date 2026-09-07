# small 9B 后续组合与离线筛选（2026-09-07）

## 用户手动命令

```powershell
& 'D:\AI\xiaoyou-core\legacy\mvp_core\scripts\experiments\run_small_9b_screen.ps1'
```

Windows venv_core 离线计算，不启动模型、WSL、Forge 或 GPU。产物为
`experiment_results/elastic_residency/c3_analysis/small_stage_screen_<timestamp>.json`，不覆盖已有输出。
入口 `experiments/elastic_residency/screen_small_9b.py` 复用既有 `compute_h`、`compute_g_safe`、
TTS/Forge 提取和 C2/C3 合并校验，不改原始画像。

输入是 small C2/C3 `20260907_143917`、small 视觉 Profile `20260907_171635`、
B2 T0/T3 `20260831_direct_nvml_v2` 与 Forge `20260906_direct_nvml_v1`。
C2/C3/视觉核对固定模型身份、来源快照和原始 child；TTS/Forge 作为通过门禁的历史独立画像先验。
显存使用完整 child 生命周期峰值减基线；视觉 FULL 联合峰值已经含 LLM，不能再加一份 LLM 或 KV。
G_safe 暂取既有算法各已知分量最大值，并纳入当前观察到的 915.015625 MiB 基线；
共置正残差、主机内存重叠、CPU 核心分配和实际兼容性仍待验证，不冻结阈值。

实现时在 Windows 对现有产物完成离线验证（`.test_tmp/small_screen_review.json`）：
三个上下文 T0 对应最高测得 GPU 容量候选 K20，T3 对应 K30，FULL 均不满足该筛选预算。
长上下文候选余量很小，不能按“筛选通过”直接当成安全配置。
FULL/C512 视觉联合画像叠加 GPU T0/T3 的估计余量分别为 −2706.15 / −1430.21 MiB。
这个否定只限已测 FULL 视觉状态，不外推为所有混合档、所有 CPU 放置均不可行。

## 用户筛选验收与首轮兼容性命令（20:45 批次）

用户产物：`c3_analysis/small_stage_screen_20260907_204516_653987.json`，
SHA-256 `7fa1b5ae48371d7bddcf7ac8e0138324e2c454d85bd4760bfa2f859284caaf41`。
Windows 离线重新核验全部输入并复算，JSON 与用户产物完全一致；没有启动真实模型。

| context | T0 最高候选 / 扣除 G_safe 后余量 MiB | T3 最高候选 / 扣除 G_safe 后余量 MiB |
|---|---:|---:|
| 512 | K20 / 109.85 | K30 / 123.79 |
| 2048 | K20 / 59.85 | K30 / 73.79 |
| 4096 | K20 / 15.85 | K30 / 9.79 |

G_safe=915.015625 MiB 仍为 provisional；上述是已测档位中的 GPU 筛选边界，
不含未测共置正残差，不能冻结 H_TTS 或证明主机内存可行。FULL 叠加 TTS 未满足此预算。
FULL/C512 的视觉+TTS 估计缺口 T0=2706.15 MiB、T3=1430.21 MiB；混合视觉档待测。

下一步由用户执行：

```powershell
& 'D:\AI\xiaoyou-core\legacy\mvp_core\scripts\experiments\run_small_9b_tts_compat.ps1'
```

启动器复用 `llm_tts_colocated.py`，固定 small SHA 和本次 screen SHA，增量同步代码、保存源码快照，
按顺序运行 `512:K18:T0`、`512:K18:T3` 各 1 轮，失败即停，输出独立
`llm_tts_colocated/small_9b_tts_compat_<timestamp>/`，不覆盖旧记录。
14 个工作物理核（使用 SMT）+2 个系统核、seed=42、n_ctx=512+128=640；
TTS 默认同一句短文本、最多 256 个生成 token。沿用隔离 T0/T3 增量作为残差先验，保存实际模型树和参考音频哈希，结果复核时须对齐 B2 身份。

两个配置统一从 K18 起步，筛选余量分别约 355.85 / 1631.79 MiB。
代码当前先在 GPU 加载完整 TTS，再将 T3 codec 移到 CPU，所以 T3 推理期候选 K30
不自动代表加载期安全；不直接安排边界 K30 或更高混合档。实际加载与完整时间线仍须复核。

本端点先将 LLM 填充到 640 个 KV 位置并保持其驻留，再生成非流式 TTS/WAV，最后卸载并归还租约。
没有 LLM 同时 decode，也没有 TTS 后 LLM 继续回答验证；不代表 SM 平衡、LLM→TTS 端到端流水线或持续会话可用性。
T0 为 GPU TTS，T3 为 GPU talker + CPU codec；不代表完整 TTS_CPU。
先验和单轮 smoke 不能替代正式重复或两个反向静态放置基线；本批不加载 V、Forge，不自动继续三/四组件实验。

验证：新增 PowerShell 测试从非项目目录以 `-PrintOnly` 检查命令，禁止调用 WSL，覆盖 small/screen 身份、
两单元矩阵、LF 换行和 fail-fast；筛选、候选矩阵、KV/音频/容量及离线共置分析共 33 项 Python 测试通过。
pytest 临时目录显式落在本仓库 `.test_tmp`，避免父仓库缓存和系统临时目录的沙箱权限限制。

## 组合按两条任务路径展开

| 阶段 | 组合 | 先回答的问题 |
|---|---|---|
| 基础语音路径 | LLM + TTS | LLM_GPU/TTS_CPU、LLM_CPU/TTS_GPU、混合 LLM/TTS_GPU 的容量、主机内存和端到端可用性 |
| 视觉理解路径 | LLM + TTS + V | V 编码时 TTS 的设备/队列安排；V 释放后恢复语音路径，或在可行配置下保留 V |
| 图像生成路径 | LLM + TTS + Diffusion | 生成期间 LLM/TTS 如何降级、排队及恢复，GPU 与主机内存门同时满足 |
| 条件性扩展 | LLM + TTS + V + Diffusion | 仅在已到达的实际任务需要重叠且容量筛选允许时尝试，不作为必须跑完的最后一级 |

V 是 VL 前端组件，编码后仍依赖 LLM 回答；Diffusion 属于另一条生成路径。
三个/四个组件参与同一工作流，不等于全部同时驻留 GPU，更不等于全部同时提交 kernel。
比较要分别记录组件驻留、计算重叠及 CPU/GPU 放置。

当前 GPU 共存先测试兼容性，速度保留为未管理计算份额的基线；正式 SM 平衡在原生 Ubuntu 能力门通过后开展。
TTS 内部非流式的限制不会由 SM 分配自动消失。CPU TTS 成对端点和共享物理核预算仍需独立安排，不能从 T0/T3 GPU 数据代替推断。

Diffusion 的 16G 主机内存负面边界继续保留。筛选脚本仅给 GPU 数值诊断，不生成可执行的 Diffusion 重跑计划，
也不会因换了 small 文件就把旧失败门解除；四组件状态先筛选，不强行测明显放不下的组合。

## 大版模型删除与版本管理

用户明确要求删除后，已核对并只删除 WSL 文件
`/home/leslie/mvp_core/models/llm/Qwen3.5-9B-q4_k_m.gguf`，大小 5,974,828,256 B（约 5.56 GiB），
删除前完整 SHA 为 `423fb3e5fe2c98df616159b7cd1bd04765f67667f5f8712a81e8f63f4e7ee1ef`。
删除后确认 small `comparison_1a35ab89/model.gguf`（5,627,044,704 B）和
`Qwen3.5-9B-f16.mmproj`（918,165,888 B）仍存在且大小不变；没有清理目录或删除实验结果。
这是释放 WSL 文件系统空间，不等同于运行期内存减少，也不保证宿主 VHDX 文件立即缩小。
历史 large 参数/比较脚本保留用于复现，但现在直接运行 Large 或双版本对比会因文件不存在而停止，需先恢复相同哈希文件。

GitHub 目标已通过仓库 API 确认为 `hakituo/EdgeSched` 私有仓库。
本轮整理此前未提交的 CPU 画像、9B 对比、视觉/完整曲线记录、论文生命周期修订及新筛选入口，
不纳入模型、原始实验产物或临时测试输出。Git 提交不改变旧实验 manifest 的源提交与快照身份。

## 验证

`tests/elastic_residency`：400 passed、5 skipped（明确要求 WSL/CUDA/MPS 的真实测试未启用）。
新增筛选测试覆盖联合视觉不重复计费、保留 Diffusion 主机门、观察基线及 Forge 来源字段兼容。
所有修改的 Python 文件 Ruff 通过且不超过 300 行；PowerShell 视觉与完整曲线入口静态测试通过。
源码范围扫描未发现模型/二进制产物或所检查的凭据模式。没有运行新模型实验。
