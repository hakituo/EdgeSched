# EdgeSched 更新记录

本文件是 `legacy/mvp_core`（EdgeSched 独立仓库）的**详细**更新记录，与父仓库 `d:\AI\xiaoyou-core\UPDATES.md` 无关，两者不要互写。

## 记录分工

| 载体 | 写什么 | 不写什么 |
| --- | --- | --- |
| `docs/UPDATES.md`（本文件） | 背景、根因、方案、影响范围、验证方式 | — |
| git commit 主题行 | 一句话：**这次改动做了什么** | 文件清单（`git show --stat` 可查）、时间戳（`git log` 自带） |
| commit 正文（脚本 `-Detail`） | 只有需要随代码留存的过程信息时才写 | 大段复盘（写本文件） |

## 条目格式

```
## 2026-09-17 S0 v2 启动策略小对照（加速后端，第一批真实数据）

- 批次：`s0_startup_20260917_v2`，K24:T0，首句即播 vs 全量预缓冲各 1 轮，带播放，
  跑在加速隔离环境，`--tts-backend cuda_graph_static`，协议 `s0_protocol_v2`，两条全通过。
- 结果：首句波形就绪 4.260 / 4.243 s，末句 24.993 / 25.296 s，逐句 RTF 0.359 / 0.366，
  窗口内供给缺口占比 **0.0 / 0.0**（应用层缺口段数 0），启动期缺音 4.14 / 1.41 s，
  首播等待 proxy 4.28 / 26.65 s（**预缓冲多等 22.4 s**）；swap 0、输出契约通过。
- 与旧后端 v1（`s0_startup_20260917_v1`：首句 12.27 s、RTF 1.88、窗口内缺口占比 0.423、
  8 段缺口）对比：换后端后**首句就绪快约 3 倍、RTF 从 >1 降到 0.36、句间供给缺口归零**。
- 判读：预缓冲不再有「掩盖断流」的价值——窗口内缺口本来就是 0，代价只是首播多等 22.4 s。
  正式比较继续冻结为首句即播。设备层仍 pending（时钟映射未通过，proxy 不是实际播放时刻）。
- 下一步：两配置（K24:T0 vs K0:T0）× 5 轮正式成对比较；本批 provenance `dirty=true`，
  晋升论文数据前需提交后重跑。

## 2026-09-17 加速隔离环境扶正为主环境（软链换名）

- 背景：S0 要同进程共置 LLM 与加速 TTS，而加速后端依赖 transformers 5.x，装不进旧主环境
  （py3.10 / torch 2.10 / transformers 4.57.3），于是出现「一个实验跨两个环境」的割裂。
- 方案：不做迁移、不重建环境，直接换名 + 软链——
  `mv /home/leslie/.venvs/mvp_core-cu128 /home/leslie/.venvs/mvp_core-cu128_legacy`
  `ln -s /home/leslie/qwen3tts-cugraph-probe/faster-qwen3-tts/.venv /home/leslie/.venvs/mvp_core-cu128`
  这样脚本里的主环境路径一个字都不用改，以后换环境只改软链。
- 可行性依据：venv 的 `bin/python` 是指向基解释器的绝对软链，`pyvenv.cfg` 记的是 base 路径，
  目录改名不影响；我们所有脚本直接调 `<venv>/bin/python`，不走 `activate`；且软链（而非搬移）
  保证加速环境内部路径与 shebang 全部保持有效。
- 补装（都在主环境里）：`sounddevice==0.5.5`、`faster-whisper==1.2.1`、
  `llama-cpp-python==0.3.35`（用 `/usr/local/cuda/bin/nvcc` 编，坑是环境里
  `CC=x86_64-linux-gnu-gcc` 会让 CMake 找不到宿主编译器，须显式 `CC=/usr/bin/gcc CXX=/usr/bin/g++`）；
  BGE 不用装（只需 `onnxruntime` + `tokenizers`，已有）。
- 验证：主环境 `python -V`=3.12.3、torch 2.11.0+cu128、transformers 5.15.1、
  `llama_supports_gpu_offload()=True`、`s0_child`/`check_tts_output_correctness`/`auxiliary_bge`
  均可导入；遗留环境 3.10.19 与 `llama_cpp` 正常，`bin/pip` 的 shebang 已改指遗留解释器。
- **同日删除遗留环境** `mvp_core-cu128_legacy`（释放约 9G），它是唯一装了上游 `qwen_tts`
  的环境。代码侧**不删可复用实现**（`--tts-backend upstream_dynamic` 开关、旧 B2 矩阵
  生成器、`services/tts/qwen3_tts_worker.py`），但去掉了指向已删环境的硬编码：S0 启动器在
  `upstream_dynamic` 下强制要求 `-Python`，两个 `small_9b_*` 旧后端脚本改为显式注释说明
  需自建环境。仍是 563 通过 + 两个启动器契约通过。

## 2026-09-17 S0 接入加速 TTS 后端（协议 v2 + 环境切换）

- 背景：B2 画像（当前接入后端，T0）实测 20/50/100 字 RTF 0.310–0.325 全部 < 1，而 S0 的
  `s0_pipeline` 仍走上游动态路径（旧后端，旧批次逐句 RTF 全部 > 1），两条证据不能混用。
- 方案：`s0_child`/`s0_parent` 新增 `--tts-backend`，默认 `cuda_graph_static`；加速路径
  复用 B2 的 `cuda_graph_synth` 闭包，只覆盖 `SentenceSynthesizer._generate`
  （`AcceleratedSentenceSynthesizer`）。协议升到 `s0_protocol_v2`。
- 影响范围：加速后端只暴露整模型 `device`，没有 talker/codec/speaker 分部件放置接口，
  因此 TTS 放置只剩 `T0`，正式配置由 4 档降为 2 档（`K24:T0`、`K0:T0`，5 轮共 10 次）；
  `K24:T3`/`K24:T4` 显式排除，不沿用旧后端数值。
- 环境：加速后端依赖 transformers 5.x，装不进 `mvp_core-cu128`；S0 的测量前提是 LLM 与
  TTS 同进程共置，故 S0 改跑隔离环境 `faster-qwen3-tts/.venv`，并补装
  `sounddevice==0.5.5` 与 CUDA 编译的 `llama-cpp-python==0.3.35`（用
  `/usr/local/cuda/bin/nvcc`；注意环境里 `CC=x86_64-linux-gnu-gcc` 会让 CMake 找不到
  宿主编译器，必须显式 `CC=/usr/bin/gcc CXX=/usr/bin/g++`）。
- 验证：隔离环境 `llama_supports_gpu_offload()=True`（RTX 5070 8150 MiB）、
  `sounddevice 0.5.5`、`faster_qwen3_tts` 可导入；Windows 侧 `tests/elastic_residency`
  507 通过、5 项 GPU 门控跳过；S0 启动器契约测试通过（含两配置与后端/解释器选择）。

## YYYY-MM-DD 标题

- 背景：为什么改
- 改动：改了哪些模块/口径，关键决策是什么
- 验证：跑了什么测试、得到什么结果（Mock 与真实 GPU/Forge 必须区分）
- 备注：遗留问题或后续动作
```

倒序追加，新条目放最上面。

## 2026-09-17 TTS 加速后端同环境对照：逐句 RTF 从 1.7 降到 0.31

- 背景：S0 正式比较在**旧后端**下的结论是「四档逐句 RTF 全部 > 1，端到端无法连续供音」，B2 隔离矩阵也显示 TTS 独占时 RTF 仍 1.60–2.92。要回答「是否必须换模型」，先要回答「同一模型换推理路径能不能到 1 以内」。
- 改动：
  - 新增 `experiments/tts/qwen3_tts_backends.py`（两条后端加载：上游动态 / CUDA Graph 静态，重依赖全部函数内惰性导入）与 `experiments/tts/profile_qwen3_tts_cudagraph.py`（同进程先后测量、纯函数汇总、manifest 记模型树哈希/参考音频哈希/git provenance），两者均 ≤300 行。
  - 新增启动器 `scripts/experiments/run_tts_backend_compare.ps1`（`-Mode both|baseline|graph`、`-Attn eager|sdpa`、`-Rounds`、`-RunId`、`-PrintOnly`，并落盘 provenance），配套静态测试 `tests/tts/test_tts_backend_compare_launcher.ps1` 与 `tests/tts/test_tts_cudagraph_probe.py`（含"重依赖必须惰性导入"守卫）。
  - 加速侧使用仓库外的隔离环境 `/home/leslie/qwen3tts-cugraph-probe/`，**不触碰 `mvp_core-cu128`**。
- 验证：
  - **同环境 A/B**（`tts_backend_both_eager_20260917_072516`，隔离环境、同进程先后测量、eager、每档 3 轮）：

    | 文本 | 上游动态 RTF（min–max） | 加速后端 RTF（min–max） | RTF 比值 | 合成耗时比 | 音频时长（上游/加速） |
    |---|---:|---:|---:|---:|---|
    | 20 字 | 1.7754（1.6215–1.8833） | **0.3168**（0.3165–0.3174） | 5.6× | 5.2×（8.38→1.60 s） | 4.72 / 5.04 s |
    | 50 字 | 1.7257（1.6534–1.7840） | **0.3067**（0.3058–0.3073） | 5.6× | 5.6×（18.36→3.29 s） | 10.64 / 10.72 s |
    | 100 字 | 1.7622（1.6612–1.8345） | **0.3032**（0.3030–0.3034） | 5.8× | 5.6×（33.27→5.93 s） | 18.88 / 19.55 s |

    一次性成本：上游 权重加载 3.8 s + prompt 构建 1.1 s；加速后端 权重加载 1.3 s + 图捕获 2.5 s。
    本次改动**同时包含静态 KV cache 与 CUDA Graph 捕获**，因此只能称「加速后端相对原后端的收益」，
    不把全部收益单独归因于 CUDA Graph；两条路径音频时长不同，故 RTF 比值与耗时比值分别列出。
  - **跨环境批次作废、产物已删除**：`cudagraph_only_20260917_v2` 报 9.5×/7.2×/8.1×（原环境基线 3.11/2.26/2.58 vs 隔离环境 0.327/0.315/0.320）是跨环境比出来的，已连同中断的探针批次一并清理，只保留本条目上表的同环境 A/B 产物。同环境下上游 eager 为 1.72–1.78，与 B2 的 sdpa 基线（1.60–1.92）基本一致，说明**注意力实现影响很小，差异来自环境/包版本**（原环境 python3.10 + 上游 `qwen-tts 0.1.1` + transformers 4.57.3）。此前「eager 让上游慢 40–60%」的推断已更正。
  - Windows 侧 `tests/tts` + `tests/elastic_residency` 524 通过、6 项 GPU 门控跳过；启动器静态测试 PASS。
- 备注（边界，必须随数字一起报告）：
  - 加速路径依赖 `qwen-tts-hf 0.1.1.post1` + `transformers>=5.15.1`，**不能装进冻结环境**：原环境能 import，但图捕获时报 `StaticLayer.lazy_initialization() takes 2 positional arguments but 3 were given`。此外官方默认 torchaudio wheel 是 CUDA 13 构建（缺 `libcudart.so.13`），须强制装 cu128 版；transformers 5.17 会让 `MimiConfig` 缺 `rope_theta` 直接崩，须钉 5.15.1。
  - **输出不确定性（同环境实测）**：上游动态路径 3/3 轮 PCM SHA 完全一致、样本数恒为 113280 / 255360 / 453120；加速后端 3/3 轮 SHA 全不同，样本数 111360/122880/128640、249600/261120、443520/472320/491520。相对上游同一文本，样本数最大偏差 **+13.6%（20 字）**、最小 −2.3%。**这只证明「不位精确」，既不等于质量不合格，也不等于统计结果不可复现**；长度差异需另行核查（重复同句、交替长短文本、检查状态重置是否完整），且必须同时报音频时长、合成耗时与文本/听感一致性，不能只看 SHA。协议里「同 seed 输出一致」「跨批次 WAV SHA 一致」两条校验在加速后端下需要改口径。
  - 尚未接进 `s0_pipeline`：接入需替换 `SentenceSynthesizer` 后端、把 `s0_protocol_v1` 升到 v2 并重跑 S0，属冻结口径变更，另行评估。
  - 论文含义（**待验证预测，不是结论**）：加速后端尚未接入流水线。隔离实验的 RTF 不能直接乘以旧批次音频时长，再与旧 LLM 交付窗口拼成新的端到端结论；而且 token 交付窗口本身受有界队列背压影响（`K24:T4` 在 GPU 却把交付拖到 ~56 s），不是纯 LLM 执行速度。准确写法是：**加速后上游供给可能成为新的限制因素，需要通过真实共跑事件验证**。

## 2026-09-17 S0 四配置正式成对比较：判为收缩主张

- 背景：协议冻结与启动策略对照完成后，需要跑四配置成对比较，回答「混合是否真的改善、是否只是协变量、哪些值得进动态切换」。
- 改动：
  - 真实批次 `s0_formal_20260917_v1`（`HEAD=ef563e8`、工作区脏、`-SkipGit` 同步）：四配置 × 5 轮 = 20 次，`--startup-policies first_sentence --enable-playback --continue-on-failure`，真实后端无 Mock fallback。
  - 修正两处判定缺陷（离线复算时发现，均补测试锁定）：`answer_questions` 原先用 `abs(delta)` 判改善，会把「变慢」当成收益；且三个主指标共用一个基线分母（首句就绪）导致缺口段数的改善被漏判。现改为「只把超过相对容差的负差算作改善」+「每个指标各用自己的基线值」。
  - 新增候选混入标记 `confound_flags` 与 `context_deltas`（句数、缺口占比、输出 token、音频时长），候选必须同时展示协变量。
- 验证：20 次中 19 通过；`K24:T4` r2 唯一失败门为 swap 增长 63.1 MiB（容差 4 MiB），r3–r5 通过，失败计入分母。聚合（n=5）：`K24:T0` 首句 12.27 s / RTF 1.88 / 缺口占比 0.423；`K24:T3` 14.42 / 2.39 / 0.542；`K0:T0` 18.66 / 2.04 / 0.543；`K24:T4` 36.66 / 8.04 / 0.861。相对 `K24:T0` 的配对差全部为正（更差）：`K24:T3` +2.15 s/+0.52、`K0:T0` +6.39 s/+0.16、`K24:T4` +24.39 s/+6.17。唯一表层改善是 `K0:T0` 缺口段数 7 vs 8，但其 `confound_flags` 命中文本不同 + 句数 8 vs 9 + token 65 vs 80，且窗口内缺口占比反而更差（0.543 vs 0.423），不构成收益。
- 备注：据此按 Gate B No-Go 第 1 条**收缩主张**——不进入以收益为前提的 `PlacementPlan`/predictor/planner 实现，只报告静态放置结论与后端实时性边界。**结论范围必须收紧**：(1) 本批四档**都是部分驻留**（`K24` = 32 层里只上卡 24 层，`K0` = 全 CPU），因此**不能用本表反证「完整共驻留（K32）可行」或「显存约束不存在」**；(2) 能明确写的只有「在这组旧后端实验中，`K24:T3`/`K24:T4` 相对 `K24:T0` 未改善所报主指标；`K24:T0` 相对 `K0:T0` 的收益判定受输出差异影响（65 vs 80 token、8 vs 9 句），**尚未完成公平归因**」——不能把「收益未确证」写成「混合驻留已被否定」；(3) 四档逐句 RTF 全部 > 1（与 B2 隔离矩阵 T0 中位数 1.81、0/15 低于 1 一致）。设备体验指标仍全部 pending。Windows 侧全量测试 500 通过、5 项 GPU 门控跳过。

## 2026-09-17 S0 测量范围分层、协议冻结与启动策略小对照

- 背景：S0 播放链路长期卡在设备时钟映射（实际首播时刻不可信），Gate B 却把「实际播放截止与播放连续性」当成 Go 判据，容易陷入「无限修播放器」或「用 callback 代理冒充实际播放」；同时 07 模块参数表一直停在「待确认」，正式比较无法开始。
- 改动：
  - 测量范围分层：新增 `s0_pipeline/measurement_scope.py`，把一次运行拆成**调度主指标**（首句/末句完整波形就绪、句级排队与合成、应用层供给缺口、资源与输出契约）与**设备体验指标**（可信实际首播、设备 underflow）。时钟未验证时设备层一律 `pending_device_validation`，只保留 callback 代理并写明语义；Gate B 形式判据改用调度层，`device_metrics_block_go=False`，避免把「测不出」当成「已通过」或「已失败」。
  - 协议冻结：`config.py` 新增 `S0_PROTOCOL_VERSION=s0_protocol_v1`、`FROZEN_PROTOCOL`（输入与生成、分句、启动策略、重复次数、输出差异、比较规则、失败处理）与启动策略常量，`FREEZE_STATUS` 从 pending 改为 frozen / pending_device_validation。
  - 启动策略：新增 `startup.py`（`StartupPlan`），支持首句即播与全量预缓冲，两者共用同一播放路径与断流统计；`runner.py` 按策略决定开播时机，`s0_child` 不再提前开流；`s0_parent` 新增 `--startup-policies`。
  - 成对编排与复算：`s0_parent` 支持 placement × 启动策略轮转排序、输出指纹与测量范围落盘；新增 `s0_evidence.py`、`s0_matrix_metrics.py`、`s0_matrix_report.py`、`analyze_s0_matrix.py`（硬校验协议版本/测量范围版本/冻结 placement，按 `run_index` 配对，回答三个冻结问题）与启动器 `scripts/experiments/run_s0_pipeline.ps1`。
  - 文档：07 模块补「测量范围分层」与冻结参数表 + 启动策略对照记录；10 模块 Gate B 改写为调度层判据 + 设备层 pending，并更新任务项；`PROJECT_INDEX.md` 同步入口与验证。
- 验证：Windows 侧 `tests/elastic_residency` 全量 497 通过、5 项 GPU 门控跳过（新增 24 项）。WSL2 真实小对照 `s0_startup_20260917_v1`（K24:T0，各 1 轮，`--enable-playback`，真实后端无 fallback，`HEAD=ef563e8`、工作区脏）：首句即播 12.61 s 首句就绪 / 8 段供给缺口 / 窗口内占比 44.08%；全量预缓冲 12.76 s / 0 段 / 0%，首播等待 12.63 s → 68.28 s（proxy）；两者输出文本 SHA-256、80 token、9 句、32.88 s 音频完全一致。
- 备注：设备时钟映射仍未通过验证，实际首播与设备 underflow 保持 pending；四配置正式成对比较另记。本次同步用 `-SkipGit`，源码在工作区尚未提交。

## 2026-09-13 把 S0 播放测量口径修正与 2026-09-12 证据写回论文指南

- 背景：S0 播放链路在 2026-09-12 做过口径修正与一次启用播放的 K24:T0 真实运行，代码侧证据已存在，但论文指南（07/10/README）与索引仍停留在旧口径与旧数字。
- 改动：
  - 07 模块：修正播放统计三处口径（缺供给 callback 次数 vs 独立断流段、启动/句间/句内/收尾/未知分阶段、生产者下标 vs 设备下标）；时钟映射判据改为"单位一致 + 配对残差 + 漂移（跨度≥1 s 才判定）+ 单调"，不再按偏移量大小判定。
  - 07 模块：K24:T0 结果改写为 10 段缺音（startup 14.7 s / 句间 8 段 27.8 s / tail 0.08 s）、窗口内占比 45.8%，只写"本次运行未实现连续供给"，不再写"性能 SLO 未通过"；v3 产物目录补四个未跟踪新模块的内容副本与重建说明。
  - 10 模块：播放口径修正 + smoke + K24:T0 合并为一条已完成项；剩余缺口改写为时钟映射验证、启动策略对照（须报首播代价）、断流到句归属。
  - 指南 README 与 `docs/PROJECT_INDEX.md`：同步上述口径，并更新 `s0_pipeline` 子包拆分现状（`playback_buffer`/`playback_clock`/`playback_starvation`，全部 ≤300 行）。
- 验证：仅文档与口径同步，未改动代码，未重跑实验；Windows/WSL2 两端时钟映射均**未通过**验证，实际播放时刻保持 `proxy`。
- 备注：本次提交信息只写了一句话摘要，细节补记在本条目；后续同类改动应先写本文件再提交。

## 2026-09-13 同步脚本自动提交必须写清"做了什么"

- 背景：`scripts/sync_to_wsl2_safe.ps1` 同步后固定提交 `auto: 同步后自动提交（时间戳）`，历史 commit 无法回答"这次改了什么、为什么改"；本仓库也没有 UPDATES.md 承载详细记录，追溯成本很高。
- 改动：
  - 脚本新增 `-Message`（必填）与 `-Detail`（可选）参数，提交主题行直接使用 `-Message`，`-Detail` 作为 commit 正文。
  - 未给 `-Message`、或 `-Message` 命中无信息黑名单（auto/update/更新/自动提交/同步 等）时直接 `exit 2`，且在 rsync 之前校验，避免白跑一次同步。
  - commit 改用 UTF-8 无 BOM 临时文件 + `git commit -F`，避免 Windows 控制台编码导致中文摘要乱码。
  - 提交前回显本次变更清单（最多 20 项），便于发现混入的无关文件。
  - 新增 `docs/UPDATES.md` 承载详细记录。
- 验证：`tests/scripts/verify_wsl_sync_policy.py` 增加断言，锁定 `-Message` 必填校验与黑名单存在，防止回退成无信息提交。
- 备注：工作区存在其他会话/其他 agent 的未提交改动时，仍应先用 `-SkipGit` 同步，再按需单独提交，避免把无关改动一起推上去。
