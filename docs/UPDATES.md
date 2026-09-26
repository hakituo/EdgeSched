# EdgeSched 更新记录

本文件是 EdgeSched 独立仓库（`/home/leslie/mvp_core`，origin = `https://github.com/hakituo/EdgeSched`）的**详细**更新记录，与父仓库 `/home/leslie/xiaoyou-core/UPDATES.md` 无关，两者不要互写。

## 记录分工

| 载体 | 写什么 | 不写什么 |
| --- | --- | --- |
| `docs/UPDATES.md`（本文件） | 背景、根因、方案、影响范围、验证方式 | — |
| git commit 主题行 | 一句话：**这次改动做了什么** | 文件清单（`git show --stat` 可查）、时间戳（`git log` 自带） |
| commit 正文 | 只有需要随代码留存的过程信息时才写 | 大段复盘（写本文件） |

## 条目格式

```
## 2026-09-21 归因分析（按长度归一 + 句数对齐）与设备时钟诊断

- 背景：v2 正式比较中混合 `K24:T0` 全面占优，但两配置输出不同（80 token/9 句 vs 65 token/8 句），
  按冻结规则不能直接算干净收益。补两条离线归因（用已有 10 条记录，**不重跑**）：
- **按输出长度归一**（`covariate_normalization`）：末句就绪/token 差距 2.2206× → **2.7331×**、
  末句/句 2.4982×、每 10 s 音频供给缺口 0 vs 2.2672；归一化后基线仍全部更差且**差距未缩小**。
  实现要点：参考值为 0 时（0 段 vs 6 段）比值无定义，按「基线更差」记（这是最强证据）；
  `audio_s_per_token` 是内容密度描述符，不进判定集。
- **句数对齐**（`sentence_aligned_comparison`）：两配置各自 5 轮输出完全确定、多跑不会一致，
  因此对齐到相同**句数**比较——只比前 8 句，基线在每一句波形就绪上都更慢且 5/5 轮一致
  （第 0 句 4.384 vs 10.740 s，第 7 句 21.294 vs 56.050 s，领先 34.8 s）。
- 结论：方向明确且协变量已量化，但**文本内容本身仍不同**，按冻结规则仍不是「已证明的干净归因」，
  故**不登记 Gate B Go**。
- **设备时钟诊断**（新增 `playback_clock_probe`，2026-09-21）：WSL2 只有 ALSA/OSS、输出设备仅
  `pulse`（虚拟）；`currentTime` 对 host 跨度比 1.000036 正常，但 `outputBufferDacTime`
  跨度比 1.2335(24k)/1.1704(44.1k)/1.0900(48k) 且**不单调**，换采样率与延迟档均不过 ±5% 容差
  → **不是字段取错，是虚拟设备没有真实采样时钟**。代码侧改进：验证失败原因现在会区分这两种情况；
  `device_underflow_count` 改为「可观测但不附时刻」（PortAudio 状态位不依赖时钟），
  实际播放时刻仍保持 pending。已作为待办写入第 10 模块，解锁路径为原生 Linux 真机或
  usbipd 直通 USB 音频设备。
- 测试：569 通过、6 跳过。

## 2026-09-26 补齐 EdgeSched 双端（Linux/Windows）自动同步 workflow

- 背景：2026-09-25 起主开发环境切到本机原生 Linux `/home/leslie/mvp_core`，Windows 侧保留
  `D:\AI\xiaoyou-core\legacy\mvp_core` 做双端开发。父仓库 `xiaoyou-core` 早有 `sync-local.yml` /
  `sync-local-linux.yml` 做双端自动同步，但 EdgeSched 是**独立仓库**且此前**没有 `.github/` 目录**，
  完全不在自动同步范围内，两端只能手动 push/pull，Windows 侧因此落后到 `bcb45b0`。
- 改动：新增 `.github/workflows/sync-local-linux.yml`（→ `/home/leslie/mvp_core`）与
  `sync-local.yml`（→ `D:\AI\xiaoyou-core\legacy\mvp_core`），沿用父仓库模板语义：push 到 main 触发，
  各端 self-hosted runner 执行 `fetch` + `merge --ff-only`；**本机 `git status --porcelain` 非空时
  取消同步**（`exit 0`，绝不覆盖本机未提交工作）；分叉时不自动合并、直接失败交人工处理。
  注意本仓库 remote 名为 `origin`（不是父仓库的 `origin-private`）。
- 同步修正的两侧 CRLF 污染（不入库、非本仓库 diff）：Windows 两份副本的工作区文件被写成 CRLF，
  导致 `porcelain` 非空（xiaoyou-core 1834 个文件、mvp_core 193 个），同步 workflow 每次静默
  `exit 0` 且 job 仍显示成功。核查确认**仓库内提交的 blob 一直是 LF**，污染只在 Windows 工作区，
  故对两份副本设 `core.autocrlf=input` 后 `git reset --hard` 重新 checkout 即可，`porcelain` 分别
  降至 0，未产生 normalize commit、未改动任何提交内容。
- 验证：两个 workflow 经 YAML 解析通过，runs-on 标签正确（Linux X64 / Windows X64），且 `.github/`
  未被 `.gitignore` 排除（`git check-ignore` 无命中）。
- 备注：Windows 侧副本仍需手动 `git pull --ff-only` 追平到最新，或等待下一次 push 由 workflow 自动
  拉动；本机无 GitHub 凭据，Linux 侧无法代为拉取。

## 2026-09-20 S0 v2 正式成对比较（加速后端）与 B2 来源门修复

- B2 重跑 `b2_tts_t0_20260920_222648`：数值与上一批一致（RTF 0.3236/0.3138/0.3082，卡级增量
  3889.1 MiB、torch 2957.2/3690.0、load 8.53 s + 图捕获 4.56 s），但来源门变为干净
  （`passed=true`、`git_provenance_complete=true`、commit `1709ec4`、`dirty=false`），
  正确性 `completed_all_ok`。
- S0 正式比较 `s0_formal_20260920_223105_8510199`：`K24:T0` vs `K0:T0` 各 5 轮 = 10 次，
  **10/10 通过、0 failure**，执行位次轮转正确，来源干净（同 commit、`dirty=false`）。
- 结果（5 轮均值）：首句就绪 **4.384 vs 10.740 s**（配对 +6.355，5/5 全变差）、末句
  **25.241 vs 56.050 s**、RTF **0.3664 vs 0.3859**、**供给缺口 0 段 vs 6 段**
  （窗口内占比 0.0 vs 0.4414）；输出 80/9 句/31.6 s vs 65/8 句/26.5 s。
- 判读：v2 下混合相对可行基线在主指标上全面且一致占优，方向与 v1 相反（v1 三项全反）；
  但协变量未受控（文本/句数/token 不同，已标注）＋设备层 pending，**仍不登记 Gate B Go**，
  动态切换候选为空。补齐归因（输出一致或按 token 归一的敏感性分析）后才可登记。

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

## 2026-09-26（续）未跟踪文件圈定入库：285 个工作文件跟踪，wheel/运行时数据忽略

- 背景：IDE 源代码管理面板仍显示大量目录装饰（绿点/`U`），对应剩余 292 个未跟踪文件；用户确认按建议一次性处置。
- 改动：
  - **跟踪入库（~285 个，约 9.5 M）**：`paper/` 全部 227 个（正文 tex/数据 JSON/图 PDF-PNG，正文数据重置后现存即当前真源）、`thesis_materials/` 18 个、`tests/` 6 个正式测试、`cpp_scheduler/` 22 个（api/optimization/main.cpp/integration 测试等未接入 CMake GLOB 的源码，先入库留档）、`docs/` 3 个评估报告、`services/biological_system.py`+`bio_state_store.py`、`domain/entities/character.py`、`memory/persistent_state.py`（生物系统 Python 桥，C++ 端参与编译）、`scripts/fix_real_workload_agg.py`（`01_paper_main/real_workload_trace.json` 生成脚本，论文证据溯源）、`assets/scheduler_trace.png`、`experiments/elastic_residency/tts_component_offload_matrix.py`（含配套测试）、根目录 `check_env.sh`/`run_test.sh`/`EXPERIMENT_RERUN_PLAN.md`/`THESIS_EXPERIMENT_GUIDE_DRAFT.md`/`TODO.md`/`todo.txt`。
  - **删除重复（2 个）**：根目录 `System_Architecture_and_Performance_Report.pdf/.tex` 与 `paper/` 副本逐字节相同（diff 确认），保留 paper/ 正本。
  - **取消跟踪（2 个）**：`data/nvram.dat`、`data/test_state_backplane.dat.bio` 为运行时 mmap 数据，`git rm --cached` 后保留本地文件，并加入 `.gitignore`（此前 gitignore 已有 `test_state_backplane.dat` 但漏了 `.bio` 变体与 `nvram.dat`，已补齐）。
  - **忽略（141 M）**：`artifacts/wheels/llama_cpp_python-0.3.35-py3-none-linux_x86_64.whl`（CUDA wheel，本地保留免重编）。
- 验证：处置后 `git status` 未跟踪归零；`pytest -o asyncio_mode=auto tests/ -q` 基线不变。

## 2026-09-26 未跟踪临时脚本清理：删除 16 个一次性/重复脚本，未跟踪文件澄清

- 背景：git 重建后 status 折叠显示 147 行未跟踪目录，展开实为 4396 个文件；其中 2594 个是 `cpp_scheduler/deps/llama.cpp-master`（CMake 引用的本地依赖源码）、1494 个是 `stable-diffusion-webui-forge-main`（Forge 服务目录），两者均从未被跟踪、也不在 .gitignore，属基建噪音而非新增工作。用户要求先清理其中的临时脚本。
- 已删除（16 个，均为一次性调试脚本或与已跟踪正本重复的散落副本）：
  - 根目录一次性调试脚本 6 个：`test_env.sh`、`test_scheduler.sh`、`test_llm.sh`（引用不存在的 qwen2.5-7b 模型名）、`test_sched_import.py`、`test_cpp_sched.py`、`_test_route_task.py`——功能已被 `check_env.sh`、`cpp_scheduler/verify_bindings.py`、`tests/` 正式测试覆盖；
  - `install_cuda_llama.sh`（引用已不存在的 `mvp_core/venv` 旧路径，无 sm_120 参数，编译命令已记录在本文档与 PROJECT_CONTEXT）；
  - 与已跟踪正本逐字节相同的散落副本 3 个：根 `run_multiple_experiments.py`（正本 `experiments/migration/`）、根 `sync_to_wsl2_safe.ps1`（正本 `scripts/`）、`experiments/charts/generate_real_charts_legacy.py`（`paper/previous/` 有同副本）；
  - WSL 时代契约验证脚本 2 个：`tests/scripts/verify_wsl_requirements.py`、`tests/scripts/verify_wsl_sync_policy.py`（守卫对象已废止，已跟踪的同族脚本迁移期已删）；
  - 陈旧杂项 4 个：`weight.json`（内含 d:\AI Windows 路径的 GPT-SoVITS 配置残留）、`experiments/test.jpg`（`experiments/overhead/comprehensive_experiment.py` 缺失时自建）、`resource_migration_summary_N5.json`（上述重复脚本的运行输出，原始 5 轮迭代 JSON 仍在 `paper/experiment_data/migration/`）、`verify_data_consistency.py`（357 行一次性核对工具，硬编码 legacy 时代数字，正文数据 2026-08-30 重置后已失准）。
- 保留待用户决定跟踪与否的非脚本类未跟踪内容：论文/学位材料（`paper/`、`thesis_materials/`）、正式测试（`tests/test_baseline_*.py` 等 8 个）、C++ 未接入构建的新源码（`cpp_scheduler/api/`、`optimization/`、`main.cpp`、`tests/integration*`——CMake 只 GLOB core/workers/queue/utils）、无 Python 引用的生物系统桥（`services/biological_system.py`、`bio_state_store.py`、`domain/entities/character.py`、`memory/persistent_state.py`——对应 C++ 端经 GLOB 参与编译）、`artifacts/wheels/` 的 llama-cpp-python 0.3.35 wheel（免重编）、`scripts/fix_real_workload_agg.py`（`01_paper_main/real_workload_trace.json` 的生成脚本，按论文证据规则保留）。
- 验证：删除前逐一通读内容并 grep 全仓引用（仅历史日期的 UPDATES 条目提及，无活引用）；三对疑似重复均先 `diff` 确认。删除后 `pytest -o asyncio_mode=auto tests/ -q`：580 passed / 5 skipped / 4 failed，失败全部为 WSL2 无 MPS control daemon 的已知 NO_GO，与删除前基线一致。
- 备注：`cpp_scheduler/deps/` 与 `stable-diffusion-webui-forge-main/` 已于同日经用户批准加入 `.gitignore`（Build 段后新增本地依赖/服务目录段）。生效后未跟踪文件总数从 4396 降到 292（其中 `paper/` 227、`cpp_scheduler/` 16），剩余均为待用户圈定的真实工作文件。

## 2026-09-25（再续）git 主仓库迁回原生 Linux：复制 .git、行尾归一、废止 WSL2 同步提交流程

- 背景：用户明确本机（无 WSL 内核特征，原生 Ubuntu-24.04-E）是以后唯一开发环境，不再使用 WSL2 双副本；要求把 Windows 副本 `D:\AI\xiaoyou-core\legacy\mvp_core\.git` 复制到 `/home/leslie/mvp_core`，并改写与此冲突的项目规则。
- 改动：
  - `.git` 整目录 `cp -a` 复制到 Linux 工作树（37 M），`git fsck --connectivity-only` 通过；remote `origin` = `https://github.com/hakituo/EdgeSched`，分支 main，HEAD `bcb45b0`；用户信息沿用 hakituo。
  - git 本地配置改为 Linux 语义：`core.symlinks=true`、`core.fileMode=false`（NTFS 历史使全仓带 755 位，不跟踪 mode 以免噪音）、`core.ignorecase=false`。
  - 新增根 `.gitattributes`（`* text=auto eol=lf` + 常见二进制显式标记）；对已跟踪文本文件做一次性 CRLF→LF 机械归一（`grep -I` 跳过二进制，已删除文件跳过）。归一后 diff 从 613 M 降到 **51 个真实内容改动**，无行尾/mode 噪音。
  - 规则改写：`AGENTS.md`（环境与运行/提交与更新记录/验证段）、`README.md`（运行环境表、提交命令、验证命令）、`docs/PROJECT_CONTEXT.md`（环境表、同步边界段、7.1 快照）、`docs/PROJECT_INDEX.md`、`.trae/skills/xiaoyou-mvp-core/SKILL.md`、`.codebuddy/rules/mvp-core-project/RULE.mdc`、`.codebuddy/rules/mvp-core-docs-index/RULE.mdc`、`experiments/README.md`（SM-A provenance 流程的命令改 bash/Linux 路径）、`docs/NINE_B_MODEL_COMPARISON.md`。统一口径：主开发/运行/git 都在 `/home/leslie/mvp_core`；提交显式 `git add` 本任务文件、一句话 commit、禁止 auto/更新/同步 文案、只在用户明确要求时 push。
  - `scripts/sync_to_wsl2_safe.ps1` 头部加废止注释并在 param 后直接 `throw`，误运行立即失败（文件保留作历史参考）；`scripts/experiments/run_*.ps1` 等 8 个 Windows 包装脚本未改动（流程上已停用）。
- 中途问题与最终方案：Windows 复制来的 index 带有 NTFS stat 信息（dev/ino=0、size/mtime 陈旧），即使全量 LF 归一和 `--really-refresh`，status 仍长期虚报 200+ 个"M"（三方 blob 相同）。改用用户建议：从 GitHub 重新 `git clone --no-checkout`（HEAD `bcb45b0` 与本地一致，认证可用），用干净 `.git` 替换并 `git reset` 重建 index，幻影立即消失。
- `experiment_results` 实体化：原为指向 D 盘的目录软链，git 不穿透目录软链做工作树比对，导致其中 19 个已跟踪归档被误报删除；已将 1.7 G 内容复制为本地实体目录（复制后与 HEAD 完全一致，无差异），不再依赖 D 盘挂载。
- 最终工作树遗留（**未提交**，等用户决定去留）：**27 M**（本会话的规则/文档/TTS 路径/backend/requirements 改动 + 2 个运行时 `data/*.dat`）、**6 D**（`requirements-linux-cu128.txt`、`requirements-linux-forge.txt`、`scripts/commit_and_push.ps1`、`scripts/sync_from_windows.sh`、`tests/scripts/verify_requirements_contract.py`、`tests/scripts/verify_sync_policy.py`——迁移期已从 Linux 工作树移除但 HEAD 仍跟踪）、**148 ??**（`paper/` 87、cpp_scheduler 17、tests 8、根目录散落 test_*.sh/todo/TODO/thesis 材料等）。
- 备注：
  - Windows 旧副本保留但不再权威，不要再在其中提交，避免双历史分叉。
  - `.gitattributes` 让后续 checkout/add 自动维持 LF；如 Windows 侧仍偶尔查看仓库，建议其编辑器不做 CRLF 转换。
  - 旧 `.git` 验证无误后已删除（GitHub 上有完整备份）。

## 2026-09-25（续）加速 TTS 后端在共享 3.13 venv 验证通过 + TTS 模型路径切换 + 970 EVO 模型复用

- 背景：用户追问为何不直接在共享 venv 升级 transformers 5.x，并告知 TTS 模型在 `/home/leslie/xiaoyou-core/models`、970 EVO（`/media/leslie/970EVO/xiaoyou-models`）上有可用模型，要求查版本差异后决定是否升级。
- 调研与实测（结论：**0.2.6 + 4.57.3 可用，不升级 5.x**）：
  - PyPI/GitHub（andimarafioti/faster-qwen3-tts）：0.2.6（2026-04）依赖 `qwen-tts>=0.1.1` 与 `transformers>=4.57`；0.3.x/0.4.0（2026-08）已转向 `qwen-tts-hf`（transformers 5 兼容构建）。0.2.6 是唯一兼容当前共享栈的版本线，故主环境刻意保持 0.2.6。
  - 不升级的硬约束：`qwen-tts 0.1.1` 精确钉死 `transformers==4.57.3`，且该 venv 是 xiaoyou-core 主项目环境；升 5.x 须同时换 qwen-tts-hf，会破坏上游路径与主项目其他模块，且没有任何性能收益（实测对比见下）。
  - 真实 smoke（模型用 `/home/leslie/xiaoyou-core/models/Qwen3-TTS-12Hz-0.6B-Base`，RTX 5070 / bf16 / x-vector / eager）：直接 API 与经 mvp 集成入口 `build_synth("cuda_graph_static")` 两条路均成功——predictor 与 talker 两个 CUDA Graph 捕获通过（旧障碍 `StaticLayer.lazy_initialization` 签名错误不复现），输出有效音频；**稳态 RTF 0.3136（合成 1.73 s / 音频 5.52 s）与历史 5.x 隔离环境同口径 0.303–0.317 一致**；首个请求 6 s 级一次性开销（历史口径本就单列 graph_capture_s = 6.045 s）。
- 改动：
  - `experiments/tts/qwen3_tts_backends.py`：新增 `resolve_faster_qwen3_tts()`，加载优先级改为「环境 pip 包 → 旧 probe 目录回退」，并记录 `faster_backend_source/version/origin` 进产物元数据；`from_pretrained` 调用按签名过滤形参（0.2.6 无 `local_files_only`）；能力探测走同一解析。
  - TTS 默认模型路径改为 `/home/leslie/xiaoyou-core/models/Qwen3-TTS-12Hz-0.6B-Base`（8 个入口：`tts_component_offload{,_matrix}`、`llm_tts_colocated`、`s0_parent/child`、`smoke_qwen3_tts_base`、`profile_qwen3_tts_{cudagraph,gpu_sentence_lengths}`；`XIAOYOU_BENCH_TTS_PATH` 仍可覆盖）。两处副本 `diff -rq` 核验一致（各 2.4 G）。
  - 970 EVO 模型：把盘上额外图像权重软链进 Forge `models/`：2 个 checkpoint（krea2 turbo nvfp4 7.7 G、z-image turbo fp8 6.2 G）、4 个 VAE（ae/krea2Real/qwen_image/anime，共约 1.5 G）、1 个 openposeXL controlnet（774 M）。个人 lora 与论文无关，未链接。
  - LLM 权重去重（用户要求不存两份）：`models/llm/` 下 4 个文件（mmproj-BF16 0.67 G、4B Q4 2.74 G、9B mmproj 0.92 G、9B Q4 5.63 G，共约 9.3 G）先逐份 sha256 与 970 EVO `xiaoyou-models/llm/` 比对（4/4 完全一致），再删本地实体并替换为绝对路径软链，代码/脚本里的原路径全部不变。
- 验证：
  - 直接调用：加载+图捕获 7.5 s，第一句 16.1 s（含首次额外初始化）、第二句稳态 1.73 s 合成 5.52 s 音频，RTF 0.3136；
  - 集成调用：`accelerated_placement_capability` 报 site-packages/0.2.6；`build_synth` 返回 `faster_backend_source=site-packages`、graph_capture_s=6.045，合成 4.88 s 音频正常（峰值 0.436）；
  - 软链全部跟随可读（`ls -lL` 显示真实大小）；TTS 8 个入口的 CLI help 默认路径正确。
  - LLM：4/4 sha256 一致；软链替换后用 llama.cpp 经原路径 `models/llm/Qwen3.5-4B-Q4_K_M.gguf`（软链→NTFS）真实加载并生成，输出连贯中文，证明 mmap 路径对实验代码透明。
- 备注：
  - S0 v2 / 50 字·100 字画像等正式重跑仍按既有协议执行并落 manifest；本次结论只覆盖「环境可用 + 20 字级 RTF 对齐历史」，不替代正式批次。
  - 970 EVO 是 NTFS 外挂盘，GGUF 与 Forge 额外权重的软链可用性依赖挂载；论文主 checkpoint（SDXL Turbo）与辅助模型仍在 nvme 本地，盘未挂载时仅 LLM 实验和额外 Forge 权重受影响。
  - 本轮改动未 git commit（git 提交只在 Windows 侧做）。

## 2026-09-25 系统迁移后环境重建：主环境复用 xiaoyou-core 3.13 共享 venv、Forge 独立重建、C++ 调度器重编

- 背景：系统迁移（当前 WSL2）后原主环境软链与加速隔离环境均已不存在：`/home/leslie/.venvs/` 只剩一个空的 `forge-cu128`（Python 3.12、0 包），`qwen3tts-cugraph-probe/faster-qwen3-tts/.venv` 变成空的 Python 3.10 venv；C++ `build/` 产物也不存在。用户决定主环境直接复用父项目 venv `/home/leslie/xiaoyou-core/.venv`（uv 管理，Python 3.13.15，torch 2.12.0+cu130），缺的包装进去即可；Forge 仍单独装环境。
- 改动：
  - 盘点确认共享 venv 已满足 requirements.txt 的绝大部分（transformers 4.57.3、accelerate 1.12、qwen-tts 0.1.1、faster-qwen3-tts 0.2.6、faster-whisper、onnxruntime、sounddevice、scipy 等 214 个包），`torch.cuda.is_available()` 正常识别 RTX 5070 Laptop（驱动 595.91.07 / CUDA 13.2）。
  - 用 uv 向共享 venv 补装：diffusers 0.37.1（刻意 `<0.38`，避免拖入 huggingface-hub 1.x）、qwen-vl-utils 0.0.14、loguru 0.7.3、safetensors 0.7→0.8、torchvision 0.27.0+cu130、pybind11 3.1.0；**numpy 保持 1.26.4 不动**，未装源码零导入的 blendmodes/peft（blendmodes==2025 会强制 numpy 2.x）。
  - 用 `/usr/local/cuda-12.8`（nvcc）源码编译 `llama-cpp-python==0.3.35`：`GGML_CUDA=on`、`GGML_CUDA_ARCHITECTURES=120`，耗时约 3 分 41 秒；`llama_supports_gpu_offload()=True` 并枚举到本卡（compute capability 12.0）。
  - 重建主环境稳定入口：`/home/leslie/.venvs/mvp_core-cu128`（软链）→ `/home/leslie/xiaoyou-core/.venv`，沿用 2026-09-17 确立的「换环境只改软链」约定，现有 shell/ps1 脚本零改动。
  - Forge：删除空的 3.12 venv，用 `/usr/bin/python3.10`（3.10.21）+ `uv venv --seed` 重建 `/home/leslie/.venvs/forge-cu128`，预装 torch 2.10.0+cu128 / torchvision 0.25.0+cu128（launch.py 默认的 2.3.1+cu121 不支持 sm_120），其余 requirements 由 `launch.py --exit` 首次启动自动安装。三个安装坑（重建时必遇）：① seed 自带 setuptools 84 已删 `pkg_resources`，老 OpenAI CLIP 仓库构建失败，须先钉 `setuptools==69.5.1` 并手动 `pip install --no-build-isolation <CLIP zip>`（launch 的 pip 调用对隔离构建无法关闭）；② legacy_preprocessor 的可选依赖会把 numpy 带到 2.2.6，与核心钉版 `numpy==1.26.2`/blendmodes 2022/scikit-image 0.21 冲突，装完扩展后须 `uv pip install numpy==1.26.2` 钉回（二次 `--exit` 验证不会再升级）；③ insightface 自动安装失败仅为 warning，属可选预处理依赖，不影响核心。
  - C++：系统补装 `libuv1-dev`（CMake 配置报缺）；改 `cpp_scheduler/build_wsl2.sh` 指向共享 venv、注入 pybind11 cmake 目录与 `-DCMAKE_CUDA_ARCHITECTURES=120`，全量构建通过，产物 `build/scheduler_py.cpython-313-x86_64-linux-gnu.so`（906 KB），设置 `LD_LIBRARY_PATH=build/bin:build/llama.cpp/src` 后导入成功，API 枚举正常。
  - 文档：AGENTS、README、PROJECT_CONTEXT（环境表 + 7.1 快照）、PROJECT_INDEX、cpp_scheduler/README、thesis_guide/01、TTS_SUBSYSTEM、requirements*.txt 头部注释、.trae 与 .codebuddy 规则同步为新口径；历史日期记录（08-28 计划、09-06 复核、UPDATES 旧条目）不改写。
- 验证：
  - 共享 venv：llama_cpp GPU offload=True、torch CUDA=True、diffusers/qwen_vl_utils/loguru/torchvision 均可导入；scheduler_py 在 3.13 导入成功。
  - C++ 编译 100% 通过（cmake + nvcc，sm_120）。本轮只做绑定/导入级验证，**未跑真实 GGUF 推理与 Forge 出图**。
- 备注：
  - S0 v2 加速后端（`cuda_graph_static`，依赖 transformers 5.x 的隔离环境）迁移后已失；共享 venv 里是 faster-qwen3-tts 0.2.6 + transformers 4.57.3，能否替代未验证。重跑 S0 v2 前须先重建加速栈，历史数值只能引用。
  - 共享 venv 同时服务父项目 xiaoyou-core，本次刻意不升级 numpy/pillow 等既有包；后续装包需同样先做 dry-run 差分。

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
