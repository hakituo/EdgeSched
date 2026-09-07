# 两份 9B 选型对比（2026-09-07）

> **文件状态更新**：用户已要求并完成删除 WSL large `423fb3e5…` GGUF，small 与 projector 保留。本文 large 结果是历史证据；Large/双版本比较入口需恢复原哈希文件后才能复测，见 [删除记录与下一步](SMALL_9B_NEXT_COMBINATIONS_20260907.md)。

> **当前选型**：用户 small 视觉 Answer 批次 `v1_vision_9b_small_answer_20260907_142945` 已 1/1 技术通过且正确描述固定图的五项内容。后续 9B 采用 small `1a35ab89…` + projector `e36da9b9…`；9B 视觉启动器默认 Small，`-ModelVariant Large` 保留复测入口。新版本完整曲线已完成（C2 90、C3 150、KV 48），见 [验收](SMALL_9B_CURVES_20260907.md)；下文“未切换/待运行”是此前阶段记录，以文末验收为准。

目的：在决定是否重跑完整 16 档前，比较两份 Qwen3.5-9B 的运行内存、速度和 KV 端点行为。
只测纯文本固定填充 prompt，不加载 projector，不评价自然问答质量；视觉来源与配对门另行验收。
不更改主模型选型，不替换已冻结的旧结果，不晋升论文数据。

## 手动入口

从任意 PowerShell 目录执行：

```powershell
& 'D:\AI\xiaoyou-core\legacy\mvp_core\scripts\experiments\run_9b_comparison.ps1'
```

默认 2 模型 × FULL/K0 × 3 轮 = 12 个独立单元，固定 512 prompt tokens + 128 generated tokens，
n_ctx=640，LLM 14 物理核、系统 2 物理核；保留 SMT，与旧端点协议一致。
轮间反转模型顺序和配置顺序，配对种子为 42/43/44。
基于旧 9B CPU 端点，预计测量约 5–10 分钟，文件准备时间另计；新模型或系统负载可能改变耗时。
`-Configs FULL` 可只比较 GPU 两端，`-Repeats 1` 可缩成单轮 smoke；完整性按实际声明矩阵验收，不能把单轮称为稳定性能画像。
`-PrintOnly` 只打印命令，不访问 WSL、不复制模型、不同步或运行实验。

## 模型与存储隔离

- large：WSL `/home/leslie/mvp_core/models/llm/Qwen3.5-9B-q4_k_m.gguf`，SHA-256 `423fb3e5fe2c98df616159b7cd1bd04765f67667f5f8712a81e8f63f4e7ee1ef`。
- small 来源：Windows `D:\AI\xiaoyou-core\models\llm\Qwen3.5-9B-Q4_K_M.gguf`，SHA-256 `1a35ab892c0c4eec814d6594d34461193593a19110cd1d1285414efceb8a26ab`。
- small 运行副本：WSL `/home/leslie/mvp_core/models/llm/comparison_1a35ab89/model.gguf`。

用户运行时，先以 `sync_to_wsl2_safe.ps1 -SkipGit` 同步源码，再核对 large、small 来源哈希，
把 small 复制到独立目录并校验。已存在且哈希相同则复用；不同则报错，绝不覆盖。
两份模型均从 WSL 原生路径读取、按顺序单独加载；不测 Windows 挂载盘相对原生磁盘的差异。
模型哈希核验会预读整个文件，既有驻留入口也在每单元加载前计算哈希，因此不能声称测到了磁盘冷启动成本。
三轮新进程不等于同一模型实例连续请求；文件小不预设 RSS 或显存按同样幅度变小。

## 复用与产物

- `experiments/elastic_residency/compare_9b_models.py`：文件身份检查、冻结 experiments 源码快照、交替编排、调用既有 `llm_residency_curve`、逐单元核验和失败停止。
- `compare_9b_support.py`：流式哈希、非覆盖复制、计划、资源提取和成对汇总；不导入模型后端。
- 结果：`experiment_results/elastic_residency/model_comparison/<run_id>/`。
- `comparison.md` 给出平均加载/TTFT/decode、整卡峰值、峰值减基线、RSS 峰值对照；`summary.json` 另含 min/max、KV、swap、逐轮差值与完整性。
- `units/` 保留原驻留入口生成的 manifest、iterations、child、timeline 和日志；外层保存计划、模型身份、源码快照和每条实际命令。

沿用真实后端、direct NVML、GPU 新鲜度、swap=0、affinity、KV 保存恢复和资源归还门禁；
jitter 使用既有 observe 模式，保留性能目标失败，不将调度抖动直接冒充无效测量。
任一运行非零退出、身份变化、raw child/summary 不一致或配对 prompt token 哈希不同，失败即停。
失败与中断保留已生成记录，不自动续跑或反复重试；中断/外层超时清理本单元进程组。
三轮只作选型诊断，不给显著性或可靠 p95 结论，不自动选模型；填充 prompt 的续写不作为问答质量证据。

## 本地验证

`tests/elastic_residency/test_compare_9b_models.py`：17 项无模型验证通过，覆盖计划、复制复用/冲突保护、
原始峰值口径、空值、配对完整性、命令参数、模型/child/源码快照篡改拒绝；Ruff 通过。
手动入口已做 PowerShell 语法及 PrintOnly 检查。新增 Python 文件均少于 300 行。
实现期间没有复制真实模型、同步 WSL、加载模型或运行真实实验。

## 首轮汇总字段修复（2026-09-07）

用户批次 `compare_9b_20260907_023233` 在 `audit_unit` 读取 `inference.output_text` 时抛出 KeyError。
实际驻留入口已经只保存 `output_text_preview`（前 200 字）与完整 `output_text_sha256`，比较器错误假设存在全文，
原假后端测试也使用了错误字段，因此未捕获问题。这是比较器输出契约错误，不是模型/CUDA 失败。

修复后按原字段保存预览和全文哈希，显式标记 preview-only，不将预览冒充全文；缺字段或哈希格式错误仍拒绝。
测试改用当前生产契约，并覆盖预览缺失、无效哈希以及预览哈希不同于全文哈希的合法情况。
17 项测试及 Ruff 通过；对用户真实 `01_large_FULL_r1` 原始产物执行离线核验通过，decode 为 54.7245 tokens/s。

原批次只完成首个 large/FULL 单元，其余 11 个未执行；原 summary 的失败状态保留，
单独写入 `audit_recovery.json` 保存恢复的只读核验记录，不把单元恢复解释为完整对比通过。
后续用户重新运行同一 PowerShell 入口即可生成新 run id；已复制的小模型校验一致后复用，不重复覆盖或拷贝。
不把旧单元拼入新批次，以保持交替顺序和单批环境口径；本次修复没有启动新实验。

## 完整对比结果（用户运行，2026-09-07）

批次：`experiment_results/elastic_residency/model_comparison/compare_9b_20260907_023703/`。
12/12 通过、失败 0、所有单元 swap 0，六组配对 prompt token 哈希一致。
Windows 离线重核每个单元的 iterations、child、模型身份、参数、acceptance checks 和源码快照，
重新汇总与原 summary 完全一致（只将 Windows 核验路径映射回原 Linux 路径）。性能目标也为 12/12 通过。
源提交 `39b8f0e44867dbe42e28928b925d13f95085d81d`，dirty=true、31 个状态条目，保留本批源码快照；不称为干净提交实验。

协议为两模型各 FULL/K0 三轮，512 输入 + 128 生成、n_ctx=640、LLM 14 物理核、系统 2 物理核。
large/small 分别为上文 `423fb3e5…` / `1a35ab89…`，均使用 WSL 原生副本。
下表均为三轮均值，MiB 使用 2^20 字节：

| 指标 | large | small | small 相对收益 |
|---|---:|---:|---|
| 文件大小 MiB | 5698.04 | 5366.37 | 减少 331.67 MiB |
| FULL 首 token s | 0.4136 | 0.2443 | 减少 40.95% |
| FULL decode tokens/s | 55.919 | 57.063 | 增加 2.05% |
| FULL 整卡峰值 MiB | 6441.27 | 6351.27 | 减少 90.00 MiB |
| FULL 峰值减基线 MiB | 5633.10 | 5543.10 | 减少 90.00 MiB |
| FULL 进程 RSS 峰值 MiB | 6113.32 | 5781.86 | 减少 331.45 MiB |
| K0 首 token s | 8.5617 | 8.1209 | 减少 5.15% |
| K0 decode tokens/s | 4.441 | 4.501 | 增加 1.35% |
| K0 进程 RSS 峰值 MiB | 8271.32 | 7931.13 | 减少 340.18 MiB |

两模型 FULL GPU KV 均为 24 MiB，计算缓冲均为 GPU 501.77 / CPU 16.77 MiB；
K0 CPU KV 均为 24 MiB。两模型 K0 整卡峰值均为 935.27 MiB、峰值减基线 127.10 MiB，
该整卡读数不能解释成模型权重占用，也不能把未报告的 GPU KV 字段当成零。
FULL 整卡基线全部为 808.17 MiB，因此本批 90 MiB 差值不是基线变化造成。

六组配对均观察到 small 首 token 更低、decode 更高，但三轮诊断不提供显著性结论。
KV 恢复均值 FULL 为 large 0.943 / small 1.033 s，K0 为 1.032 / 1.154 s，不能称所有操作都更快。
RSS 是进程口径，mmap/页缓存影响系统可用内存，不能把 RSS 降幅直接当成整机释放量。

选型建议：small 值得作为后续候选，主要收益是少约 332 MiB 文件与约 331–340 MiB 进程峰值，
GPU decode 小幅改善，实际显存只省 90 MiB，不能据此判定 TTS 共置可行。
用户目标仍为 9B + 视觉：small 尚未做 projector 配对、真实视觉和自然问答验收，当前不改主模型默认值。
若后续正式选定 small，其完整驻留/KV 曲线需要按新哈希重建；已完成的 4B 数据保留。
历史 C2/C3 9B 为另一哈希 `de10bfef…`，不属于本批 large，旧曲线不能改名归到任一新文件。

本次只离线核验与登记，未运行模型。另修复 comparison.md 分组名中的竖线造成表格错列，
补充无模型验证并仅从原 summary 重生成派生 Markdown；原 JSON、日志及源码快照保持原样。
当前比较器无模型测试 18/18 通过，Ruff 通过；三个相关 Python 文件均少于 300 行。

## 小版 9B 视觉试测入口（2026-09-07，待用户运行）

用户已要求试用 small。复用现有视觉入口，以 `-ModelVariant Small` 选择已存在的 WSL 独立副本；
固定 small SHA `1a35ab89…` 和现有 projector SHA `e36da9b9…`，运行时完整核验。
默认省略该参数仍使用 Large，历史配置和产物不覆盖；small 自动 run id 为 `v1_vision_9b_small_<stage>_<timestamp>`。

```powershell
& 'D:\AI\xiaoyou-core\legacy\mvp_core\scripts\experiments\run_vision_9b_smoke.ps1' -ModelVariant Small -Stage Answer
```

本次先跑一轮 FULL 视觉回答，512 上下文预算 + 1024 生成预算（n_ctx=1536），
同时验证真实图像编码、生成、KV 和资源门禁，并留下足够长的回答预览供复核。
仍沿用固定生成预算、遇 EOS 不停止的既有协议；技术通过不代表回答正确，也不代表多轮性能画像。
运行后检查 child 的回答预览、图形内容与门禁，再决定是否选定 small 并启动完整曲线；不自动串行补跑其他阶段。
脚本由用户执行时会增量同步代码，不提交 Git、不重拷贝模型；结果在 `vision_endpoint/` 下。

静态配对修复：small 的 name 为 `Qwen3.5 9B`、basename 为 `Qwen3.5`；现有 projector 的 name 为
`Qwen3.5-9B`、basename 缺失。两者基础模型名和来源 URL 完全相同，投影/隐藏维度均为 4096。
原门禁将单侧 basename 缺失直接判失败；现在单列
`projector_normalized_name_and_source_match_with_partial_basename`，仅忽略名称的空格/连字符，
要求非空基础模型名及来源 URL 精确一致。双侧 basename 存在但不一致仍拒绝，双侧缺失保留原精确名称分支。
结构、SHA、来源以及运行期门禁均保留；该分支只允许进入实际兼容性试测，不证明不同量化发布物视觉质量相同。

Windows 真实文件头核验通过（未重算大文件哈希、未加载模型），记录为
`model_comparison/small_vision_header_preflight_20260907.json`；WSL 实际文件身份仍由启动时固定 SHA 门核验。
配对/端点门无模型测试 33 项通过，包含错误型号、来源缺失/冲突、投影维度、哈希、双侧 basename 冲突拒绝；
PowerShell 两版本三阶段 PrintOnly 检查和 Ruff 通过。实现期间没有同步 WSL 或运行真实实验。
随后对全部视觉静态/假后端测试回归，86/86 通过；不产生真实模型性能或视觉质量结论。

### PowerShell → Bash 换行修复

用户启动 small Answer 时出现 `invalid option name` / `pipefail` 错误；检查发现启动器为 CRLF，
PowerShell here-string 将行尾 CR 一并传给 `bash -lc`，使首行的选项成为 `pipefail\r`。
错误发生在 Python 入口之前，不是 small 模型或视觉加载失败。
视觉启动器与同结构的双模型比较启动器现均在传递前将命令换行统一为 LF，避免依赖 Git/编辑器的文件换行设置。
原静态测试只查参数文本，未检查 CR；现在补充无 CR、首行精确值，以及两个启动器分别以 LF/CRLF 文件生成命令的回归。
PowerShell 验证全部通过，WSL 调用由测试桩禁止；没有启动真实实验。用户重跑上面的同一条命令即可生成新批次。

## 小版视觉验收与选型（2026-09-07）

原始路径：`experiment_results/elastic_residency/vision_endpoint/v1_vision_9b_small_answer_20260907_142945/`。
用户运行完成 1/1、0 failure/fallback，34 项运行门禁全通过，swap 全程 0，资源租约归还完成。
离线核验 summary/iterations/child 一致、全文 SHA 一致、源码快照 SHA 一致；
真实文件哈希为 small `1a35ab892c0c4eec814d6594d34461193593a19110cd1d1285414efceb8a26ab`
和 projector `e36da9b9891e13a172b25028ad18b4fc63bc4d681cf391a44b2a9a3b51bc797a`。
源提交为 `39b8f0e44867dbe42e28928b925d13f95085d81d`、dirty=true、31 状态条目，源码快照 SHA
`dcf5e9917603abcf404b5c00e562bba841f5e0c9abe36da6af6c1bdb78ad5ea9`。

实际 448×448 图像产生 196 个视觉 tokens，推进 14 个位置；文本 23 tokens，生成 1024 tokens，
KV 保存恢复 1061 个位置通过。配置 FULL / VISION_WARM_GPU、n_ctx=1536、14+2 物理核。
完整回答正确描述白色背景、黑色边框、中央红圆、左上蓝色正方形、右下绿色正方形。
与 large 旧 Answer 漏黑边相比，本张图描述完整，但单图不能支持普遍质量优于 large 的结论。
输出包含思考及 12 个 EOS 后续写，固定预算忽略 EOS 的协议会继续生成；不是生产对话的停止行为验收。

| 指标 | small 本次 Answer | large 旧 Answer 20260906_204651 |
|---|---:|---:|
| 首 token s（不含模型加载） | 0.4251 | 0.5641 |
| decode tokens/s | 58.738 | 57.456 |
| 整卡基线 MiB | 787.92 | 767.38 |
| 整卡峰值 MiB | 7501.02 | 7568.47 |
| 峰值减基线 MiB | 6713.10 | 6801.10 |
| 进程 RSS 峰值 MiB | 5780.14 | 6111.90 |
| GPU KV buffer MiB | 48 | 48 |

两次 Answer 的 prompt 文本/token 哈希、图像身份与关键预算一致，但跨时段、跨版本源码单轮观察，不能当作严格配对速度比较。
峰值减基线少 88 MiB，与之前纯文本三轮少 90 MiB 的方向一致；原始整卡峰值差只有 67.46 MiB，基线不同须分开解释。
此 n_ctx 的 48 MiB GPU KV 已包含在峰值内，不能再加一次；不能混入 n_ctx=640、KV=24 MiB 的五轮 Profile 曲线。

结合纯文本 12/12 和本次视觉验收，按用户要求选 small 作为后续 9B 版本，并将专用视觉启动器默认改为 Small；
Large 通过显式参数保留，4B 对照和旧 9B 证据保持原模型身份。通用驻留 CLI 的历史默认值未在此次修改，重跑须显式固定 small 路径/哈希。
下一步是为选定版本重建完整驻留与 KV 曲线，再重算容量候选；TTS 共置/SM 调度和新版本视觉多轮画像均未由本次单轮替代。
未晋升论文正文数据。本次只读取既有产物、修改默认入口与文档，并通过 PowerShell 静态回归，没有运行模型。

### 小版完整曲线手动命令

```powershell
& 'D:\AI\xiaoyou-core\legacy\mvp_core\scripts\experiments\run_9b_small_curves.ps1'
```

依次执行 C2 六档 × 三上下文 × 五轮（90）、C3 其余十档 × 三上下文 × 五轮（150）、
KV 十六档 × 三上下文 × 一轮（48）。上下文为 512/2048/4096，生成 128，14+2 物理核，seed=42，
驻留 jitter 沿用 observe；任一阶段失败立即停止，不自动续跑。此为 TEXT_ONLY，视觉另列画像。
脚本增量同步代码、核对 small 固定 SHA、保存共享源码快照，三阶段输出各自独立 run id，旧模型与数据不覆盖。
`-PrintOnly` 只查看命令；`tests/elastic_residency/test_9b_small_curves_launcher.ps1` 已验证矩阵、身份与 LF，未调用 WSL。

## 完整曲线已完成

用户批次 `20260907_143917` 已 288/288 通过并经原始记录逐条复核，swap 全程 0。完整表格、来源与下一步见 [小版曲线验收](SMALL_9B_CURVES_20260907.md)。此前“待重建”保留为历史阶段记录，不表示还要重跑。

## small 五轮视觉画像已完成

用户 `v1_vision_9b_small_profile_20260907_171635` 已 5/5、swap 0，原始记录核验通过；见 [阶段统计与边界](SMALL_9B_VISION_PROFILE_20260907.md)。基础生命周期画像不需重复；持续会话/保温/提前释放仍待验证。
