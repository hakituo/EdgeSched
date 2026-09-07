# 辅助任务 CPU 核数画像

入口：`python -m experiments.elastic_residency.auxiliary_core_budget`，真实运行使用
WSL2 `/home/leslie/.venvs/mvp_core-cu128/bin/python`，不修改在线调度器或 LLM 默认模型。

## 实现边界

- `auxiliary_inputs.py`：冻结 1000 条合成 JSONL 文档、六个查询和中文 TTS 截取音频；逐文件 SHA-256 校验，拒绝路径逃逸与输入篡改。
- `auxiliary_workloads.py`：驻内存字符 Jaccard top-32 JSON RAG、BGE 后端注册与 Faster-Whisper CPU int8；保留 `BgeRerank` 导入兼容。JSON 检索不含磁盘加载；ASR 包含音频读取/特征提取及完整转写，不是只创建生成器。
- `auxiliary_bge.py`：共享 ONNX BGE CLS/L2 编码器；`bge` 每次重编码 query+32 候选，`bge_cached` 预计算 32 个候选后每次只编码 query，再排序输出 top-8。两种模式的 max length 与 padding 均固定为 128，只改变候选缓存因素；它不是父项目所有配置的完整复刻。
- `auxiliary_child.py`：只接受 Linux sysfs 完整物理核组，复用 `PhysicalCoreAllocator`、`process_affinity_scope` 和 `ResourceSampler`；SMT 关闭，每个任务核用一个代表线程。重库导入先绑核，原生线程在推理前后逐一核验，异常也归还租约。CPU 时间/墙钟时间给出实际平均核利用量，不能把 affinity 数量当利用率。
- `auxiliary_report.py`：预先冻结并随机化矩阵，分开测量有效性与 SLA；完整五轮、至少 30 个事件才报告 p95。缺失/失败/重复轮次不得推荐最低核数。
- `analyze_auxiliary_core_budget.py`：独立检查源码/输入哈希、父/子配置、原始事件、线程/有效性门禁及 CPU 首样本契约，离线重算汇总，输出 CSV 与可选曲线。
- `auxiliary_core_budget.py`：配置独立子进程、超时、失败即停/显式继续、逐单元 checkpoint、模型/输入/环境身份与实际运行源码 ZIP。拒绝覆盖 run id。

## 冻结协议

默认 `R_sys=2`，JSON RAG 为 1/2 核，BGE/Whisper 为 1/2/4 核。
每配置五次独立进程，每轮先对六个输入分别预热，再测六个事件（30 个事件/配置）。
SLA 固定为 JSON RAG p95 50 ms、BGE p95 150 ms、Whisper p95 RTF 0.8。
最大测试预算仍失败时标记 infeasible，不自动扩大矩阵。

`--system-cores 0 1 2 4` 运行预留核敏感性矩阵。R_sys>0 时采样线程放到系统租约，
R_sys=0 时显式与辅助 worker 共享；预留不会修改宿主操作系统或其他进程的 affinity。
这只证明隔离辅助任务在不同预留设置下的表现，**不是 D3 正式 R_sys 验收**。
D3 必须加入同一固定 LLM→TTS 主流水线、同放置无辅助任务配对基线、真实 chunk
播放 underrun、decode/TTFA/完整回复损失 ≤5% 与 jitter 对照。本入口在这些证据缺失时
保持 `formal_r_sys.status=not_evaluated`，不判定最优 R_sys，也不放行 S0 Go/No-Go。

## 运行示例

先在有音频的目录冻结输入（来源必须是至少十秒的 PCM16 中文语音）：

```bash
python -m experiments.elastic_residency.auxiliary_inputs \
  --output experiment_results/elastic_residency/auxiliary_inputs/my_inputs \
  --source-wav /absolute/path/to/chinese_tts.wav
```

然后用完整、真实的 Windows Git commit 与工作区状态运行：

```bash
python -m experiments.elastic_residency.auxiliary_core_budget \
  --run-id my_auxiliary_curve \
  --inputs experiment_results/elastic_residency/auxiliary_inputs/my_inputs/inputs.json \
  --system-cores 0 1 2 4 --repeats 5 \
  --source-git-commit "$SOURCE_COMMIT" \
  --source-git-dirty false --source-git-status-entry-count 0
```

模型默认来自 WSL `models/auxiliary/`，支持 `--bge-model`/`--whisper-model` 覆盖。
只有显式 `--keep-going` 才跨过测量失败；SLA 不达标不属于测量失败。
`--repeats 1` 是 smoke，不输出 p95 或最低核数结论。

产物包含 manifest、run_config、原始 iterations、summary、结构化 failures、每单元日志、
输入副本、运行源码快照和每个子进程的资源时间线。数据不自动晋升论文正文。
音频由已有 TTS 长句裁剪为 6/8/10 秒；六次查询只覆盖三段音频，重复事件不是独立语音样本。
语料与音频仅用于该冻结负载的性能画像，不证明检索准确率或真实人声 ASR 泛化。

CPU 时间为整个子进程（含采样线程）的差分；短任务受到计时分辨率影响。任务结束后会在计时窗口之外等一个有效资源样本，不把采样等待计入耗时。

## BGE 候选缓存补测（2026-09-06）

用户执行的 `d1_auxiliary_manual_v1` 已完成 40/40 次、240 个事件并通过离线复算。
R_sys=2 下，JSON RAG 1 核 p95=2.35 ms，Whisper 1 核 p95 RTF=0.691；BGE 全量重编码
1/2/4 核 p95 分别为 1073/667/696 ms，未满足 150 ms SLA。旧数据保留为候选全量重编码
负载的结果，不外推为候选已缓存时不可行，不自动晋升论文正文。

代码核查发现，父项目记忆链路在写入时保存候选 embedding，检索时仅编码未缓存 query，
然后读取/解码候选向量；本项目演示入口仍是关键词检索。因此增加独立 `bge_cached` 画像，
不把它描述成已经接入在线调度器。

新模式在每次独立进程内，模型加载后、预热及请求计时前调用一次 `prepare()`：

- `load_s` 仅记录模型/Tokenizer 加载；`preparation.elapsed_s` 单独记录候选编码和缓存准备。
- 缓存保存 32×512 的归一化只读向量、字节数、向量 SHA-256 与候选 ID/内容身份。
- 每次预热和测量仍调用模型编码 query，不缓存 query 或最终排序结果；请求耗时包括
  query 分词/编码、归一化、32 候选相似度及 top-8 排序。
- 汇总按 `bge` / `bge_cached` 分开，SLA 仍为 150 ms。候选预计算只报告五轮原始值、
  mean/min/max，不把五个样本当作可靠 p95，也不将准备成本隐藏到请求指标里。
- 缺少缓存、错误向量、请求重新编码候选、缓存身份不一致或启用了 query 缓存时不通过门禁。
- 当前只通过静态和假 ONNX 后端测试；真实性能由用户执行下列命令后判断。

从任意 PowerShell 目录运行：

```powershell
& 'D:\AI\xiaoyou-core\legacy\mvp_core\scripts\experiments\run_auxiliary_cached.ps1'
```

脚本自行定位项目、以 `-SkipGit` 增量同步、读取最新 Git 来源，生成带时间戳的新 run id，
只运行 `--tasks bge_cached --system-cores 2 --repeats 5`，即 1/2/4 核共 15 次、90 个计时事件。
不会加载 LLM，也不重跑 JSON RAG/Whisper。加 `-PrintOnly` 只显示命令，不同步、不启动 WSL。
