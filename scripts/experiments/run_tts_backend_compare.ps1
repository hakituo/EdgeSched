# 用户手动运行：Qwen3-TTS 推理后端对照（上游动态 vs CUDA Graph 静态）与 B2 画像。
#   -Mode both      两条路径在同一进程内先后测量（同环境基线 + 加速，推荐）
#   -Mode baseline  只跑上游动态路径
#   -Mode graph     只跑 CUDA Graph 静态路径
#   -Mode b2        只跑加速路径的 B2 画像：20/50/100 字 × 5 轮 + 状态残留序列，
#                   并在主环境补 ASR 回转核对（correctness.json）
# 说明：CUDA Graph 路径依赖隔离环境里的 faster-qwen3-tts（qwen-tts-hf +
# transformers 5.15.1），不能装进 mvp_core-cu128；-Python 可换解释器。
# 画像与 ASR 核对默认都用加速隔离环境（已装 faster-whisper 1.2.1；
# BGE 只需 onnxruntime + tokenizers，该环境已有），见 -MainPython 可覆盖。
param(
    [ValidateSet('both', 'baseline', 'graph', 'b2')][string]$Mode = 'both',
    [ValidateSet('eager', 'sdpa')][string]$Attn = 'eager',
    [int]$Rounds = 3,
    [switch]$PrintOnly,
    [string]$RunId = '',
    [string]$Python = '/home/leslie/.venvs/mvp_core-cu128/bin/python',
    [string]$MainPython = '/home/leslie/.venvs/mvp_core-cu128/bin/python'
)
$ErrorActionPreference = 'Stop'
$Project = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path

$PathArg = switch ($Mode) {
    'both' { 'upstream_dynamic cuda_graph_static' }
    'baseline' { 'upstream_dynamic' }
    'graph' { 'cuda_graph_static' }
    'b2' { 'cuda_graph_static' }
}
$Root = if ($Mode -eq 'b2') {
    'experiment_results/elastic_residency/b2'
} else {
    'experiment_results/tts/cudagraph'
}
if ($Mode -eq 'b2' -and -not $PSBoundParameters.ContainsKey('Rounds')) { $Rounds = 5 }
if ($Rounds -lt 1) { throw "轮次必须为正整数，当前为 $Rounds。" }
if ($RunId -and ($RunId -notmatch '^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$')) {
    throw "非法 RunId：$RunId（只允许字母数字与 _ . -，且首字符为字母数字）。"
}
if ($Mode -in @('graph', 'b2') -and $Attn -eq 'sdpa') {
    Write-Host '[警告] CUDA Graph 路径官方基准使用 eager；sdpa 下图捕获可能失败。' -ForegroundColor Yellow
}
$Stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
if (-not $RunId) {
    $RunId = if ($Mode -eq 'b2') { "b2_tts_t0_$Stamp" } else { "tts_backend_${Mode}_${Attn}_$Stamp" }
}
$StateRunId = "${RunId}_state"

Push-Location -LiteralPath $Project
try {
    if (-not $PrintOnly) {
        & (Join-Path $Project 'scripts/sync_to_wsl2_safe.ps1') -SkipGit
        if ($LASTEXITCODE -ne 0) { throw '同步失败，停止。' }
    }
    $Commit = git rev-parse HEAD
    if ($LASTEXITCODE -ne 0) { throw '读取 Git 提交失败。' }
    $StatusLines = @(git status --porcelain)
    $Dirty = if ($StatusLines.Count) { 'true' } else { 'false' }
    # 每条命令一行，用数组拼装；here-string 末尾换行会被吞，不能直接相加。
    # 来源门由主机侧读到的 commit/dirty 传入：WSL2 副本没有 .git，进程内探测必然不可用。
    $Source = "--source-git-commit '$Commit' --source-git-dirty '$Dirty' --source-git-status-entry-count $($StatusLines.Count)"
    $Body = @(
        "$Python -u -m experiments.tts.profile_qwen3_tts_cudagraph --paths $PathArg --attn-implementation $Attn --rounds $Rounds --output-root '$Root' --run-id '$RunId' $Source || status=1"
    )
    if ($Mode -eq 'b2') {
        $Body += "$Python -u -m experiments.tts.profile_qwen3_tts_cudagraph --paths $PathArg --attn-implementation $Attn --sequence correctness --output-root '$Root' --run-id '$StateRunId' $Source || status=1"
        $Body += "$MainPython -u -m experiments.tts.check_tts_output_correctness --run-dir '$Root/$RunId' --cores 4 || status=1"
        $Body += "$MainPython -u -m experiments.tts.check_tts_output_correctness --run-dir '$Root/$StateRunId' --cores 4 || status=1"
    }
    $BodyText = $Body -join "`n"
    $Command = @"
set -uo pipefail
cd /home/leslie/mvp_core
status=0
$BodyText
mkdir -p '$Root/$RunId'
printf 'commit=%s\ndirty=%s\nstatus=%s\nmode=%s\nattn=%s\npython=%s\nmain_python=%s\n' \
  '$Commit' '$Dirty' "`$status" '$Mode' '$Attn' '$Python' '$MainPython' > '$Root/$RunId/provenance.txt' || true
exit `$status
"@
    $Command = $Command.Replace("`r`n", "`n").Replace("`r", "`n")
    if ($PrintOnly) { Write-Output $Command; return }
    if ($Mode -eq 'b2') {
        Write-Host "B2 TTS 画像：accelerated=[cuda_graph_static]，attn=[$Attn]，rounds=[$Rounds]，run=[$RunId]。"
        Write-Host '口径：0.6B Base + x-vector 克隆 + 非流式完整波形 + do_sample=False；20/50/100 字各五轮。'
        Write-Host '记录：RTF、波形就绪、加载与图准备成本、准备/执行阶段内存峰值、输出长度与 ASR 回转正确性。'
        Write-Host '放置：当前后端只支持整模型 GPU（T0）；T1–T4 未经后端支持确认不得重测或沿用旧值。'
    } else {
        Write-Host "TTS 后端对照：paths=[$PathArg]，attn=[$Attn]，rounds=[$Rounds]。"
        Write-Host '口径：0.6B Base + x-vector 克隆 + 非流式完整波形 + do_sample=False；两侧同一进程先后测量。'
        Write-Host '边界：CUDA Graph 静态 KV cache 与上游不是位精确，且逐次音频可能不同，必须逐条记录。'
    }
    wsl -e bash -lc $Command
    if ($LASTEXITCODE -ne 0) { throw "TTS $Mode $RunId 未通过，请保留日志与产物。" }
    Write-Host "结果：$Root/$RunId"
    if ($Mode -eq 'b2') { Write-Host "状态残留与正确性：$Root/$StateRunId/correctness.json" }
} finally { Pop-Location }
