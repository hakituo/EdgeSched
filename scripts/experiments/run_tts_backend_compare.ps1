# 用户手动运行：Qwen3-TTS 推理后端对照（上游动态 vs CUDA Graph 静态）。
#   -Mode both      两条路径在同一进程内先后测量（同环境基线 + 加速，推荐）
#   -Mode baseline  只跑上游动态路径
#   -Mode graph     只跑 CUDA Graph 静态路径
# 说明：CUDA Graph 路径依赖隔离环境里的 faster-qwen3-tts（qwen-tts-hf +
# transformers 5.15.1），不能装进 mvp_core-cu128；-Python 可换解释器。
param(
    [ValidateSet('both', 'baseline', 'graph')][string]$Mode = 'both',
    [ValidateSet('eager', 'sdpa')][string]$Attn = 'eager',
    [int]$Rounds = 3,
    [switch]$PrintOnly,
    [string]$RunId = '',
    [string]$Python = '/home/leslie/qwen3tts-cugraph-probe/faster-qwen3-tts/.venv/bin/python'
)
$ErrorActionPreference = 'Stop'
$Project = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$Root = 'experiment_results/tts/cudagraph'

$PathArg = switch ($Mode) {
    'both' { 'upstream_dynamic cuda_graph_static' }
    'baseline' { 'upstream_dynamic' }
    'graph' { 'cuda_graph_static' }
}
if ($Rounds -lt 1) { throw "轮次必须为正整数，当前为 $Rounds。" }
if ($RunId -and ($RunId -notmatch '^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$')) {
    throw "非法 RunId：$RunId（只允许字母数字与 _ . -，且首字符为字母数字）。"
}
if ($Mode -eq 'graph' -and $Attn -eq 'sdpa') {
    Write-Host '[警告] CUDA Graph 路径官方基准使用 eager；sdpa 下图捕获可能失败。' -ForegroundColor Yellow
}
$Stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
if (-not $RunId) { $RunId = "tts_backend_${Mode}_${Attn}_$Stamp" }

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
    $Command = @"
set -uo pipefail
cd /home/leslie/mvp_core
$Python -u -m experiments.tts.profile_qwen3_tts_cudagraph \
  --paths $PathArg \
  --attn-implementation $Attn \
  --rounds $Rounds \
  --run-id '$RunId'
status=`$?
mkdir -p '$Root/$RunId'
printf 'commit=%s\ndirty=%s\nstatus=%s\nmode=%s\nattn=%s\npython=%s\n' \
  '$Commit' '$Dirty' "`$status" '$Mode' '$Attn' '$Python' > '$Root/$RunId/provenance.txt' || true
exit `$status
"@
    $Command = $Command.Replace("`r`n", "`n").Replace("`r", "`n")
    if ($PrintOnly) { Write-Output $Command; return }
    Write-Host "TTS 后端对照：paths=[$PathArg]，attn=[$Attn]，rounds=[$Rounds]。"
    Write-Host '口径：0.6B Base + x-vector 克隆 + 非流式完整波形 + do_sample=False；两侧同一进程先后测量。'
    Write-Host '边界：CUDA Graph 静态 KV cache 与上游不是位精确，且逐次音频可能不同，必须逐条记录。'
    wsl -e bash -lc $Command
    if ($LASTEXITCODE -ne 0) { throw "TTS 后端对照 $RunId 未通过，请保留日志与产物。" }
    Write-Host "结果：$Root/$RunId"
} finally { Pop-Location }
