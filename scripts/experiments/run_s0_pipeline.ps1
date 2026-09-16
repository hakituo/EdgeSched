# 用户手动运行：S0 无图像 LLM→TTS 流水线。
#   -Mode startup  仅 K24:T0，首句即播 vs 全量预缓冲启动策略小对照（各 1 轮）
#   -Mode formal   四配置 × 冻结输入 × 5 轮正式成对比较（20 次）
#   -Mode analyze  在 Windows 侧离线复算最近一次正式批次（不加载模型）
param(
    [ValidateSet('startup', 'formal', 'analyze')][string]$Mode = 'startup',
    [switch]$PrintOnly,
    [int]$Runs = 0,
    [switch]$FailFast,
    [string]$RunId = ''
)
$ErrorActionPreference = 'Stop'
$Project = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$WindowsPython = (Resolve-Path (Join-Path $Project '../../venv_core/Scripts/python.exe')).Path
$WslPython = '/home/leslie/.venvs/mvp_core-cu128/bin/python'
$Root = 'experiment_results/elastic_residency/s0_pipeline'

if ($Mode -eq 'startup') {
    $Placements = 'K24:T0'
    $Policies = 'first_sentence full_prebuffer'
    if ($Runs -lt 1) { $Runs = 1 }
} elseif ($Mode -eq 'formal') {
    $Placements = 'K24:T0 K24:T3 K0:T0 K24:T4'
    $Policies = 'first_sentence'
    if ($Runs -lt 1) { $Runs = 5 }
} else {
    if (-not $RunId) {
        $Latest = Get-ChildItem -LiteralPath (Join-Path $Project $Root) -Directory |
            Where-Object { $_.Name -like 's0_formal_*' } |
            Sort-Object Name -Descending | Select-Object -First 1
        if (-not $Latest) { throw '找不到正式批次目录，请用 -RunId 指定。' }
        $RunId = $Latest.Name
    }
    Push-Location -LiteralPath $Project
    try {
        $Analysis = & $WindowsPython -m experiments.elastic_residency.analyze_s0_matrix `
            --run-dir (Join-Path $Root $RunId)
        if ($LASTEXITCODE -ne 0) { throw '离线复算失败，请保留错误信息。' }
        Write-Output $Analysis
    } finally { Pop-Location }
    return
}

if ($Runs -lt 1) { throw "轮次必须为正整数，当前为 $Runs。" }
$Stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fffffff'
$RunId = "s0_${Mode}_$Stamp"
$Continue = if ($FailFast) { '' } else { '--continue-on-failure' }

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
$WslPython -m experiments.elastic_residency.s0_pipeline.s0_parent \
  --placements $Placements \
  --startup-policies $Policies \
  --runs $Runs \
  --enable-playback \
  --run-id '$RunId' \
  $Continue
status=`$?
mkdir -p '$Root/$RunId'
printf 'commit=%s\ndirty=%s\nstatus=%s\n' '$Commit' '$Dirty' "`$status" > '$Root/$RunId/provenance.txt' || true
exit `$status
"@
    $Command = $Command.Replace("`r`n", "`n").Replace("`r", "`n")
    if ($PrintOnly) { Write-Output $Command; return }
    Write-Host "S0 $Mode：placements=[$Placements]，policies=[$Policies]，runs=[$Runs]。"
    Write-Host '协议 s0_protocol_v1：调度主指标构成 Gate B 判据，设备体验指标在时钟验证前保持 pending。'
    wsl -e bash -lc $Command
    if ($LASTEXITCODE -ne 0) { throw "S0 批次 $RunId 未通过，请保留结果与日志，先复核再决定补测。" }
    Write-Host "结果：$Root/$RunId；离线复算：.\scripts\experiments\run_s0_pipeline.ps1 -Mode analyze -RunId $RunId"
} finally { Pop-Location }
