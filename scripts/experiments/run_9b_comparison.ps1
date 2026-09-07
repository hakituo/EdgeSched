param(
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$')]
    [string]$RunId = ('compare_9b_' + (Get-Date -Format 'yyyyMMdd_HHmmss')),
    [ValidateRange(1, 5)]
    [int]$Repeats = 3,
    [ValidateSet('FULL', 'K0')]
    [string[]]$Configs = @('FULL', 'K0'),
    [switch]$PrintOnly
)

$ErrorActionPreference = 'Stop'
$ProjectPath = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
if (($Configs | Select-Object -Unique).Count -ne $Configs.Count) { throw '配置不能重复。' }
if ($Configs.Count -eq 0) { throw '配置不能为空。' }
Push-Location -LiteralPath $ProjectPath
try {
    if (-not $PrintOnly) {
        & (Join-Path $ProjectPath 'scripts/sync_to_wsl2_safe.ps1') -SkipGit
        if ($LASTEXITCODE -ne 0) { throw '代码同步失败。' }
    }
    $SourceCommit = git rev-parse HEAD
    if ($LASTEXITCODE -ne 0) { throw '读取源提交失败。' }
    $SourceStatus = @(git status --porcelain)
    if ($LASTEXITCODE -ne 0) { throw '读取 Git 状态失败。' }
    $SourceCount = $SourceStatus.Count
    $SourceDirty = if ($SourceCount -gt 0) { 'true' } else { 'false' }
    $ConfigArgs = $Configs -join ' '
    $WslCommand = @"
set -euo pipefail
cd /home/leslie/mvp_core
/home/leslie/.venvs/mvp_core-cu128/bin/python -m experiments.elastic_residency.compare_9b_models --run-id '$RunId' --repeats $Repeats --configs $ConfigArgs --source-git-commit '$SourceCommit' --source-git-dirty '$SourceDirty' --source-git-status-entry-count $SourceCount
"@
    # PowerShell here-string 继承脚本的 CRLF；传给 Bash 前必须统一为 LF。
    $WslCommand = $WslCommand.Replace("`r`n", "`n").Replace("`r", "`n")
    if ($PrintOnly) { Write-Output $WslCommand; return }
    Write-Host "Run ID: $RunId；共 $(2 * $Configs.Count * $Repeats) 个单元。"
    Write-Host '先将 Windows 小模型复制到 WSL 独立目录并核验；两个模型串行测量，不同时加载。'
    Write-Host '比较纯文本 CPU/GPU 端点；含固定生成、KV、显存/RSS/swap，未加载视觉组件。'
    wsl -e bash -lc $WslCommand
    if ($LASTEXITCODE -ne 0) { throw '对比未通过，已停止；保留当前批次结果和日志。' }
} finally {
    Pop-Location
}
