param(
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$')]
    [string]$RunId = ('d1_bge_cached_' + (Get-Date -Format 'yyyyMMdd_HHmmss')),
    [switch]$PrintOnly
)

$ErrorActionPreference = 'Stop'
$ProjectPath = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
Push-Location -LiteralPath $ProjectPath
try {
    # 仅打印时不进行同步，也不启动任何 WSL 或模型进程。
    if (-not $PrintOnly) {
        & (Join-Path $ProjectPath 'scripts/sync_to_wsl2_safe.ps1') -SkipGit
        if ($LASTEXITCODE -ne 0) { throw 'WSL 同步失败，停止运行。' }
    }
    $SourceCommit = (git rev-parse HEAD)
    if ($LASTEXITCODE -ne 0) { throw '读取 Git 提交失败。' }
    $SourceStatus = @(git status --porcelain)
    if ($LASTEXITCODE -ne 0) { throw '读取 Git 工作区状态失败。' }
    $SourceCount = $SourceStatus.Count
    $SourceDirty = if ($SourceCount -gt 0) { 'true' } else { 'false' }
    $WslCommand = @"
cd /home/leslie/mvp_core && /home/leslie/.venvs/mvp_core-cu128/bin/python -m experiments.elastic_residency.auxiliary_core_budget --run-id $RunId --tasks bge_cached --inputs experiment_results/elastic_residency/auxiliary_inputs/d1_chinese_tts_crops_v1/inputs.json --system-cores 2 --repeats 5 --source-git-commit $SourceCommit --source-git-dirty $SourceDirty --source-git-status-entry-count $SourceCount
"@
    if ($PrintOnly) {
        Write-Output $WslCommand
        return
    }
    Write-Host "Run ID: $RunId"
    Write-Host '仅运行 BGE 候选缓存模式：1/2/4 核，各五轮，共 15 次；不加载 LLM。'
    wsl -e bash -lc $WslCommand
    if ($LASTEXITCODE -ne 0) { throw '实验未正常完成，请保留终端输出与失败记录。' }
} finally {
    Pop-Location
}
