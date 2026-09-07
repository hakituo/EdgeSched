param(
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$')]
    [string]$RunId,
    [ValidateSet('Smoke', 'Answer', 'Profile')]
    [string]$Stage = 'Smoke',
    [ValidateSet('Large', 'Small')]
    [string]$ModelVariant = 'Small',
    [ValidatePattern('^(FULL|K([0-9]|[12][0-9]|3[0-2]))$')]
    [string]$Config = 'FULL',
    [switch]$PrintOnly
)

$ErrorActionPreference = 'Stop'
$ProjectPath = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$Models = @{
    Large = @{ Path = '/home/leslie/mvp_core/models/llm/Qwen3.5-9B-q4_k_m.gguf'; Sha = '423fb3e5fe2c98df616159b7cd1bd04765f67667f5f8712a81e8f63f4e7ee1ef'; Prefix = 'v1_vision_9b_' }
    Small = @{ Path = '/home/leslie/mvp_core/models/llm/comparison_1a35ab89/model.gguf'; Sha = '1a35ab892c0c4eec814d6594d34461193593a19110cd1d1285414efceb8a26ab'; Prefix = 'v1_vision_9b_small_' }
}
$ModelPath = $Models[$ModelVariant].Path
$ModelSha = $Models[$ModelVariant].Sha
$ProjectorPath = '/home/leslie/mvp_core/models/llm/Qwen3.5-9B-f16.mmproj'
# 完整回答与固定长度画像分开，避免把不同生成预算合成性能曲线。
$StageSettings = @{
    Smoke = @{ Runs = 1; Tokens = 128; States = 'TEXT_ONLY_CHAT VISION_WARM_GPU'; Units = 2 }
    Answer = @{ Runs = 1; Tokens = 1024; States = 'VISION_WARM_GPU'; Units = 1 }
    Profile = @{ Runs = 5; Tokens = 128; States = 'VISION_WARM_GPU'; Units = 5 }
}
$Settings = $StageSettings[$Stage]
if (-not $PSBoundParameters.ContainsKey('RunId')) {
    $RunId = $Models[$ModelVariant].Prefix + $Stage.ToLowerInvariant() + '_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
}
$RunCount = $Settings.Runs
$GenerationTokens = $Settings.Tokens
$ResidencyStates = $Settings.States
Push-Location -LiteralPath $ProjectPath
try {
    if (-not $PrintOnly) {
        & (Join-Path $ProjectPath 'scripts/sync_to_wsl2_safe.ps1') -SkipGit
        if ($LASTEXITCODE -ne 0) { throw 'WSL 代码同步失败。' }
    }
    $SourceCommit = git rev-parse HEAD
    if ($LASTEXITCODE -ne 0) { throw '读取 Git 提交失败。' }
    $SourceStatus = @(git status --porcelain)
    if ($LASTEXITCODE -ne 0) { throw '读取 Git 状态失败。' }
    $SourceCount = $SourceStatus.Count
    $SourceDirty = if ($SourceCount -gt 0) { 'true' } else { 'false' }
    $SnapshotDir = '/home/leslie/mvp_core/experiment_results/elastic_residency/vision_inputs/source_snapshots'
    $SnapshotPath = "$SnapshotDir/$RunId.tar.gz"
    # 使用已校验的 WSL 原生模型副本，文件存在性在 WSL 内检查。
    # 工作区未提交时冻结实际 WSL 实验源码，避免只传 dirty 标志导致来源门禁失败。
    $WslCommand = @"
set -euo pipefail
cd /home/leslie/mvp_core
test -f '$ModelPath' || { echo 'WSL 9B 主模型不存在：$ModelPath' >&2; exit 1; }
test -f '$ProjectorPath' || { echo 'WSL 9B 视觉文件不存在：$ProjectorPath' >&2; exit 1; }
mkdir -p '$SnapshotDir'
test ! -e '$SnapshotPath'
tar --exclude='__pycache__' --exclude='*.pyc' -czf '$SnapshotPath' experiments
/home/leslie/.venvs/mvp_core-cu128/bin/python -m experiments.elastic_residency.vision.vision_endpoint_smoke --run-id '$RunId' --model '$ModelPath' --mmproj '$ProjectorPath' --expect-model-sha256 '$ModelSha' --expect-mmproj-sha256 'e36da9b9891e13a172b25028ad18b4fc63bc4d681cf391a44b2a9a3b51bc797a' --configs '$Config' --residency-states $ResidencyStates --runs $RunCount --context-tokens 512 --generation-tokens $GenerationTokens --vision-eval-mode split_encode --source-git-commit '$SourceCommit' --source-git-dirty '$SourceDirty' --source-git-status-entry-count $SourceCount --source-git-snapshot '$SnapshotPath'
"@
    # PowerShell here-string 继承脚本的 CRLF；传给 Bash 前必须统一为 LF。
    $WslCommand = $WslCommand.Replace("`r`n", "`n").Replace("`r", "`n")
    if ($PrintOnly) {
        Write-Output $WslCommand
        return
    }
    Write-Host "Run ID: $RunId"
    Write-Host "9B ($ModelVariant) + 9B projector，$Stage / $Config，共 $($Settings.Units) 次，固定生成 $GenerationTokens tokens。"
    if ($Stage -eq 'Answer') {
        Write-Host '完整回答补测：仍按固定预算生成，遇 EOS 不停止；实际内容需人工复核，不能作为正式性能画像。'
    }
    if ($Stage -eq 'Profile') {
        Write-Host '5 个独立子进程，每轮加载并 warm projector 后实际编码图片；不是同一常驻会话的连续请求或 TTS 共置。'
    }
    Write-Host '模型从 WSL 原生目录读取、推理在 WSL 执行；这是端点试跑，不是 TTS/Forge 共置验收。'
    wsl -e bash -lc $WslCommand
    if ($LASTEXITCODE -ne 0) { throw '试跑未通过，请保留终端输出及本次产物。' }
} finally {
    Pop-Location
}
