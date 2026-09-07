# 无模型验证：WSL 调用一律失败，核对完整矩阵及换行。
$ErrorActionPreference = 'Stop'
function wsl { throw '验证禁止调用 WSL。' }
$Launcher = (Resolve-Path (Join-Path $PSScriptRoot '../../scripts/experiments/run_9b_small_curves.ps1')).Path
Push-Location $env:TEMP
try {
    $Command = (& $Launcher -PrintOnly) -join "`n"
    if ($Command.Contains("`r")) { throw 'Bash 命令存在 CR。' }
    foreach ($Expected in @('set -euo pipefail', 'sha256sum -c -', 'comparison_1a35ab89/model.gguf',
        '--configs K0 K8 K16 K24 K32 FULL --runs 5',
        '--configs K2 K4 K6 K10 K12 K14 K18 K20 K28 K30 --runs 5',
        '--configs K0 K2 K4 K6 K8 K10 K12 K14 K16 K18 K20 K24 K28 K30 K32 FULL --runs 1')) {
        if (-not $Command.Contains($Expected)) { throw "缺少契约：$Expected" }
    }
    if (($Command.Split('--contexts 512 2048 4096').Count - 1) -ne 3) { throw '三阶段上下文不一致。' }
    if ($Command.Contains('--continue-on-failure') -or $Command.Contains('--mmproj')) { throw '错误运行范围。' }
    Write-Output 'PASS：小版身份、C2/C3/KV 矩阵、LF 和 fail-fast；未运行 WSL。'
} finally { Pop-Location }
