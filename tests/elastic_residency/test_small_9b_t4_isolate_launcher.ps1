# 无模型验证：从非项目目录执行，并阻止意外启动 WSL。
$ErrorActionPreference = 'Stop'
function wsl { throw '静态验证禁止调用 WSL。' }
$Launcher = (Resolve-Path (Join-Path $PSScriptRoot '../../scripts/experiments/run_small_9b_t4_isolate.ps1')).Path
Push-Location $env:TEMP
try {
    $Command = (& $Launcher -PrintOnly) -join "`n"
    if ($Command.Contains("`r")) { throw 'Bash 命令存在 CR。' }
    foreach ($Expected in @('set -euo pipefail', '--configs T4', '--seed 42',
        'Qwen3-TTS-12Hz-0.6B-Base', '--source-git-snapshot-sha256',
        'b1_small_t4_cpu_smoke_')) {
        if (-not $Command.Contains($Expected)) { throw "缺少契约：$Expected" }
    }
    foreach ($Forbidden in @('--configs T0', 'K24', 'K18', 'llm_tts_colocated')) {
        if ($Command.Contains($Forbidden)) { throw "错误范围：$Forbidden" }
    }
    Write-Output 'PASS：T4 隔离 smoke 启动器固定 T4/seed/快照 SHA，未运行 WSL。'
} finally { Pop-Location }
