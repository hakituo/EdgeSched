# 只验证 S0 启动器的命令契约；任何 WSL 调用都会使测试失败。
$ErrorActionPreference = 'Stop'
$Launcher = (Resolve-Path (Join-Path $PSScriptRoot '../../scripts/experiments/run_s0_pipeline.ps1')).Path
function wsl { throw '静态验证禁止调用 WSL。' }
function Assert-Contains([string]$Text, [string]$Expected) {
    if (-not $Text.Contains($Expected)) { throw "缺少命令契约：$Expected" }
}
function Assert-BashLines([string]$Text) {
    if ($Text.Contains("`r")) { throw 'Bash 命令混入 CR 换行符。' }
    if (($Text -split "`n")[0] -cne 'set -uo pipefail') { throw 'Bash 首行选项错误。' }
}
Push-Location -LiteralPath $env:TEMP
try {
    $Startup = (& $Launcher -Mode startup -PrintOnly) -join "`n"
    Assert-BashLines $Startup
    Assert-Contains $Startup '--placements K24:T0 \'
    Assert-Contains $Startup '--startup-policies first_sentence full_prebuffer'
    Assert-Contains $Startup '--runs 1'
    Assert-Contains $Startup '--enable-playback'
    Assert-Contains $Startup "'s0_startup_"
    Assert-Contains $Startup 'provenance.txt'

    $Formal = (& $Launcher -Mode formal -PrintOnly) -join "`n"
    Assert-BashLines $Formal
    Assert-Contains $Formal '--placements K24:T0 K24:T3 K0:T0 K24:T4'
    Assert-Contains $Formal '--startup-policies first_sentence'
    Assert-Contains $Formal '--runs 5'
    Assert-Contains $Formal '--continue-on-failure'
    Assert-Contains $Formal "'s0_formal_"

    $FailFast = (& $Launcher -Mode formal -Runs 2 -FailFast -PrintOnly) -join "`n"
    Assert-Contains $FailFast '--runs 2'
    if ($FailFast.Contains('--continue-on-failure')) { throw 'FailFast 模式不应追加 continue-on-failure。' }

    $Rejected = $false
    try { & $Launcher -Mode unknown -PrintOnly | Out-Null } catch { $Rejected = $true }
    if (-not $Rejected) { throw '非法 Mode 未被拒绝。' }
    Write-Output 'PASS：startup/formal/FailFast 命令契约与 provenance 写入；未调用 WSL。'
} finally {
    Pop-Location
}
