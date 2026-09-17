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
    Assert-Contains $Startup '--tts-backend cuda_graph_static'
    Assert-Contains $Startup "'s0_startup_"
    Assert-Contains $Startup 'provenance.txt'
    # 主环境路径已软链到加速环境：S0 同进程共置 LLM 与加速 TTS
    Assert-Contains $Startup '.venvs/mvp_core-cu128/bin/python'

    $Formal = (& $Launcher -Mode formal -PrintOnly) -join "`n"
    Assert-BashLines $Formal
    # s0_protocol_v2：TTS 放置只剩 T0，T3/T4 已按后端能力排除
    Assert-Contains $Formal '--placements K24:T0 K0:T0'
    Assert-Contains $Formal '--startup-policies first_sentence'
    Assert-Contains $Formal '--runs 5'
    Assert-Contains $Formal '--continue-on-failure'
    Assert-Contains $Formal "'s0_formal_"

    # 遗留环境已删：upstream_dynamic 没有默认解释器，不给 -Python 必须拒绝
    $LegacyRejected = $false
    try { & $Launcher -Mode formal -TtsBackend upstream_dynamic -PrintOnly | Out-Null } catch { $LegacyRejected = $true }
    if (-not $LegacyRejected) { throw 'upstream_dynamic 未要求显式 -Python。' }

    $Legacy = (& $Launcher -Mode formal -TtsBackend upstream_dynamic -Python /custom/legacy-python -PrintOnly) -join "`n"
    Assert-Contains $Legacy '--tts-backend upstream_dynamic'
    Assert-Contains $Legacy '/custom/legacy-python'

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
