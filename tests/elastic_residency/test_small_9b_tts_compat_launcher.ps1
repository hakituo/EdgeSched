# 无模型验证：从非项目目录执行，并阻止意外启动 WSL。
$ErrorActionPreference = 'Stop'
function wsl { throw '静态验证禁止调用 WSL。' }
$Launcher = (Resolve-Path (Join-Path $PSScriptRoot '../../scripts/experiments/run_small_9b_tts_compat.ps1')).Path
Push-Location $env:TEMP
try {
    $Command = (& $Launcher -PrintOnly) -join "`n"
    if ($Command.Contains("`r")) { throw 'Bash 命令存在 CR。' }
    foreach ($Expected in @('set -euo pipefail', 'comparison_1a35ab89/model.gguf',
        '1a35ab892c0c4eec814d6594d34461193593a19110cd1d1285414efceb8a26ab',
        '7fa1b5ae48371d7bddcf7ac8e0138324e2c454d85bd4760bfa2f859284caaf41',
        '--candidates 512:K18:T0 512:K18:T3 --runs 1',
        '--generation-tokens 128', '--llm-cores 14 --system-cores 2',
        '--safety-margin-mb 915.015625', '--source-git-snapshot',
        'T0=3255.03515625 T3=1979.09765625')) {
        if (-not $Command.Contains($Expected)) { throw "缺少契约：$Expected" }
    }
    if (($Command.Split('sha256sum -c -').Count - 1) -ne 2) { throw '缺少模型或筛选身份校验。' }
    foreach ($Forbidden in @('--continue-on-failure', '--resume', '--mmproj', 'K30', 'K28', '4B')) {
        if ($Command.Contains($Forbidden)) { throw "错误范围：$Forbidden" }
    }
    $K20 = (& $Launcher -PrintOnly -KConfig 'K20') -join "`n"
    if (-not $K20.Contains('--candidates 512:K20:T0 512:K20:T3')) { throw '驻留档参数未生效。' }
    if ($K20.Contains('512:K18')) { throw '默认档位未被替换。' }
    $Rejected = $false
    try { & $Launcher -PrintOnly -KConfig 'K30' | Out-Null } catch { $Rejected = $true }
    if (-not $Rejected) { throw '超过 K24 的驻留档未被拒绝。' }
    $FiveRuns = (& $Launcher -PrintOnly -KConfig 'K24' -Runs 5) -join "`n"
    if (-not $FiveRuns.Contains('--candidates 512:K24:T0 512:K24:T3 --runs 5')) { throw '正式轮次参数未生效。' }
    $BadRuns = $false
    try { & $Launcher -PrintOnly -Runs 0 | Out-Null } catch { $BadRuns = $true }
    if (-not $BadRuns) { throw '非法轮次未被拒绝。' }
    Write-Output 'PASS：非项目目录、固定 small/screen 身份、两单元、LF、fail-fast 与驻留档上限；未运行 WSL。'
} finally { Pop-Location }
