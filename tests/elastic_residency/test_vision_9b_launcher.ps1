# 只验证手动入口的命令契约；任何 WSL 调用都会使测试失败。
$ErrorActionPreference = 'Stop'
$Launcher = (Resolve-Path (Join-Path $PSScriptRoot '../../scripts/experiments/run_vision_9b_smoke.ps1')).Path
function wsl { throw '静态验证禁止调用 WSL。' }
function Assert-Contains([string]$Text, [string]$Expected) {
    if (-not $Text.Contains($Expected)) { throw "缺少命令契约：$Expected" }
}
function Assert-BashLines([string]$Text) {
    if ($Text.Contains("`r")) { throw 'Bash 命令混入 CR 换行符。' }
    if (($Text -split "`n")[0] -cne 'set -euo pipefail') { throw 'Bash 首行选项错误。' }
}
$Cases = @(
    @{ Stage = 'Smoke'; Runs = 1; Tokens = 128; States = 'TEXT_ONLY_CHAT VISION_WARM_GPU' },
    @{ Stage = 'Answer'; Runs = 1; Tokens = 1024; States = 'VISION_WARM_GPU' },
    @{ Stage = 'Profile'; Runs = 5; Tokens = 128; States = 'VISION_WARM_GPU' }
)
Push-Location -LiteralPath $env:TEMP
try {
    foreach ($Case in $Cases) {
        $Command = (& $Launcher -PrintOnly -ModelVariant Large -Stage $Case.Stage) -join "`n"
        Assert-BashLines $Command
        Assert-Contains $Command ('--run-id ''v1_vision_9b_' + $Case.Stage.ToLowerInvariant() + '_')
        Assert-Contains $Command "--runs $($Case.Runs) --context-tokens 512 --generation-tokens $($Case.Tokens)"
        Assert-Contains $Command "--residency-states $($Case.States) --runs"
        Assert-Contains $Command "--model '/home/leslie/mvp_core/models/llm/Qwen3.5-9B-q4_k_m.gguf'"
        Assert-Contains $Command "--mmproj '/home/leslie/mvp_core/models/llm/Qwen3.5-9B-f16.mmproj'"
        Assert-Contains $Command '--expect-model-sha256'
        Assert-Contains $Command '--expect-mmproj-sha256'
        Assert-Contains $Command '--source-git-snapshot'
        Assert-Contains $Command 'test ! -e '
        if ($Command.Contains('/mnt/d/AI/xiaoyou-core/models/llm')) { throw '误用 D 盘模型路径。' }
    }
    $Default = (& $Launcher -PrintOnly -RunId 'static_custom_id' -Config K20) -join "`n"
    Assert-Contains $Default "--run-id 'static_custom_id'"
    Assert-Contains $Default "--configs 'K20'"
    Assert-Contains $Default '--runs 1 --context-tokens 512 --generation-tokens 128'
    Assert-Contains $Default "--model '/home/leslie/mvp_core/models/llm/comparison_1a35ab89/model.gguf'"
    Assert-Contains $Default "--expect-model-sha256 '1a35ab892c0c4eec814d6594d34461193593a19110cd1d1285414efceb8a26ab'"
    foreach ($Case in $Cases) {
        $Small = (& $Launcher -PrintOnly -ModelVariant Small -Stage $Case.Stage) -join "`n"
        Assert-BashLines $Small
        Assert-Contains $Small ('--run-id ''v1_vision_9b_small_' + $Case.Stage.ToLowerInvariant() + '_')
        Assert-Contains $Small "--model '/home/leslie/mvp_core/models/llm/comparison_1a35ab89/model.gguf'"
        Assert-Contains $Small "--expect-model-sha256 '1a35ab892c0c4eec814d6594d34461193593a19110cd1d1285414efceb8a26ab'"
        Assert-Contains $Small "--expect-mmproj-sha256 'e36da9b9891e13a172b25028ad18b4fc63bc4d681cf391a44b2a9a3b51bc797a'"
        Assert-Contains $Small "--runs $($Case.Runs) --context-tokens 512 --generation-tokens $($Case.Tokens)"
        if ($Small.Contains('423fb3e5')) { throw '小版入口混入大版哈希。' }
    }
    foreach ($Invalid in @(@{Stage='Unknown'}, @{RunId='bad;command'}, @{Config='K33'}, @{ModelVariant='Other'})) {
        $Rejected = $false
        try { & $Launcher -PrintOnly @Invalid | Out-Null } catch { $Rejected = $true }
        if (-not $Rejected) { throw '非法参数未被拒绝。' }
    }
    # 使用两种实际文件换行编码，防止仅在当前 checkout 为 LF 时碰巧通过。
    $Project = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
    $FixtureRoot = Join-Path $Project ('.test_tmp/launcher_lines_' + [guid]::NewGuid().ToString('N'))
    $FixtureScripts = Join-Path $FixtureRoot 'scripts/experiments'
    New-Item -ItemType Directory -Path $FixtureScripts -Force | Out-Null
    foreach ($Name in @('run_vision_9b_smoke.ps1', 'run_9b_comparison.ps1')) {
        $Source = [IO.File]::ReadAllText((Join-Path $Project "scripts/experiments/$Name")).Replace("`r`n", "`n")
        foreach ($Ending in @("`n", "`r`n")) {
            $Fixture = Join-Path $FixtureScripts $Name
            [IO.File]::WriteAllText($Fixture, $Source.Replace("`n", $Ending))
            $Options = @{ PrintOnly = $true }
            if ($Name -eq 'run_vision_9b_smoke.ps1') {
                $Options.ModelVariant = 'Small'
                $Options.Stage = 'Answer'
            }
            $Text = (& $Fixture @Options) -join "`n"
            Assert-BashLines $Text
        }
    }
    Write-Output 'PASS：两模型三阶段、路径/哈希/来源、非法参数及两启动器 LF/CRLF 回归；未调用 WSL。'
} finally {
    Pop-Location
}
