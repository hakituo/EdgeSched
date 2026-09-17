# 只验证 TTS 后端对照启动器的命令契约；任何 WSL 调用都会使测试失败。
$ErrorActionPreference = 'Stop'
$Launcher = (Resolve-Path (Join-Path $PSScriptRoot '../../scripts/experiments/run_tts_backend_compare.ps1')).Path
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
    $Both = (& $Launcher -Mode both -PrintOnly) -join "`n"
    Assert-BashLines $Both
    Assert-Contains $Both '-m experiments.tts.profile_qwen3_tts_cudagraph'
    Assert-Contains $Both '--paths upstream_dynamic cuda_graph_static'
    Assert-Contains $Both '--attn-implementation eager'
    Assert-Contains $Both '--rounds 3'
    Assert-Contains $Both "'tts_backend_both_eager_"
    Assert-Contains $Both 'provenance.txt'

    $Baseline = (& $Launcher -Mode baseline -Attn sdpa -Rounds 5 -PrintOnly) -join "`n"
    Assert-BashLines $Baseline
    Assert-Contains $Baseline '--paths upstream_dynamic'
    Assert-Contains $Baseline '--attn-implementation sdpa'
    Assert-Contains $Baseline '--rounds 5'

    $Graph = (& $Launcher -Mode graph -RunId static_graph_only -PrintOnly) -join "`n"
    Assert-BashLines $Graph
    Assert-Contains $Graph '--paths cuda_graph_static'
    Assert-Contains $Graph "--run-id 'static_graph_only'"

    $B2 = (& $Launcher -Mode b2 -RunId b2_tts_t0_test -PrintOnly) -join "`n"
    Assert-BashLines $B2
    Assert-Contains $B2 '--paths cuda_graph_static'
    Assert-Contains $B2 '--rounds 5'
    Assert-Contains $B2 "--output-root 'experiment_results/elastic_residency/b2'"
    Assert-Contains $B2 '--sequence correctness'
    Assert-Contains $B2 "--run-id 'b2_tts_t0_test_state'"
    Assert-Contains $B2 "check_tts_output_correctness --run-dir 'experiment_results/elastic_residency/b2/b2_tts_t0_test'"
    # 主环境路径已软链到加速环境，画像与 ASR 核对同环境
    Assert-Contains $B2 '.venvs/mvp_core-cu128/bin/python'
    Assert-Contains $B2 'status=0'
    # 来源门由主机侧传入；两次画像运行都要带（值随工作树状态变化，只断言标志）
    Assert-Contains $B2 "--source-git-commit '"
    Assert-Contains $B2 "--source-git-dirty '"
    Assert-Contains $B2 '--source-git-status-entry-count'

    foreach ($Invalid in @(
        @{ Mode = 'unknown' },
        @{ Mode = 'both'; RunId = 'bad;command' },
        @{ Mode = 'both'; Rounds = 0 }
    )) {
        $Rejected = $false
        try { & $Launcher -PrintOnly @Invalid | Out-Null } catch { $Rejected = $true }
        if (-not $Rejected) { throw "非法参数未被拒绝：$($Invalid | ConvertTo-Json -Compress)" }
    }
    Write-Output 'PASS：both/baseline/graph、attn 与 rounds 传递、run-id 与 provenance 契约；未调用 WSL。'
} finally {
    Pop-Location
}
