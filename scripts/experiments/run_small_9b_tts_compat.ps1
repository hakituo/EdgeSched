# 用户手动运行：small 9B 保持 LLM/KV 驻留时，验证 T0/T3 合成兼容性。
param([switch]$PrintOnly)
$ErrorActionPreference = 'Stop'
$Project = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$Stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fffffff'
$Model = '/home/leslie/mvp_core/models/llm/comparison_1a35ab89/model.gguf'
$Sha = '1a35ab892c0c4eec814d6594d34461193593a19110cd1d1285414efceb8a26ab'
$Root = '/home/leslie/mvp_core/experiment_results/elastic_residency'
$Snapshot = "$Root/vision_inputs/source_snapshots/small_tts_compat_$Stamp.tar.gz"
$Screen = "$Root/c3_analysis/small_stage_screen_20260907_204516_653987.json"
$ScreenSha = '7fa1b5ae48371d7bddcf7ac8e0138324e2c454d85bd4760bfa2f859284caaf41'
$Python = '/home/leslie/.venvs/mvp_core-cu128/bin/python'
Push-Location -LiteralPath $Project
try {
    if (-not $PrintOnly) {
        & (Join-Path $Project 'scripts/sync_to_wsl2_safe.ps1') -SkipGit
        if ($LASTEXITCODE -ne 0) { throw '同步失败，停止。' }
    }
    $Commit = git rev-parse HEAD
    if ($LASTEXITCODE -ne 0) { throw '读取 Git 提交失败。' }
    $Status = @(git status --porcelain)
    if ($LASTEXITCODE -ne 0) { throw '读取 Git 状态失败。' }
    $Count = $Status.Count
    $Dirty = if ($Count) { 'true' } else { 'false' }
    $Command = @"
set -euo pipefail
cd /home/leslie/mvp_core
printf '%s  %s\n' '$Sha' '$Model' | sha256sum -c -
printf '%s  %s\n' '$ScreenSha' '$Screen' | sha256sum -c -
mkdir -p '$Root/vision_inputs/source_snapshots'
test ! -e '$Snapshot'
tar --exclude='__pycache__' --exclude='*.pyc' -czf '$Snapshot' experiments
$Python -m experiments.elastic_residency.llm_tts_colocated --llm-model '$Model' --tts-model '/home/leslie/mvp_core/models/Qwen3-TTS-12Hz-0.6B-Base' --ref-audio '/home/leslie/mvp_core/models/tts_refs/ref_calm.wav' --candidates 512:K18:T0 512:K18:T3 --runs 1 --generation-tokens 128 --max-new-tokens 256 --llm-cores 14 --system-cores 2 --seed 42 --safety-margin-mb 915.015625 --tts-isolated-increment-mb T0=3255.03515625 T3=1979.09765625 --source-git-commit '$Commit' --source-git-dirty '$Dirty' --source-git-status-entry-count $Count --source-git-snapshot '$Snapshot' --output-root '$Root/llm_tts_colocated' --run-id 'small_9b_tts_compat_$Stamp'
"@
    $Command = $Command.Replace("`r`n", "`n").Replace("`r", "`n")
    if ($PrintOnly) { Write-Output $Command; return }
    Write-Host 'small 9B：C512/K18 + T0、C512/K18 + T3，各 1 轮，失败即停。'
    Write-Host 'LLM 填充并保留 640 个 KV 位置后合成 TTS；这是驻留兼容性，不是并行生成或 SM 性能测试。'
    Write-Host '本批不加载视觉或 Diffusion。T3 的 talker 仍在 GPU，codec 在 CPU。'
    wsl -e bash -lc $Command
    if ($LASTEXITCODE -ne 0) { throw "兼容性批次 $Stamp 未通过，请保留结果与日志，先复核再决定补测。" }
    Write-Host "结果：$Root/llm_tts_colocated/small_9b_tts_compat_$Stamp"
} finally { Pop-Location }
