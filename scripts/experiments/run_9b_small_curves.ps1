# 用户手动启动：依次重建小版 9B 的 C2、C3、KV，失败立即停止。
param([switch]$PrintOnly)
$ErrorActionPreference = 'Stop'
$Project = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$Stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$Model = '/home/leslie/mvp_core/models/llm/comparison_1a35ab89/model.gguf'
$Sha = '1a35ab892c0c4eec814d6594d34461193593a19110cd1d1285414efceb8a26ab'
$Root = '/home/leslie/mvp_core/experiment_results/elastic_residency'
$Snapshot = "$Root/vision_inputs/source_snapshots/small_curves_$Stamp.tar.gz"
# 主环境（已软链到加速隔离环境：llama_cpp CUDA + transformers 5.x + 加速 TTS）
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
    $Common = "--model '$Model' --contexts 512 2048 4096 --generation-tokens 128 --llm-cores 14 --system-cores 2 --seed 42 --source-git-commit '$Commit' --source-git-dirty '$Dirty' --source-git-status-entry-count $Count --source-git-snapshot '$Snapshot'"
    $Command = @"
set -euo pipefail
cd /home/leslie/mvp_core
printf '%s  %s\n' '$Sha' '$Model' | sha256sum -c -
mkdir -p '$Root/vision_inputs/source_snapshots'
test ! -e '$Snapshot'
tar --exclude='__pycache__' --exclude='*.pyc' -czf '$Snapshot' experiments
printf 'C2: 90 records\n'
$Python -m experiments.elastic_residency.llm_residency_curve $Common --configs K0 K8 K16 K24 K32 FULL --runs 5 --jitter-gate-mode observe --output-root '$Root/c2' --run-id 'c2_9b_small_$Stamp'
printf 'C3: 150 records\n'
$Python -m experiments.elastic_residency.llm_residency_curve $Common --configs K2 K4 K6 K10 K12 K14 K18 K20 K28 K30 --runs 5 --jitter-gate-mode observe --output-root '$Root/c3' --run-id 'c3_9b_small_$Stamp'
printf 'KV: 48 records\n'
$Python -m experiments.elastic_residency.llm_kv_cache_calibration $Common --configs K0 K2 K4 K6 K8 K10 K12 K14 K16 K18 K20 K24 K28 K30 K32 FULL --runs 1 --output-root '$Root/kv_calibration' --run-id 'kv_9b_small_$Stamp'
"@
    $Command = $Command.Replace("`r`n", "`n").Replace("`r", "`n")
    if ($PrintOnly) { Write-Output $Command; return }
    Write-Host "小版 9B，批次 $Stamp：C2 90 + C3 150 + KV 48，依次运行，失败即停。"
    Write-Host '纯文本驻留与 KV 画像；不加载视觉文件，不是视觉/共置实验。'
    wsl -e bash -lc $Command
    if ($LASTEXITCODE -ne 0) { throw "批次 $Stamp 未完整通过；保留结果，先复核失败原因，不要反复重跑。" }
    Write-Host "完成。结果：$Root 下 c2/c2_9b_small_$Stamp、c3/c3_9b_small_$Stamp、kv_calibration/kv_9b_small_$Stamp"
} finally {
    Pop-Location
}
