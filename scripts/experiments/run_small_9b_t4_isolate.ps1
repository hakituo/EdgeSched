# 用户手动运行：T4（TTS 全 CPU）单文本隔离 smoke，采集 S0 需要的 GPU 隔离增量先验。
param([switch]$PrintOnly)
$ErrorActionPreference = 'Stop'
$Project = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$Stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fffffff'
$TtsModel = '/home/leslie/mvp_core/models/Qwen3-TTS-12Hz-0.6B-Base'
$Ref = '/home/leslie/mvp_core/models/tts_refs/ref_calm.wav'
$Root = '/home/leslie/mvp_core/experiment_results/elastic_residency'
$Snapshot = "$Root/vision_inputs/source_snapshots/small_t4_isolate_$Stamp.tar.gz"
# 旧后端脚本：需要上游 qwen_tts（transformers 4.x）。遗留环境已删除，
# 重跑需自建环境并把路径传给本变量（或改脚本参数）。
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
mkdir -p '$Root/vision_inputs/source_snapshots'
test ! -e '$Snapshot'
tar --exclude='__pycache__' --exclude='*.pyc' -czf '$Snapshot' experiments
SNAP_SHA=`$(sha256sum '$Snapshot' | cut -d' ' -f1)
$Python -m experiments.elastic_residency.tts_component_offload --model '$TtsModel' --ref-audio '$Ref' --configs T4 --seed 42 --output-root '$Root/b1' --run-id 'b1_small_t4_cpu_smoke_$Stamp' --source-git-commit '$Commit' --source-git-dirty '$Dirty' --source-git-status-entry-count $Count --source-git-snapshot '$Snapshot' --source-git-snapshot-sha256 `$SNAP_SHA
"@
    $Command = $Command.Replace("`r`n", "`n").Replace("`r", "`n")
    if ($PrintOnly) { Write-Output $Command; return }
    Write-Host 'T4 隔离 smoke：TTS talker/codec/speaker 全 CPU，单文本单配置，失败即停。'
    Write-Host '目的：采集 T4 的 GPU 隔离增量与 CPU talker 耗时量级，供 K24:T4 共置批次使用。'
    wsl -e bash -lc $Command
    if ($LASTEXITCODE -ne 0) { throw "T4 隔离 smoke $Stamp 未通过，请保留日志先复核。" }
    Write-Host "结果：$Root/b1/b1_small_t4_cpu_smoke_$Stamp"
} finally { Pop-Location }
