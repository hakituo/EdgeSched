param(
    [switch]$SkipGit
)

$ErrorActionPreference = "Stop"

$SourceWinPath = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$DriveLetter = $SourceWinPath.Substring(0, 1).ToLower()
$PathRemainder = $SourceWinPath.Substring(2) -replace "\\", "/"
$SourceWslPath = "/mnt/$DriveLetter$PathRemainder"

$DestWslPath = "/home/leslie/mvp_core"

Write-Host "Source (Windows): $SourceWinPath"
Write-Host "Source (WSL):     $SourceWslPath/"
Write-Host "Dest (WSL):       $DestWslPath/"
Write-Host ""
Write-Host "Policy: sync code only; do not touch models/experiment_results/output/Forge; no deletion."
Write-Host "Warning: do NOT use rsync --delete or robocopy /MIR to WSL2, it may delete WSL2 models."
Write-Host ""

$RsyncCmd = @"
set -euo pipefail
mkdir -p '$DestWslPath'
rsync -a --info=stats2 \
  --exclude '/models/' \
  --exclude '/experiment_results/' \
  --exclude '/output/' \
  --exclude '/stable-diffusion-webui-forge-main/' \
  --exclude '.git/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.pytest_cache/' \
  --exclude '.ruff_cache/' \
  --exclude '.test_tmp*/' \
  '$SourceWslPath/' '$DestWslPath/'
"@

wsl -e bash -lc $RsyncCmd

if ($SkipGit) {
    Write-Host ""
    Write-Host "Step 2: skipped Git commit/push (-SkipGit)."
    Write-Host ""
    Write-Host "Done: synced to WSL2 safely (no deletion on WSL2 side)."
    exit 0
}

Write-Host ""
Write-Host "Step 2: auto-commit & push to GitHub (EdgeSched)..."

Push-Location $SourceWinPath
try {
    git add -A
    $HasChanges = (git status --porcelain | Measure-Object -Line).Lines
    if ($HasChanges -gt 0) {
        $Stamp = [DateTime]::Now.ToString("yyyy-MM-dd HH:mm")
        git -c user.name=hakituo -c user.email=hakituo@users.noreply.github.com commit -m "auto: 同步后自动提交（$Stamp）"
        git push origin main
        Write-Host "Auto-commit & push OK."
    } else {
        Write-Host "No changes to commit, skip push."
    }
} finally {
    Pop-Location
}

Write-Host ""
Write-Host "Done: synced to WSL2 safely (no deletion on WSL2 side) and pushed to GitHub."
