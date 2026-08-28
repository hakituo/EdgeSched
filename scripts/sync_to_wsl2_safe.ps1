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
  --exclude 'models/' \
  --exclude 'experiment_results' \
  --exclude 'output/' \
  --exclude 'stable-diffusion-webui-forge-main/' \
  '$SourceWslPath/' '$DestWslPath/'
"@

wsl -e bash -lc $RsyncCmd

Write-Host ""
Write-Host "Done: synced to WSL2 safely (no deletion on WSL2 side)."
