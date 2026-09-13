param(
    [switch]$SkipGit,
    [string]$Message = "",
    [string]$Detail = ""
)

$ErrorActionPreference = "Stop"

# 无信息量的提交信息黑名单：这类文案无法回答"这次提交做了什么"，一律拒绝
$MeaninglessMessages = @(
    "auto", "auto commit", "autocommit", "update", "updates", "wip",
    "同步", "同步后自动提交", "自动提交", "提交", "更新", "修改", "日常提交", "备份"
)

function Test-MeaninglessMessage {
    param([string]$Text)

    $Normalized = ($Text -replace "\s", "").ToLowerInvariant()
    if ($Normalized.Length -lt 4) {
        return $true
    }
    return $MeaninglessMessages -contains $Normalized
}

function Assert-NativeCommandSucceeded {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Step
    )

    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with exit code $LASTEXITCODE."
    }
}

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

# 提交信息校验放在同步之前：没有可读的提交说明就不提交，也不浪费一次 rsync
$Summary = ($Message | Out-String).Trim()
if (-not $SkipGit) {
    if (-not $Summary) {
        Write-Host "[错误] 自动提交必须提供 -Message，写清楚本次改动做了什么。" -ForegroundColor Red
        Write-Host "示例: .\scripts\sync_to_wsl2_safe.ps1 -Message '修复 Phase-Aware 阶段切换的显存估算偏差'"
        Write-Host "       .\scripts\sync_to_wsl2_safe.ps1 -Message '重跑 C1 驻留曲线' -Detail '补充 3 组随机种子，替换 01_paper_main 旧数据'"
        Write-Host "只同步不提交请加 -SkipGit。"
        exit 2
    }
    if (Test-MeaninglessMessage -Text $Summary) {
        Write-Host "[错误] 提交信息 '$Summary' 没有说明本次改动做了什么，请改写后重试。" -ForegroundColor Red
        Write-Host "要求：一句话说清改动对象与结果，例如 '把 sync 脚本的自动提交改为必须写提交说明'。"
        exit 2
    }
}

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
Assert-NativeCommandSucceeded -Step "WSL2 rsync"

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
    Assert-NativeCommandSucceeded -Step "git add"
    $StatusLines = @(git status --porcelain)
    Assert-NativeCommandSucceeded -Step "git status"
    $HasChanges = $StatusLines.Count
    if ($HasChanges -gt 0) {
        # 先回显本次要提交的范围，混入无关文件时一眼能看出来
        Write-Host "本次提交 $HasChanges 项变更："
        $StatusLines | Select-Object -First 20 | ForEach-Object { Write-Host "  - $_" }
        if ($HasChanges -gt 20) {
            Write-Host "  ... 还有 $($HasChanges - 20) 项"
        }
        # 主题行写"做了什么"，-Detail 作为正文补细节；文件清单不进 commit（git show --stat 可查）
        $CommitText = "$Summary`n"
        $DetailText = ($Detail | Out-String).Trim()
        if ($DetailText) {
            $CommitText += "`n$DetailText`n"
        }
        $Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
        $MessageFile = Join-Path $env:TEMP ("edgesched_commit_" + [Guid]::NewGuid().ToString("N") + ".txt")
        [System.IO.File]::WriteAllText($MessageFile, $CommitText, $Utf8NoBom)
        try {
            git -c user.name=hakituo -c user.email=hakituo@users.noreply.github.com commit -F $MessageFile
            Assert-NativeCommandSucceeded -Step "git commit"
        } finally {
            Remove-Item -LiteralPath $MessageFile -Force -ErrorAction SilentlyContinue
        }
        git push origin main
        Assert-NativeCommandSucceeded -Step "git push"
        Write-Host "Auto-commit & push OK."
    } else {
        Write-Host "No changes to commit, skip push."
    }
} finally {
    Pop-Location
}

Write-Host ""
Write-Host "Done: synced to WSL2 safely (no deletion on WSL2 side) and pushed to GitHub."
