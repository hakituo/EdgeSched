param(
    [string]$Message = "",
    [string]$Detail = "",
    [switch]$NoPush
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

$RepoPath = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

Write-Host "仓库（Windows，代码真源）：$RepoPath"
Write-Host "运行环境（本机 Linux）：/home/leslie/mvp_core"
Write-Host ""
Write-Host "分工：Windows 侧只做编辑、静态验证、git 收尾；代码送到 Linux 由 Linux 侧执行"
Write-Host "      bash scripts/sync_from_windows.sh（增量，永不 --delete）。"
Write-Host ""

$Summary = ($Message | Out-String).Trim()
if (-not $Summary) {
    Write-Host "[错误] 必须提供 -Message，写清楚本次改动做了什么。" -ForegroundColor Red
    Write-Host "示例: .\scripts\commit_and_push.ps1 -Message '修复 Phase-Aware 阶段切换的显存估算偏差'"
    Write-Host "       .\scripts\commit_and_push.ps1 -Message '重跑 C1 驻留曲线' -Detail '补充 3 组随机种子，替换 01_paper_main 旧数据'"
    Write-Host "只想提交不推送请加 -NoPush。"
    exit 2
}
if (Test-MeaninglessMessage -Text $Summary) {
    Write-Host "[错误] 提交信息 '$Summary' 没有说明本次改动做了什么，请改写后重试。" -ForegroundColor Red
    Write-Host "要求：一句话说清改动对象与结果，例如 '把同步入口从 WSL2 换成本机 Linux'。"
    exit 2
}

Push-Location $RepoPath
try {
    git add -A
    Assert-NativeCommandSucceeded -Step "git add"
    $StatusLines = @(git status --porcelain)
    Assert-NativeCommandSucceeded -Step "git status"
    $HasChanges = $StatusLines.Count
    if ($HasChanges -eq 0) {
        Write-Host "没有需要提交的变更，跳过。"
    } else {
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
        if ($NoPush) {
            Write-Host "已提交，按 -NoPush 跳过推送。"
        } else {
            git push origin main
            Assert-NativeCommandSucceeded -Step "git push"
            Write-Host "已提交并推送到 GitHub（origin/main）。"
        }
    }
} finally {
    Pop-Location
}
