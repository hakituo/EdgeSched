# Windows 离线分析：不加载模型、不同步 WSL。
param()
$ErrorActionPreference = 'Stop'
$Project = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$Python = (Resolve-Path (Join-Path $Project '../../venv_core/Scripts/python.exe')).Path
Push-Location -LiteralPath $Project
try {
    & $Python -m experiments.elastic_residency.screen_small_9b
    if ($LASTEXITCODE -ne 0) { throw '离线筛选失败，请保留错误信息。' }
} finally { Pop-Location }
