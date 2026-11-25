# Install local git pre-commit hook running scripts/check.ps1
$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$HooksDir = Join-Path $RepoRoot ".git\hooks"
New-Item -ItemType Directory -Force -Path $HooksDir | Out-Null

$HookPath = Join-Path $HooksDir "pre-commit"
$HookContent = @"
#!/bin/sh
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/check.ps1
"@

# Write LF-only shell hook without BOM
[System.IO.File]::WriteAllText($HookPath, $HookContent.Replace("`r`n", "`n"), (New-Object System.Text.UTF8Encoding($false)))
Write-Host "Installed pre-commit hook at $HookPath" -ForegroundColor Green
