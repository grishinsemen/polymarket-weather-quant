# PowerShell 5.1 unified pre-commit and CI verification script
$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$PytestExe = Join-Path $RepoRoot ".venv\Scripts\pytest.exe"
$FrontendDir = Join-Path $RepoRoot "frontend"

if (-not (Test-Path $PytestExe)) {
    throw "Missing virtual environment pytest at $PytestExe"
}

Write-Host "[check] 1/3 Running backend pytest + architecture guardrails..." -ForegroundColor Cyan
$env:PYTHONPATH = Join-Path $RepoRoot "backend\src"
& $PytestExe (Join-Path $RepoRoot "backend\tests") -q
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "[check] 2/3 Running frontend TypeScript check (tsc --noEmit)..." -ForegroundColor Cyan
Push-Location $FrontendDir
try {
    & npx tsc --noEmit
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }

    Write-Host "[check] 3/3 Running frontend Vitest suite..." -ForegroundColor Cyan
    & npm test
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
} finally {
    Pop-Location
}

Write-Host "ALL CHECKS PASSED" -ForegroundColor Green
