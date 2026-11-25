# PowerShell 5.1 compatible one-command launcher for Polymarket Weather Quant Engine & Pro Terminal
$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$BackendDir = Join-Path $RepoRoot "backend"
$FrontendDir = Join-Path $RepoRoot "frontend"
$EnvFile = Join-Path $RepoRoot ".env"
$VenvDir = Join-Path $RepoRoot ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

# 1. Load .env variables into current process if .env exists
if (Test-Path $EnvFile) {
    Write-Host "[WeatherQuant] Loading environment from .env..." -ForegroundColor Cyan
    Get-Content $EnvFile | ForEach-Object {
        $line = $_.Trim()
        if ($line.Length -eq 0 -or $line.StartsWith("#")) {
            return
        }
        $idx = $line.IndexOf("=")
        if ($idx -gt 0) {
            $key = $line.Substring(0, $idx).Trim()
            $val = $line.Substring($idx + 1).Trim()
            if (($val.StartsWith('"') -and $val.EndsWith('"')) -or ($val.StartsWith("'") -and $val.EndsWith("'"))) {
                $val = $val.Substring(1, $val.Length - 2)
            }
            if ($key.Length -gt 0) {
                [Environment]::SetEnvironmentVariable($key, $val, "Process")
            }
        }
    }
}

# 2. Ensure PYTHONPATH includes backend\src
$BackendSrc = Join-Path $BackendDir "src"
if ($env:PYTHONPATH) {
    $env:PYTHONPATH = "$BackendSrc;$($env:PYTHONPATH)"
} else {
    $env:PYTHONPATH = $BackendSrc
}

if (-not $env:NEXT_PUBLIC_API_BASE_URL) {
    $env:NEXT_PUBLIC_API_BASE_URL = "http://127.0.0.1:8000"
}

# 3. Bootstrap Python .venv only if missing
if (-not (Test-Path $VenvPython)) {
    Write-Host "[WeatherQuant] .venv not found. Bootstrapping Python virtual environment..." -ForegroundColor Yellow
    & python -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create Python virtual environment at $VenvDir"
    }
    & $VenvPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to upgrade pip in $VenvDir"
    }
    & $VenvPython -m pip install -e "$BackendDir[dev]"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install backend dependencies from $BackendDir"
    }
}

# 4. Bootstrap frontend/node_modules only if missing
$NodeModulesDir = Join-Path $FrontendDir "node_modules"
if (-not (Test-Path $NodeModulesDir)) {
    Write-Host "[WeatherQuant] frontend\node_modules not found. Running npm install..." -ForegroundColor Yellow
    Push-Location $FrontendDir
    try {
        & npm install
        if ($LASTEXITCODE -ne 0) {
            throw "npm install failed in $FrontendDir"
        }
    } finally {
        Pop-Location
    }
}

# 5. Launch FastAPI backend in a labelled PowerShell window
$BackendCmd = @"
`$Host.UI.RawUI.WindowTitle = 'WeatherQuant Backend (FastAPI :8000)'
Set-Location '$RepoRoot'
`$env:PYTHONPATH = '$($env:PYTHONPATH)'
& '$VenvPython' -m uvicorn weather_quant.api.app:app --host 127.0.0.1 --port 8000
"@
Start-Process -FilePath "powershell.exe" -ArgumentList @("-NoExit", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $BackendCmd) | Out-Null

# 6. Launch Next.js Pro Terminal in a labelled PowerShell window
$FrontendCmd = @"
`$Host.UI.RawUI.WindowTitle = 'WeatherQuant Terminal (Next.js :3000)'
Set-Location '$FrontendDir'
`$env:NEXT_PUBLIC_API_BASE_URL = '$($env:NEXT_PUBLIC_API_BASE_URL)'
npm run dev
"@
Start-Process -FilePath "powershell.exe" -ArgumentList @("-NoExit", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $FrontendCmd) | Out-Null

# 7. Wait briefly for /api/health
$HealthUrl = "http://127.0.0.1:8000/api/health"
$HealthReady = $false
Write-Host "[WeatherQuant] Waiting for FastAPI backend (/api/health)..." -ForegroundColor Cyan
for ($i = 0; $i -lt 20; $i++) {
    try {
        $resp = Invoke-WebRequest -Uri $HealthUrl -UseBasicParsing -TimeoutSec 2
        if ($resp.StatusCode -eq 200) {
            $HealthReady = $true
            break
        }
    } catch {
        Start-Sleep -Seconds 1
    }
}

if ($HealthReady) {
    Write-Host "[WeatherQuant] Backend is healthy at $HealthUrl" -ForegroundColor Green
} else {
    Write-Host "[WeatherQuant] Backend is still starting; check the Backend window if needed." -ForegroundColor Yellow
}

# 8. Open Pro Terminal in default browser
$FrontendUrl = "http://localhost:3000"
try {
    Start-Process $FrontendUrl | Out-Null
} catch {
    Write-Host "[WeatherQuant] Open browser manually: $FrontendUrl" -ForegroundColor Yellow
}

# 9. Print clear RU + EN operator URLs
Write-Host ""
Write-Host "==================================================================" -ForegroundColor DarkCyan
Write-Host "  Polymarket Weather Quant Engine & Pro Terminal — Started" -ForegroundColor White
Write-Host "  Терминал и квантовый движок запущены" -ForegroundColor White
Write-Host "==================================================================" -ForegroundColor DarkCyan
Write-Host "  Pro Web Terminal (Веб-терминал):    http://localhost:3000" -ForegroundColor Green
Write-Host "  Backend API Docs (Swagger UI):      http://127.0.0.1:8000/docs" -ForegroundColor Cyan
Write-Host "  Health Check (Статус движка):       http://127.0.0.1:8000/api/health" -ForegroundColor Cyan
Write-Host "  Telegram Radar (Статус радара):     http://127.0.0.1:8000/api/telegram/status" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor DarkCyan
