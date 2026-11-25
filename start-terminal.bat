@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start-all.ps1" %*
if errorlevel 1 (
    echo.
    echo [WeatherQuant] Startup failed. Press any key to exit...
    pause >nul
    exit /b 1
)
