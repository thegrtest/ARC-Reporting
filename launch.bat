@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM Dynamic launcher for ARC-Reporting components (no hardcoded paths)
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
set "PY=%ROOT%\.venv\Scripts\python.exe"

if not exist "%PY%" (
  echo [ERROR] Python virtual environment not found at:
  echo   "%PY%"
  echo Run Install.bat to set up dependencies.
  pause
  exit /b 1
)

cd /d "%ROOT%"

if /I "%~1"=="monitor" (
  echo Starting CIPMonitor.py (Dash dashboard on port 8050)...
  "%PY%" "CIPMonitor.py"
) else (
  echo Starting CIP.py (PySide6 poller GUI)...
  "%PY%" "CIP.py"
)

if errorlevel 1 (
  echo.
  echo [ERROR] Program exited with an error.
  pause
)

endlocal
