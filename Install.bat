@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================
REM ARC-Reporting Installer (venv + requirements.txt)
REM - Ensures Python 3.10 is available (python.org + winget fallback)
REM - Rebuilds venv each run to avoid stale paths
REM - Installs dependencies from requirements.txt (pandas, dash, dash-daq, PySide6, pylogix)
REM - Creates Run_ARC-Reporting.bat and Desktop shortcut
REM ============================================================

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

set "APP_NAME=ARC-Reporting"
set "VENV_DIR=%ROOT%\.venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
set "RUN_BAT=%ROOT%\Run_ARC-Reporting.bat"
set "ICON_FILE=%ROOT%\remington.ico"
set "LOG=%ROOT%\install.log"

> "%LOG%" (
  echo ============================================================
  echo %APP_NAME% Install Log - %DATE% %TIME%
  echo Root: "%ROOT%"
  echo ============================================================
)

call :LOGI "============================================================"
call :LOGI "%APP_NAME% Installer"
call :LOGI "Root: %ROOT%"
call :LOGI "Log : %LOG%"
call :LOGI "============================================================"

where powershell >nul 2>&1
if errorlevel 1 (
  call :LOGE "PowerShell not found."
  pause
  exit /b 1
)

REM ------------------------------------------------------------
REM STEP 1/6: Ensure Python 3.10 exists (install if missing)
REM ------------------------------------------------------------
call :LOGI "STEP 1/6: Locating Python 3.10..."
call :FIND_PY310

if not defined SYS_PY (
  call :LOGI "Python 3.10 not found. Installing Python 3.10.11..."
  call :INSTALL_PY310
  if errorlevel 1 (
    call :LOGE "Python install failed. See install.log for details."
    pause
    exit /b 1
  )
  call :LOGI "Rechecking Python 3.10 after install..."
  call :FIND_PY310
)

if not defined SYS_PY (
  call :LOGE "Python 3.10 still not detected after install."
  pause
  exit /b 1
)

call :LOGI "Using Python: %SYS_PY%"

REM ------------------------------------------------------------
REM STEP 2/6: Rebuild venv (always) to avoid old-path pip
REM ------------------------------------------------------------
call :LOGI "STEP 2/6: Rebuilding venv..."
if exist "%VENV_DIR%" (
  rmdir /s /q "%VENV_DIR%" >> "%LOG%" 2>&1
)

REM ------------------------------------------------------------
REM STEP 3/6: Create venv
REM ------------------------------------------------------------
call :LOGI "STEP 3/6: Creating venv..."
"%SYS_PY%" -m venv "%VENV_DIR%" >> "%LOG%" 2>&1
if errorlevel 1 (
  call :LOGE "venv creation failed."
  pause
  exit /b 1
)

set PYTHONHOME=
set PYTHONPATH=

REM ------------------------------------------------------------
REM STEP 4/6: Upgrade pip
REM ------------------------------------------------------------
call :LOGI "STEP 4/6: Upgrading pip..."
"%VENV_PY%" -m pip install --upgrade pip setuptools wheel >> "%LOG%" 2>&1
if errorlevel 1 (
  call :LOGE "pip upgrade failed."
  pause
  exit /b 1
)

REM ------------------------------------------------------------
REM STEP 5/6: Install dependencies from requirements.txt
REM ------------------------------------------------------------
call :LOGI "STEP 5/6: Installing dependencies (may take a few minutes)..."

if not exist "%ROOT%\requirements.txt" (
  call :LOGE "requirements.txt not found in %ROOT%"
  echo [ERROR] Missing requirements.txt next to Install.bat.
  pause
  exit /b 1
)

"%VENV_PY%" -m pip install --upgrade -r "%ROOT%\requirements.txt" >> "%LOG%" 2>&1

if errorlevel 1 (
  call :LOGE "Dependency install failed."
  echo [ERROR] Dependency install failed. See install.log: "%LOG%"
  pause
  exit /b 1
)

call :LOGI "Dependencies installed."

REM ------------------------------------------------------------
REM STEP 6/6: Create Run_ARC-Reporting.bat + Desktop shortcut
REM ------------------------------------------------------------
call :LOGI "STEP 6/6: Creating launcher and Desktop shortcut..."

> "%RUN_BAT%" (
  echo @echo off
  echo setlocal EnableExtensions
  echo set "ROOT=%%~dp0"
  echo if "%%ROOT:~-1%%"=="\" set "ROOT=%%ROOT:~0,-1%%"
  echo set "PY=%%ROOT%%\.venv\Scripts\python.exe"
  echo if not exist "%%PY%%" ^(
  echo   echo [ERROR] venv not found. Run Install.bat first.
  echo   pause
  echo   exit /b 1
  echo ^)
  echo cd /d "%%ROOT%%"
  echo if /I "%%~1"=="monitor" ^(
  echo   echo Starting CIPMonitor.py (Dash dashboard on port 8050)...
  echo   "%%PY%%" "CIPMonitor.py"
  echo ^) else ^(
  echo   echo Starting CIP.py (PySide6 poller GUI)...
  echo   "%%PY%%" "CIP.py"
  echo ^)
  echo if errorlevel 1 ^(
  echo   echo.
  echo   echo [ERROR] Program exited with an error.
  echo   pause
  echo ^)
)

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$WshShell = New-Object -ComObject WScript.Shell;" ^
  "$Desktop = [Environment]::GetFolderPath('Desktop');" ^
  "$ShortcutPath = Join-Path $Desktop '%APP_NAME%.lnk';" ^
  "if(Test-Path $ShortcutPath){Remove-Item $ShortcutPath -Force};" ^
  "$Shortcut = $WshShell.CreateShortcut($ShortcutPath);" ^
  "$Shortcut.TargetPath = '%RUN_BAT%';" ^
  "$Shortcut.WorkingDirectory = '%ROOT%';" ^
  "if (Test-Path '%ICON_FILE%') { $Shortcut.IconLocation = '%ICON_FILE%,0'; }" ^
  "$Shortcut.Save();" >> "%LOG%" 2>&1

call :LOGI "Install complete."
echo.
echo [OK] Installed %APP_NAME%.
echo [OK] Desktop shortcut created (Run_ARC-Reporting.bat)
echo [OK] Log: "%LOG%"
pause
exit /b 0


REM ================= Helpers =================
:LOGI
echo [INFO] %~1
>> "%LOG%" echo [INFO] %~1
goto :EOF

:LOGE
echo [ERROR] %~1
>> "%LOG%" echo [ERROR] %~1
goto :EOF


REM ================= Python finder =================
:FIND_PY310
set "SYS_PY="

REM Prefer py launcher if present
for /f "usebackq delims=" %%P in (`py -3.10 -c "import sys; print(sys.executable)" 2^>nul`) do set "SYS_PY=%%P"
if defined SYS_PY if exist "%SYS_PY%" goto :EOF
set "SYS_PY="

REM Typical per-user install path
set "CAND=%LocalAppData%\Programs\Python\Python310\python.exe"
if exist "%CAND%" ( set "SYS_PY=%CAND%" & goto :EOF )

goto :EOF


REM ================= Robust Python installer =================
:INSTALL_PY310
setlocal EnableExtensions EnableDelayedExpansion

set "PY_VER=3.10.11"
set "PY_INSTALLER=python-%PY_VER%-amd64.exe"
set "PY_URL=https://www.python.org/ftp/python/%PY_VER%/%PY_INSTALLER%"
set "TMP=%TEMP%\ARC_REPORTING_SETUP"
if not exist "%TMP%" mkdir "%TMP%" >nul 2>&1
set "PY_INSTALLER_PATH=%TMP%\%PY_INSTALLER%"

call :LOGI "Attempt A: python.org installer download -> silent install (per-user)."

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "[Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12;" ^
  "$ProgressPreference='SilentlyContinue';" ^
  "for($i=1;$i -le 3;$i++){ try{ Invoke-WebRequest -Uri '%PY_URL%' -OutFile '%PY_INSTALLER_PATH%' -UseBasicParsing; break } catch { if($i -eq 3){ throw } Start-Sleep -Seconds 2 } }" >> "%LOG%" 2>&1
if errorlevel 1 (
  call :LOGE "Download failed (python.org)."
  goto :WINGET_FALLBACK
)

if not exist "%PY_INSTALLER_PATH%" (
  call :LOGE "Downloaded installer missing at: %PY_INSTALLER_PATH%"
  goto :WINGET_FALLBACK
)

set "TARGET_DIR=%LocalAppData%\Programs\Python\Python310"

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$p = Start-Process -FilePath '%PY_INSTALLER_PATH%' -ArgumentList @('/quiet','InstallAllUsers=0','PrependPath=1','Include_test=0',('TargetDir=%TARGET_DIR%')) -Wait -PassThru;" ^
  "Write-Output ('Python installer exit code: ' + $p.ExitCode);" ^
  "exit $p.ExitCode" >> "%LOG%" 2>&1

set "EC=%ERRORLEVEL%"
call :LOGI "Python installer returned exit code: %EC%"

if "%EC%"=="0" (
  call :LOGI "Python installed successfully via python.org installer."
  endlocal & exit /b 0
)

call :LOGE "python.org installer failed with exit code %EC%."

:WINGET_FALLBACK
call :LOGI "Attempt B (fallback): winget install Python 3.10 (if available)."

where winget >nul 2>&1
if errorlevel 1 (
  call :LOGE "winget not available. Cannot use fallback."
  endlocal & exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "winget install --id Python.Python.3.10 -e --silent --accept-package-agreements --accept-source-agreements" >> "%LOG%" 2>&1

if errorlevel 1 (
  call :LOGE "winget install failed."
  endlocal & exit /b 1
)

call :LOGI "winget install completed."
endlocal & exit /b 0
