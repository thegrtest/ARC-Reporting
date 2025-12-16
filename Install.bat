@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================
REM RemVision Installer (Pinned + venv, inline deps) - STABLE
REM - Robust Python 3.10 install (python.org + winget fallback)
REM - Always rebuilds venv (prevents old-path pip)
REM - Installs deps INLINE (no requirements file, avoids '' invalid requirement)
REM - Creates Run_RemVision.bat and Desktop shortcut with remington.ico
REM ============================================================

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

set "APP_NAME=RemVision"
set "VENV_DIR=%ROOT%\.venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
set "RUN_BAT=%ROOT%\Run_RemVision.bat"
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
REM STEP 5/6: Install dependencies INLINE (no -r file)
REM ------------------------------------------------------------
call :LOGI "STEP 5/6: Installing dependencies (may take several minutes)..."

REM We install PyTorch CPU wheels from PyTorch index + PyPI for everything else
"%VENV_PY%" -m pip install --upgrade ^
  --index-url https://download.pytorch.org/whl/cpu ^
  --extra-index-url https://pypi.org/simple ^
  "torch==2.9.1+cpu" ^
  "torchaudio==2.9.1+cpu" ^
  "torchvision==0.24.1+cpu" ^
  "absl-py==2.3.1" ^
  "loguru==0.7.3" ^
  "opencv-python==4.12.0.88" ^
  "pillow==12.0.0" ^
  "pycocotools==2.0.10" ^
  "PySide6==6.10.0" ^
  "PySide6_Addons==6.10.0" ^
  "PySide6_Essentials==6.10.0" ^
  "PyYAML==6.0.3" ^
  "QDarkStyle==3.2.3" ^
  "requests==2.32.5" ^
  "tabulate==0.9.0" ^
  "tensorboard==2.20.0" ^
  "tensorboard-data-server==0.7.2" ^
  "tqdm==4.67.1" ^
  "matplotlib" ^
  "psutil" ^
  "thop" >> "%LOG%" 2>&1

if errorlevel 1 (
  call :LOGE "Dependency install failed."
  echo [ERROR] Dependency install failed. See install.log: "%LOG%"
  pause
  exit /b 1
)

call :LOGI "Dependencies installed."

REM ------------------------------------------------------------
REM STEP 6/6: Create Run_RemVision.bat + Desktop shortcut
REM ------------------------------------------------------------
call :LOGI "STEP 6/6: Creating launcher and Desktop shortcut..."

> "%RUN_BAT%" (
  echo @echo off
  echo setlocal EnableExtensions
  echo set "ROOT=%%~dp0"
  echo if "%%ROOT:~-1%%"=="\" set "ROOT=%%ROOT:~0,-1%%"
  echo set "PY=%%ROOT%%\.venv\Scripts\python.exe"
  echo if not exist "%%PY%%" ^(
  echo   echo [ERROR] venv not found. Run installer.
  echo   pause
  echo   exit /b 1
  echo ^)
  echo cd /d "%%ROOT%%"
  echo "%%PY%%" "main.py"
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
echo [OK] Desktop shortcut created (Run_RemVision.bat + remington.ico)
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
set "TMP=%TEMP%\REMVISION_SETUP"
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
