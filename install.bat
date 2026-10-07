@echo off
setlocal enabledelayedexpansion

echo ================================================================================
echo              X-TUBIT DEPENDENCY INSTALLER ^& ENVIRONMENT SETUP                   
echo ================================================================================

:: Check for Python in venv
if exist .venv\Scripts\python.exe (
    echo [+] Existing virtual environment found at .venv
    goto :INSTALL
)

:: Locate Python executable
where python >nul 2>nul
if %ERRORLEVEL% equ 0 (
    set PY_CMD=python
    goto :CREATE_VENV
)

if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
    set PY_CMD="%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    goto :CREATE_VENV
)

echo [!] Python not found in PATH. Please install Python 3.11 or run install.ps1
pause
exit /b 1

:CREATE_VENV
echo [*] Creating virtual environment (.venv)...
%PY_CMD% -m venv .venv

:INSTALL
echo [*] Installing dependencies from requirements.txt...
.venv\Scripts\python.exe -m pip install --upgrade pip --quiet
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install -e . --no-deps

echo [*] Running verification test suite...
.venv\Scripts\python.exe -m pytest tests/ -v

echo ================================================================================
echo   SUCCESS: All X-TUBIT dependencies are installed and verified!
echo ================================================================================
echo To launch the pipeline: .venv\Scripts\python scripts/run_pipeline.py
echo To launch the app     : .venv\Scripts\streamlit run app/streamlit_app.py
echo ================================================================================
pause
