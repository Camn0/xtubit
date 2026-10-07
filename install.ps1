# X-TUBIT Automated Environment & Dependency Installer (PowerShell)
# Usage: .\install.ps1

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "             X-TUBIT DEPENDENCY INSTALLER & ENVIRONMENT SETUP                   " -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan

# 1. Locate suitable Python 3.10+ executable
$pythonExe = $null

$candidates = @(
    "python",
    "py",
    "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
    "C:\Program Files\Python311\python.exe",
    "C:\Program Files\Python312\python.exe"
)

foreach ($c in $candidates) {
    try {
        $ver = & $c --version 2>$null
        if ($ver -match "Python 3\.(1[0-9]|[2-9][0-9])") {
            $pythonExe = $c
            Write-Host "[+] Found compatible Python: $ver ($c)" -ForegroundColor Green
            break
        }
    } catch {}
}

if (-not $pythonExe) {
    Write-Host "[!] Compatible Python 3.10+ not found in PATH or standard directories." -ForegroundColor Red
    Write-Host "[!] Installing Python 3.11 via winget..." -ForegroundColor Yellow
    winget install Python.Python.3.11 --accept-source-agreements --accept-package-agreements
    $pythonExe = "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"
}

# 2. Create virtual environment .venv if not present
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "[*] Creating virtual environment (.venv)..." -ForegroundColor Yellow
    & $pythonExe -m venv .venv
} else {
    Write-Host "[+] Existing virtual environment detected at .venv" -ForegroundColor Green
}

$venvPython = ".venv\Scripts\python.exe"

# 3. Upgrade pip
Write-Host "[*] Upgrading pip inside .venv..." -ForegroundColor Yellow
& $venvPython -m pip install --upgrade pip --quiet

# 4. Install all dependencies from requirements.txt and package editable
Write-Host "[*] Installing all dependencies from requirements.txt..." -ForegroundColor Yellow
& $venvPython -m pip install -r requirements.txt
& $venvPython -m pip install -e . --no-deps

# 5. Verify installation by running test suite
Write-Host "[*] Running verification unit test suite..." -ForegroundColor Yellow
& $venvPython -m pytest tests/ -v

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "  SUCCESS: All X-TUBIT dependencies installed and verified!                    " -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "Quick commands to run:"
Write-Host "  Activate environment : .\.venv\Scripts\Activate.ps1"
Write-Host "  Execute pipeline     : .\.venv\Scripts\python scripts/run_pipeline.py"
Write-Host "  Launch web UI        : .\.venv\Scripts\streamlit run app/streamlit_app.py"
Write-Host "================================================================================" -ForegroundColor Cyan
