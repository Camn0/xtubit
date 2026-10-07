@echo off
if not exist .venv\Scripts\python.exe (
    echo [!] Virtual environment not found. Running installer first...
    call install.bat
)
echo [*] Executing X-TUBIT Computational Pipeline...
.venv\Scripts\python.exe scripts/run_pipeline.py
pause
