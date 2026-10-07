@echo off
if not exist .venv\Scripts\streamlit.exe (
    echo [!] Virtual environment not found. Running installer first...
    call install.bat
)
echo [*] Starting X-TUBIT Streamlit Dashboard...
.venv\Scripts\streamlit.exe run app/streamlit_app.py
pause
