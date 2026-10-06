@echo off
title MEM LLM Studio - Autonomous Control Center
cls
echo ===============================================================================
echo   MEM LLM STUDIO - STARTING AUTONOMOUS CONTROL CENTER
echo ===============================================================================
echo.

if not exist "mem_v3\.venv\Scripts\python.exe" (
    echo [!] Virtual environment not detected. Running installer first...
    call install.bat
)

if exist "MEM_Desktop.exe" (
    echo [*] Launching Native Desktop Studio...
    start "" "MEM_Desktop.exe"
) else (
    echo [*] Launching via Python launcher...
    start "" mem_v3\.venv\Scripts\python.exe MEM_Launcher.py
)

exit
