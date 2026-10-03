@echo off
title MEM Orchestrator - Production LLM Training Studio
cd /d "%~dp0"

echo =====================================================================
echo           MEM ORCHESTRATOR - PRODUCTION TRAINING STUDIO
echo =====================================================================

set "PY_EXE=mem_v3\.venv\Scripts\python.exe"

if not exist "%PY_EXE%" (
    set "PY_EXE=.venv\Scripts\python.exe"
)

if not exist "%PY_EXE%" (
    set "PY_EXE=python"
)

echo [*] Starting production backend and launching desktop interface...
"%PY_EXE%" MEM_Launcher.py

pause
