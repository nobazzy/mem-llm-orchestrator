@echo off
title MEM LLM Orchestrator - Control Center
cd /d "%~dp0"

echo =====================================================================
echo                MEM LLM ORCHESTRATOR - CONTROL CENTER
echo =====================================================================

set "PY_EXE=mem_v3\.venv\Scripts\python.exe"

if not exist "%PY_EXE%" (
    set "PY_EXE=.venv\Scripts\python.exe"
)

if not exist "%PY_EXE%" (
    set "PY_EXE=python"
)

echo [*] Iniciando servico e interface desktop...
"%PY_EXE%" MEM_Launcher.py

pause
