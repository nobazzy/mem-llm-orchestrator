@echo off
cd /d "C:\Users\vasco\.gemini\antigravity\scratch\mem-llm-orchestrator\mem_v3"
call .\.venv\Scripts\activate.bat
python application\dashboard.py 8089
pause
