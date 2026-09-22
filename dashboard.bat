@echo off
cd /d "%~dp0mem_v3"
call .\.venv\Scripts\activate.bat
python application\dashboard.py 8089
pause
