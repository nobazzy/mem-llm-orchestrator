@echo off
cd /d "C:\Users\vasco\.gemini\antigravity\scratch\mem-llm-orchestrator\mem_v3"
call .\.venv\Scripts\activate.bat
python scripts\demo_video_chaos_defense.py --steps 300 --shock-interval 100 --shock-duration 50 %*
pause
