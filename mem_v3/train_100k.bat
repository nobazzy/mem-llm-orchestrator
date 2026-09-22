@echo off
cd /d "C:\Users\vasco\.gemini\antigravity\scratch\mem-llm-orchestrator\mem_v3"
call .\.venv\Scripts\activate.bat
python scripts\train_production_100k.py --steps 100000 --model-preset large_130m --eval-window 25 --checkpoint-interval 1000 %*
pause
