@echo off
setlocal
cd /d "%~dp0\.."

echo ============================================================
echo   MEM ORCHESTRATOR - PRESET 250M PARAMETROS ^| 10M STEPS
echo   Modelo: xlarge_250m (~255M Params) ^| Device: NVIDIA RTX 5060 Ti
echo   Dataset: allenai/c4 (en) [Web Corpus Ruidoso]
echo   Target: 10,000,000 steps
echo ============================================================

python main.py ^
  --steps 10000000 ^
  --model-preset xlarge_250m ^
  --dataset-name allenai/c4 ^
  --dataset-config en ^
  --sequence-length 256 ^
  --start-lane safe_seq256 ^
  --precision fp16 ^
  --resume-latest

pause
