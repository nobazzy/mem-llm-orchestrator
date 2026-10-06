@echo off
setlocal enabledelayedexpansion
title MEM LLM Studio - Automated Installation & Hardware Setup
cls
echo ===============================================================================
echo   MEM LLM STUDIO - AUTOMATED ENVIRONMENT INSTALLER
echo   Commercial Local AI Training & Inference Platform
echo ===============================================================================
echo.

:: 1. Check Python installation
echo [*] Checking Python environment...
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python was not found in your system PATH.
    echo Please install Python 3.10, 3.11, or 3.12 from: https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"') do set PY_VER=%%i
echo [*] Detected Python version: %PY_VER%

:: 2. Create or verify virtual environment in mem_v3\.venv
echo [*] Setting up Python virtual environment (mem_v3\.venv)...
if not exist "mem_v3\.venv" (
    python -m venv mem_v3\.venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [*] Virtual environment created successfully.
) else (
    echo [*] Virtual environment already exists.
)

:: 3. Upgrade pip and set up wheel
echo [*] Upgrading pip, setuptools, and wheel...
mem_v3\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel --quiet

:: 4. Install PyTorch with CUDA or CPU
echo [*] Detecting GPU capabilities and installing PyTorch...
where nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo [*] NVIDIA GPU detected! Installing PyTorch with CUDA 12.1 acceleration...
    mem_v3\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet
) else (
    echo [*] No NVIDIA GPU found. Installing standard PyTorch (CPU mode)...
    mem_v3\.venv\Scripts\python.exe -m pip install torch torchvision --quiet
)

:: 5. Install Studio dependencies
echo [*] Installing production dependencies...
if exist "mem_v3\requirements.txt" (
    mem_v3\.venv\Scripts\python.exe -m pip install -r mem_v3\requirements.txt --quiet
)
mem_v3\.venv\Scripts\python.exe -m pip install certifi safetensors psutil --quiet

:: 6. Verification test
echo.
echo ===============================================================================
echo   HARDWARE & ENVIRONMENT VERIFICATION
echo ===============================================================================
mem_v3\.venv\Scripts\python.exe -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available:  {torch.cuda.is_available()}'); print(f'Active Device:   {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
echo ===============================================================================
echo.
echo [SUCCESS] MEM LLM Studio installed successfully!
echo.
echo You can now launch the studio anytime by double-clicking:
echo   -^> start_studio.bat
echo.
pause
