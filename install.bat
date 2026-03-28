@echo off
REM install.bat — torch-amd-setup installer for Windows
REM For Linux/macOS: use install.sh instead
REM
REM Usage:
REM   install.bat              — DirectML (AMD/Intel/NVIDIA, Python 3.11 required)
REM   install.bat --cpu        — CPU only (any Python version)
REM   install.bat --cuda       — CUDA (NVIDIA GPU, any Python version)
REM   install.bat --check      — verify environment only

setlocal EnableDelayedExpansion
set MODE=directml

for %%A in (%*) do (
    if "%%A"=="--cpu"   set MODE=cpu
    if "%%A"=="--cuda"  set MODE=cuda
    if "%%A"=="--check" set MODE=check
)

echo.
echo ════════════════════════════════════════════════════
echo   torch-amd-setup installer ^(Windows^)
echo ════════════════════════════════════════════════════
echo.

if "%MODE%"=="check" (
    python -c "
import sys, platform
print('Python:', sys.version)
try:
    import torch
    print('torch :', torch.__version__)
    print('CUDA  :', torch.cuda.is_available())
except: print('torch : NOT INSTALLED')
try:
    import torch_directml as dml
    print('DML   :', dml.__version__, '→', dml.device_name(0) if hasattr(dml, 'device_name') else 'OK')
except: print('DML   : not installed')
try:
    from torch_amd_setup import get_best_device
    print('setup :', get_best_device())
except: print('torch-amd-setup: NOT INSTALLED')
"
    goto :end
)

if "%MODE%"=="directml" (
    echo [INFO] DirectML mode — requires Python 3.11
    echo.

    where py >nul 2>&1
    if %errorlevel% neq 0 (
        echo [ERROR] Python Launcher not found.
        echo         Install Python 3.11 from https://python.org
        goto :fail
    )

    py -3.11 --version >nul 2>&1
    if %errorlevel% neq 0 (
        echo [ERROR] Python 3.11 not found.
        echo         Download: https://python.org/downloads/release/python-3119/
        echo         torch-directml requires Python ^<= 3.11 — this is a hard limit.
        goto :fail
    )

    echo [1/3] Creating .venv311...
    if not exist ".venv311" (
        py -3.11 -m venv .venv311
    ) else (
        echo       Already exists — skipping
    )

    echo [2/3] Installing torch-directml + torch-amd-setup...
    call .venv311\Scripts\activate.bat
    pip install torch-directml
    pip install torch-amd-setup

    echo [3/3] Verifying...
    python -c "
import sys
print('Python:', sys.version.split()[0])
import torch; print('torch :', torch.__version__)
import torch_directml as dml
print('DML   :', dml.__version__)
from torch_amd_setup import get_best_device, device_info
best = get_best_device()
print('Device:', best)
import json
info = device_info()
for k,v in info.items(): print(f'  {k}: {v}')
"
    echo.
    echo ════════════════════════════════════════════════════
    echo   Done! Activate and run:
    echo.
    echo   .venv311\Scripts\activate
    echo   python examples\basic_usage.py
    echo ════════════════════════════════════════════════════
    goto :end
)

if "%MODE%"=="cpu" (
    echo [1/2] Installing torch ^(CPU^)...
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install torch-amd-setup
    python -c "from torch_amd_setup import get_best_device; print('Device:', get_best_device())"
    goto :end
)

if "%MODE%"=="cuda" (
    echo [1/2] Installing torch+CUDA 12.1...
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    pip install torch-amd-setup
    python -c "from torch_amd_setup import get_best_device; print('Device:', get_best_device())"
    goto :end
)

:fail
echo.
pause
exit /b 1

:end
echo.
pause
