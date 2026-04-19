@echo off
REM torch-amd-setup Windows menu launcher
REM Simple interactive menu to run scripts

:menu
cls
echo.
echo ============================================================
echo  torch-amd-setup - PyTorch on AMD GPU Windows via DirectML
echo ============================================================
echo.
echo  1. Verify GPU setup (verify_gpu.py)
echo  2. Run Hello GPU demo (hello_gpu.py)
echo  3. Run benchmarks (benchmark.py)
echo  4. Setup environment (setup_env.py)
echo  5. Exit
echo.
set /p choice="Select an option (1-5): "

if "%choice%"=="1" (
    echo.
    python scripts\verify_gpu.py
    pause
    goto menu
)

if "%choice%"=="2" (
    echo.
    python scripts\hello_gpu.py
    pause
    goto menu
)

if "%choice%"=="3" (
    echo.
    python scripts\benchmark.py
    pause
    goto menu
)

if "%choice%"=="4" (
    echo.
    python scripts\setup_env.py
    pause
    goto menu
)

if "%choice%"=="5" (
    exit /b 0
)

echo Invalid choice, please try again.
pause
goto menu
