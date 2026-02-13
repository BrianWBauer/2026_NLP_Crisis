@echo off
REM ============================================
REM Project Setup (Windows)
REM 2026_NLP_Crisis
REM ============================================
REM Run this once on a new machine to create the
REM conda environment for this project.
REM
REM Usage: Double-click or run from Anaconda Prompt
REM ============================================

set ENV_NAME=nlp_ema

REM Check if environment already exists
call conda info --envs | findstr /C:"%ENV_NAME%" >nul 2>&1
if %errorlevel% equ 0 (
    echo Environment '%ENV_NAME%' already exists.
    echo To rebuild: conda env remove -n %ENV_NAME%, then rerun this script.
    pause
    exit /b 0
)

call conda env create -f environment.yml

echo.
echo =========================================
echo Setup complete.
echo Run: conda activate nlp_ema
echo =========================================
pause
