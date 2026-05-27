@echo off
:: ============================================================
::  EXAONE-3.5-2.4B-Instruct  full pipeline runner
::  Branch: exaone
::  Estimated runtime: ~16h
:: ============================================================

set CONDA_ENV=C:\Users\ksjga\miniconda3\envs\korean_cot
set PYTHON=%CONDA_ENV%\python.exe
set PROJECT_DIR=%~dp0

echo ============================================================
echo  EXAONE-3.5-2.4B-Instruct Pipeline Start
echo  Working dir: %PROJECT_DIR%
echo ============================================================

cd /d "%PROJECT_DIR%"

:: ── STEP 1: Training (5 setups) ──────────────────────────────
echo.
echo [1/7] Training setup_b  (English CoT FT)
"%PYTHON%" -m src.train.sft --setup b
if %ERRORLEVEL% NEQ 0 ( echo ERROR in setup_b & exit /b 1 )

echo.
echo [2/7] Training setup_c  (Korean CoT FT)
"%PYTHON%" -m src.train.sft --setup c
if %ERRORLEVEL% NEQ 0 ( echo ERROR in setup_c & exit /b 1 )

echo.
echo [3/7] Training setup_d  (Bilingual Mix FT)
"%PYTHON%" -m src.train.sft --setup d
if %ERRORLEVEL% NEQ 0 ( echo ERROR in setup_d & exit /b 1 )

echo.
echo [4/7] Training setup_e  (DALR)
"%PYTHON%" -m src.train.sft --setup e
if %ERRORLEVEL% NEQ 0 ( echo ERROR in setup_e & exit /b 1 )

echo.
echo [5/7] Training setup_e_random  (DALR ablation)
"%PYTHON%" -m src.train.sft --setup e_random
if %ERRORLEVEL% NEQ 0 ( echo ERROR in setup_e_random & exit /b 1 )

:: ── STEP 2: Evaluation (all setups x all benches) ────────────
echo.
echo [6/7] Evaluation: all setups x all benches
"%PYTHON%" -m src.eval.evaluate --setup all --bench all
if %ERRORLEVEL% NEQ 0 ( echo ERROR in evaluate & exit /b 1 )

:: ── STEP 3: XLSC + Cascade ───────────────────────────────────
echo.
echo [7a/7] XLSC  (setup_e / hrm8k / n=3 / temp=0.7)
"%PYTHON%" -m src.eval.xlsc --setup e --bench hrm8k --n 3 --temp 0.7
if %ERRORLEVEL% NEQ 0 ( echo ERROR in xlsc & exit /b 1 )

echo.
echo [7b/7] Cascade XLSC  (setup_e / hrm8k)
"%PYTHON%" -m src.eval.cascade_xlsc --setup e --bench hrm8k
if %ERRORLEVEL% NEQ 0 ( echo ERROR in cascade_xlsc & exit /b 1 )

echo.
echo ============================================================
echo  ALL DONE!  Results in: %PROJECT_DIR%results\
echo ============================================================
