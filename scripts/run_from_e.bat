@echo off
:: ============================================================
::  EXAONE pipeline  — setup_e 부터 끝까지 (b/c/d 완료 후)
::  실행: 이 파일을 더블클릭하거나 CMD에서 실행
::  Claude와 무관하게 독립적으로 동작
:: ============================================================

set PYTHON=C:\Users\ksjga\miniconda3\envs\korean_cot\python.exe
set PROJECT=C:\Users\ksjga\OneDrive\바탕 화면\261RCOSE46101

cd /d "%PROJECT%"

echo.
echo ============================================================
echo  [1/5] setup_e  (DALR)
echo ============================================================
"%PYTHON%" -m src.train.sft --setup e
if %ERRORLEVEL% NEQ 0 ( echo [ERROR] setup_e 실패 & pause & exit /b 1 )

echo.
echo ============================================================
echo  [2/5] setup_e_random  (DALR ablation)
echo ============================================================
"%PYTHON%" -m src.train.sft --setup e_random
if %ERRORLEVEL% NEQ 0 ( echo [ERROR] setup_e_random 실패 & pause & exit /b 1 )

echo.
echo ============================================================
echo  [3/5] evaluate  (전체 평가 - 약 6~7시간)
echo ============================================================
"%PYTHON%" -m src.eval.evaluate --setup all --bench all
if %ERRORLEVEL% NEQ 0 ( echo [ERROR] evaluate 실패 & pause & exit /b 1 )

echo.
echo ============================================================
echo  [4/5] XLSC
echo ============================================================
"%PYTHON%" -m src.eval.xlsc --setup e --bench hrm8k --n 3 --temp 0.7
if %ERRORLEVEL% NEQ 0 ( echo [ERROR] xlsc 실패 & pause & exit /b 1 )

echo.
echo ============================================================
echo  [5/5] Cascade XLSC
echo ============================================================
"%PYTHON%" -m src.eval.cascade_xlsc --setup e --bench hrm8k
if %ERRORLEVEL% NEQ 0 ( echo [ERROR] cascade_xlsc 실패 & pause & exit /b 1 )

echo.
echo ============================================================
echo  모든 실험 완료!
echo  결과 위치: %PROJECT%\results\
echo ============================================================
pause
