@echo off
:: ============================================================
::  EXAONE pipeline  — setup_e_random 부터 끝까지
::  b/c/d/e 완료 후 실행 (e_random 재시작)
::  실행: 이 파일을 더블클릭하거나 CMD에서 실행
:: ============================================================

set PYTHON=C:\Users\ksjga\miniconda3\envs\korean_cot\python.exe
set PROJECT=C:\Users\ksjga\OneDrive\바탕 화면\261RCOSE46101

cd /d "%PROJECT%"

echo.
echo ============================================================
echo  [1/4] setup_e_random  (DALR ablation)
echo ============================================================
"%PYTHON%" -m src.train.sft --setup e_random >> "%PROJECT%\logs\train_e_random.log" 2>&1
if %ERRORLEVEL% NEQ 0 ( echo [ERROR] setup_e_random 실패 >> "%PROJECT%\logs\train_e_random.log" & goto :fail )
echo [DONE] setup_e_random >> "%PROJECT%\logs\train_e_random.log"

echo.
echo ============================================================
echo  [2/4] evaluate  (전체 평가 - 약 6~7시간)
echo ============================================================
"%PYTHON%" -m src.eval.evaluate --setup all --bench all >> "%PROJECT%\logs\evaluate.log" 2>&1
if %ERRORLEVEL% NEQ 0 ( echo [ERROR] evaluate 실패 >> "%PROJECT%\logs\evaluate.log" & goto :fail )
echo [DONE] evaluate >> "%PROJECT%\logs\evaluate.log"

echo.
echo ============================================================
echo  [3/4] XLSC
echo ============================================================
"%PYTHON%" -m src.eval.xlsc --setup e --bench hrm8k --n 3 --temp 0.7 >> "%PROJECT%\logs\xlsc.log" 2>&1
if %ERRORLEVEL% NEQ 0 ( echo [ERROR] xlsc 실패 >> "%PROJECT%\logs\xlsc.log" & goto :fail )
echo [DONE] xlsc >> "%PROJECT%\logs\xlsc.log"

echo.
echo ============================================================
echo  [4/4] Cascade XLSC
echo ============================================================
"%PYTHON%" -m src.eval.cascade_xlsc --setup e --bench hrm8k >> "%PROJECT%\logs\cascade.log" 2>&1
if %ERRORLEVEL% NEQ 0 ( echo [ERROR] cascade_xlsc 실패 >> "%PROJECT%\logs\cascade.log" & goto :fail )
echo [DONE] cascade_xlsc >> "%PROJECT%\logs\cascade.log"

echo.
echo ============================================================
echo  모든 실험 완료!
echo  결과 위치: %PROJECT%\results\
echo ============================================================
echo [ALL DONE] %DATE% %TIME% >> "%PROJECT%\logs\pipeline.log"
exit /b 0

:fail
echo [PIPELINE FAILED] %DATE% %TIME% >> "%PROJECT%\logs\pipeline.log"
exit /b 1
