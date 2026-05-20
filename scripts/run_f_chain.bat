@echo off
cd /d "C:\Users\tuni1\Desktop\nlp\korean_cot_distill"
set PYTHONIOENCODING=utf-8
set WANDB_PROJECT=korean-cot-distill

echo [TRAIN F] start %date% %time% >> "C:\Users\tuni1\Desktop\nlp\korean_cot_distill\logs\pipeline_f.log" 2>&1
"C:\Users\tuni1\Desktop\nlp\korean_cot_distill\venv\Scripts\python.exe" -m src.train.sft --setup f >> "C:\Users\tuni1\Desktop\nlp\korean_cot_distill\logs\train_f.log" 2>&1
echo [TRAIN F] exit %errorlevel% %date% %time% >> "C:\Users\tuni1\Desktop\nlp\korean_cot_distill\logs\pipeline_f.log" 2>&1
if errorlevel 1 goto :fail

echo [EVAL F] start %date% %time% >> "C:\Users\tuni1\Desktop\nlp\korean_cot_distill\logs\pipeline_f.log" 2>&1
"C:\Users\tuni1\Desktop\nlp\korean_cot_distill\venv\Scripts\python.exe" -m src.eval.evaluate --setup f --bench all --limit 500 --tag final >> "C:\Users\tuni1\Desktop\nlp\korean_cot_distill\logs\eval_f.log" 2>&1
echo [EVAL F] exit %errorlevel% %date% %time% >> "C:\Users\tuni1\Desktop\nlp\korean_cot_distill\logs\pipeline_f.log" 2>&1
if errorlevel 1 goto :fail

echo [CLSC FINAL] start %date% %time% >> "C:\Users\tuni1\Desktop\nlp\korean_cot_distill\logs\pipeline_f.log" 2>&1
"C:\Users\tuni1\Desktop\nlp\korean_cot_distill\venv\Scripts\python.exe" -m src.eval.clsc --bench all --limit 500 >> "C:\Users\tuni1\Desktop\nlp\korean_cot_distill\logs\clsc_final.log" 2>&1
echo [CLSC FINAL] exit %errorlevel% %date% %time% >> "C:\Users\tuni1\Desktop\nlp\korean_cot_distill\logs\pipeline_f.log" 2>&1

echo [ALL DONE] %date% %time% >> "C:\Users\tuni1\Desktop\nlp\korean_cot_distill\logs\pipeline_f.log" 2>&1
goto :end

:fail
echo [FAILED] step failed, stopping >> "C:\Users\tuni1\Desktop\nlp\korean_cot_distill\logs\pipeline_f.log" 2>&1

:end
