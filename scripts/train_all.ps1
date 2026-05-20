# Chain all 5 training runs sequentially.
# Logs go to logs/train_<setup>.log.
# Setup E Stage 2 resumes from Stage 1 adapter.

$ErrorActionPreference = "Stop"
Set-Location "C:\Users\tuni1\Desktop\nlp\korean_cot_distill"

$env:PYTHONIOENCODING = "utf-8"
$env:WANDB_PROJECT = "korean-cot-distill"

$python = "C:\Users\tuni1\Desktop\nlp\korean_cot_distill\venv\Scripts\python.exe"
$logsDir = "C:\Users\tuni1\Desktop\nlp\korean_cot_distill\logs"
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null

function Run-Setup($name, $extraArgs) {
    $log = Join-Path $logsDir "train_$name.log"
    Write-Host "=== Training setup $name ==="
    Write-Host "Log: $log"
    $cmd = "$python -m src.train.sft --setup $name $extraArgs"
    Write-Host "Cmd: $cmd"
    Invoke-Expression "$cmd *>&1 | Tee-Object -FilePath `"$log`""
    if ($LASTEXITCODE -ne 0) {
        Write-Host "!! Setup $name failed with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }
    Write-Host "=== Setup $name DONE ==="
}

Run-Setup "b"  ""
Run-Setup "c"  ""
Run-Setup "d"  ""
Run-Setup "e1" ""
Run-Setup "e2" "--resume_from C:\Users\tuni1\Desktop\nlp\korean_cot_distill\weights\setup_e_stage1"

Write-Host ""
Write-Host "============================================="
Write-Host "[ALL DONE] Phase 4 complete - 5 setups trained"
Write-Host "============================================="
