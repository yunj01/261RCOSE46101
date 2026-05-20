# ============================================================
# Setup F (DALR) + CLSC full pipeline
#   Step 1: CLSC  (B+C+D+E voting, 기존 결과 활용 — 즉시)
#   Step 2: DALR data prep
#   Step 3: DALR training  (~2.5h)
#   Step 4: DALR eval      (~2.5h)
#   Step 5: CLSC re-run    (F 포함 voting)
# ============================================================

$ErrorActionPreference = "Stop"
Set-Location "C:\Users\tuni1\Desktop\nlp\korean_cot_distill"

$env:PYTHONIOENCODING = "utf-8"
$env:WANDB_PROJECT    = "korean-cot-distill"

$python  = "C:\Users\tuni1\Desktop\nlp\korean_cot_distill\venv\Scripts\python.exe"
$logsDir = "C:\Users\tuni1\Desktop\nlp\korean_cot_distill\logs"
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null

function Run-Step($label, $cmd, $logFile) {
    Write-Host ""
    Write-Host "============================================================"
    Write-Host "  [$label]  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    Write-Host "  CMD: $cmd"
    Write-Host "============================================================"
    Invoke-Expression "$cmd *>&1 | Tee-Object -FilePath `"$logFile`""
    if ($LASTEXITCODE -ne 0) {
        Write-Host "!! [$label] FAILED (exit $LASTEXITCODE)"
        exit $LASTEXITCODE
    }
    Write-Host "  [$label] DONE  $(Get-Date -Format 'HH:mm:ss')"
}

# ── Step 1: CLSC with existing results (B, C, D, E) ────────────────────────
Run-Step "CLSC (B/C/D/E)" `
    "$python -m src.eval.clsc --bench all --limit 500" `
    "$logsDir\clsc_before_f.log"

# ── Step 2: DALR data preparation ──────────────────────────────────────────
Run-Step "DALR data prep" `
    "$python -m src.data.make_dalr_data" `
    "$logsDir\dalr_data_prep.log"

# ── Step 3: DALR training ───────────────────────────────────────────────────
Run-Step "DALR training (setup f)" `
    "$python -m src.train.sft --setup f" `
    "$logsDir\train_f.log"

# ── Step 4: DALR eval ───────────────────────────────────────────────────────
Run-Step "DALR eval (setup f, limit 500)" `
    "$python -m src.eval.evaluate --setup f --bench all --limit 500 --tag final" `
    "$logsDir\eval_f.log"

# ── Step 5: CLSC re-run with F included ────────────────────────────────────
Run-Step "CLSC (B/C/D/E/F — final)" `
    "$python -m src.eval.clsc --bench all --limit 500" `
    "$logsDir\clsc_final.log"

# ── Done ────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "============================================================"
Write-Host "  ALL STEPS COMPLETE  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "  Results → results/"
Write-Host "  Logs    → logs/"
Write-Host "============================================================"
