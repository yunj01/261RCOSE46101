# Run a command detached from the SSH session.
# Survives SSH disconnect, laptop close, etc.
#
# Usage:
#   .\scripts\run_detached.ps1 -Cmd "venv\Scripts\python.exe -m src.train.sft --setup f_random" -LogName "train_f_random"
#   .\scripts\run_detached.ps1 -Cmd "..." -LogName "eval_g_hrm8k"
#
# Output:
#   logs/<LogName>.log         (stdout+stderr)
#   logs/<LogName>.pid         (PID for later checking)
#
# Check status:
#   Get-Content logs/<LogName>.log -Tail 30
#   Get-Process -Id (Get-Content logs/<LogName>.pid)   # alive if no error

param(
    [Parameter(Mandatory=$true)][string]$Cmd,
    [Parameter(Mandatory=$true)][string]$LogName
)

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$LogsDir = Join-Path $ProjectRoot "logs"
if (-not (Test-Path $LogsDir)) { New-Item -ItemType Directory -Path $LogsDir | Out-Null }

$LogFile = Join-Path $LogsDir "$LogName.log"
$PidFile = Join-Path $LogsDir "$LogName.pid"
$Wrapper = Join-Path $LogsDir "$LogName.cmd"

# Build a wrapper .cmd that cd's to project root and runs the command,
# redirecting both stdout and stderr to the log file.
$WrapperContent = @"
@echo off
cd /d "$ProjectRoot"
$Cmd 1>"$LogFile" 2>&1
"@
Set-Content -Path $Wrapper -Value $WrapperContent -Encoding ascii

# Start fully detached: no shell window, no parent process tie.
$Proc = Start-Process -FilePath "cmd.exe" -ArgumentList "/c", "`"$Wrapper`"" `
    -WindowStyle Hidden -PassThru

Set-Content -Path $PidFile -Value $Proc.Id -Encoding ascii

Write-Host "[detached] Started PID=$($Proc.Id)"
Write-Host "[detached] Log : $LogFile"
Write-Host "[detached] PID : $PidFile"
Write-Host ""
Write-Host "Check status with:"
Write-Host "  Get-Content `"$LogFile`" -Tail 30"
Write-Host "  Get-Process -Id $($Proc.Id) -ErrorAction SilentlyContinue"
