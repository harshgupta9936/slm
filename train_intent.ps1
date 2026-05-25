# Build a large intent dataset and fine-tune the classifier (target: under 8 hours on CPU).
# Usage:  .\train_intent.ps1
#         .\train_intent.ps1 -Fast          # logistic only (~5 min)
#         .\train_intent.ps1 -Epochs 3      # shorter hard run

param(
    [switch]$Fast,
    [int]$Epochs = 4,
    [int]$BatchSize = 32
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "==> Building intent dataset (--hard)..." -ForegroundColor Cyan
python 04_intent_dataset.py --movies databse.csv --hard --chat-addon
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$mode = if ($Fast) { "fast" } else { "hard" }
Write-Host "==> Training intent classifier (mode=$mode)..." -ForegroundColor Cyan
python train_intent_classifier.py --mode $mode --epochs $Epochs --batch-size $BatchSize --max-hours 7.5
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`nDone. Restart web_chat.py to use the new intent_model/." -ForegroundColor Green
