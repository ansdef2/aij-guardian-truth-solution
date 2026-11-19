# Скачивает веса в solution\model и собирает архив для платформы (Windows PowerShell).
#   powershell -ExecutionPolicy Bypass -File tools\run_all.ps1
#   powershell -ExecutionPolicy Bypass -File tools\run_all.ps1 -Entry predict_border.py
param([string]$Entry = "predict.py")
$repo  = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$model = Join-Path $repo "solution\model"
$name  = if ($Entry -eq "predict.py") { "final" } else { "final_" + ([IO.Path]::GetFileNameWithoutExtension($Entry) -replace '^predict_', '') }
$zip   = Join-Path $repo "dist\$name.zip"

$py = "py"
if (-not (Get-Command py -ErrorAction SilentlyContinue)) { $py = "python" }

Write-Host "[1/3] Ставлю huggingface_hub..." -ForegroundColor Cyan
& $py -m pip install -U huggingface_hub hf_xet
if ($LASTEXITCODE -ne 0) { Write-Host "Не удалось установить huggingface_hub" -ForegroundColor Red; exit 1 }

Write-Host "[2/3] Скачиваю веса Qwen3.6-35B-A3B-FP8 (~35 ГБ) в $model ..." -ForegroundColor Cyan
& $py -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.6-35B-A3B-FP8', local_dir=r'$model', max_workers=16); print('веса скачаны')"
if ($LASTEXITCODE -ne 0) { Write-Host "Скачивание прервалось — запустите скрипт ещё раз, оно продолжится" -ForegroundColor Red; exit 1 }

Write-Host "[3/3] Собираю $zip (точка входа: $Entry) ..." -ForegroundColor Cyan
& $py (Join-Path $repo "tools\build_zip.py") (Join-Path $repo "solution") $zip --entry $Entry
if ($LASTEXITCODE -ne 0) { Write-Host "Сборка архива не удалась" -ForegroundColor Red; exit 1 }
Write-Host "ГОТОВО: $zip" -ForegroundColor Green
