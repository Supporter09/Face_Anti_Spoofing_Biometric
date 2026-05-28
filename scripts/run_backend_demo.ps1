$ErrorActionPreference = "Stop"

$ROOT_DIR = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ROOT_DIR

$PythonPath = Join-Path $ROOT_DIR ".venv\Scripts\python.exe"

if (-not (Test-Path $PythonPath)) {
    Write-Host "Missing .venv. Create it and install dependencies first."
    exit 1
}

if ($env:LIVENESS_MODEL_PATH) {
    $MODEL_PATH = $env:LIVENESS_MODEL_PATH
} else {
    $MODEL_PATH = Join-Path $ROOT_DIR "Kaggle_Outputs\context_mobilenetv2_224\mobilenetv2_context_scripted.pt"
}

if (-not (Test-Path $MODEL_PATH)) {
    Write-Host "Model checkpoint not found: $MODEL_PATH"
    exit 1
}

$env:LIVENESS_MODEL_PATH = $MODEL_PATH

if (-not $env:LIVENESS_SPOOF_THRESHOLD) {
    $env:LIVENESS_SPOOF_THRESHOLD = "0.3"
}

if (-not $env:LIVENESS_LIVE_THRESHOLD) {
    $env:LIVENESS_LIVE_THRESHOLD = "0.5"
}

# Debug mode: set LIVENESS_DEBUG_DIR to save frames + model info per scan
# Example:
# $env:LIVENESS_DEBUG_DIR="C:\tmp\fas_debug"
# .\scripts\run_backend_demo.ps1

if ($env:LIVENESS_DEBUG_DIR) {
    Write-Host "  LIVENESS_DEBUG_DIR=$env:LIVENESS_DEBUG_DIR  (debug ON, max 300 frames)"
}

Write-Host "Starting backend with:"
Write-Host "  LIVENESS_MODEL_PATH=$env:LIVENESS_MODEL_PATH"
Write-Host "  LIVENESS_SPOOF_THRESHOLD=$env:LIVENESS_SPOOF_THRESHOLD"
Write-Host "  LIVENESS_LIVE_THRESHOLD=$env:LIVENESS_LIVE_THRESHOLD"

& .\.venv\Scripts\uvicorn.exe services.api.app:app --reload --host 127.0.0.1 --port 8000