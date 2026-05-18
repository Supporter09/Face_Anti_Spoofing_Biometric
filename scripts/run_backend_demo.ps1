$ErrorActionPreference = "Stop"

$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
$env:CUDNNPATH = "C:\Program Files\NVIDIA\CUDNN\v9.13\bin\13.0"
$env:PATH = "$env:CUDA_PATH\bin;$env:CUDA_PATH\bin\x64;$env:CUDA_PATH\libnvvp;$env:CUDNNPATH;$env:PATH"

$RootDir = Resolve-Path "$PSScriptRoot\.."
Set-Location $RootDir

$PythonPath = ".\.venv\Scripts\python.exe"

if (!(Test-Path $PythonPath)) {
    Write-Host "Missing .venv. Create it and install dependencies first."
    exit 1
}

if (-not $env:LIVENESS_MODEL_PATH) {
    $env:LIVENESS_MODEL_PATH = "$RootDir\Kaggle_Outputs\celeba_spoof_training_full\best_model_scripted.pt"
}

if (!(Test-Path $env:LIVENESS_MODEL_PATH)) {
    Write-Host "Model checkpoint not found: $env:LIVENESS_MODEL_PATH"
    exit 1
}

if (-not $env:LIVENESS_SPOOF_THRESHOLD) {
    $env:LIVENESS_SPOOF_THRESHOLD = "0.10"
}

if (-not $env:LIVENESS_LIVE_THRESHOLD) {
    $env:LIVENESS_LIVE_THRESHOLD = "0.80"
}

Write-Host "Starting backend with:"
Write-Host "  LIVENESS_MODEL_PATH=$env:LIVENESS_MODEL_PATH"
Write-Host "  LIVENESS_SPOOF_THRESHOLD=$env:LIVENESS_SPOOF_THRESHOLD"
Write-Host "  LIVENESS_LIVE_THRESHOLD=$env:LIVENESS_LIVE_THRESHOLD"

& $PythonPath -m uvicorn services.api.app:app --reload --host 127.0.0.1 --port 8000