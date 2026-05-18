@echo off
setlocal enabledelayedexpansion


set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
set "CUDNNPATH=C:\Program Files\NVIDIA\CUDNN\v9.13\bin\13.0"
set "PATH=%CUDA_PATH%\bin;%CUDA_PATH%\bin\x64;%CUDA_PATH%\libnvvp;%CUDNNPATH%;%PATH%"

REM Get project root directory (parent of this script's folder)
set "ROOT_DIR=%~dp0.."
cd /d "%ROOT_DIR%"

REM Check if virtual environment exists
if not exist ".venv\Scripts\python.exe" (
    echo Missing .venv. Create it and install dependencies first.
    exit /b 1
)

REM Set default model path if not already defined
if "%LIVENESS_MODEL_PATH%"=="" (
    set "MODEL_PATH=%ROOT_DIR%\Kaggle_Outputs\celeba_spoof_training_full\best_model_scripted.pt"
) else (
    set "MODEL_PATH=%LIVENESS_MODEL_PATH%"
)

REM Check model file exists
if not exist "%MODEL_PATH%" (
    echo Model checkpoint not found: %MODEL_PATH%
    exit /b 1
)

REM Export environment variables
set "LIVENESS_MODEL_PATH=%MODEL_PATH%"

if "%LIVENESS_SPOOF_THRESHOLD%"=="" (
    set "LIVENESS_SPOOF_THRESHOLD=0.10"
)

if "%LIVENESS_LIVE_THRESHOLD%"=="" (
    set "LIVENESS_LIVE_THRESHOLD=0.80"
)

echo Starting backend with:
echo   LIVENESS_MODEL_PATH=%LIVENESS_MODEL_PATH%
echo   LIVENESS_SPOOF_THRESHOLD=%LIVENESS_SPOOF_THRESHOLD%
echo   LIVENESS_LIVE_THRESHOLD=%LIVENESS_LIVE_THRESHOLD%

REM Start uvicorn server
call ".venv\Scripts\uvicorn.exe" services.api.app:app --reload --host 127.0.0.1 --port 8000