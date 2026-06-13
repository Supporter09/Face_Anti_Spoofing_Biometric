@echo off
setlocal enabledelayedexpansion

set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
set "CUDNNPATH=C:\Program Files\NVIDIA\CUDNN\v9.13\bin\13.0"
set "PATH=%CUDA_PATH%\bin;%CUDA_PATH%\bin\x64;%CUDA_PATH%\libnvvp;%CUDNNPATH%;%PATH%"

REM Get project root directory (parent of this script's folder)
set "ROOT_DIR=%~dp0.."
set "WEB_DIR=%ROOT_DIR%\apps\web"

cd /d "%WEB_DIR%"

REM Set default VITE_API_BASE_URL if not already defined
if "%VITE_API_BASE_URL%"=="" (
    set "VITE_API_BASE_URL=http://127.0.0.1:8000"
)

echo Starting frontend with VITE_API_BASE_URL=%VITE_API_BASE_URL%

REM Start Vite dev server
call npm run dev -- --host 127.0.0.1 --port 5173