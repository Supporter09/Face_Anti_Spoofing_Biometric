$ErrorActionPreference = "Stop"

$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
$env:CUDNNPATH = "C:\Program Files\NVIDIA\CUDNN\v9.13\bin\13.0"
$env:PATH = "$env:CUDA_PATH\bin;$env:CUDA_PATH\bin\x64;$env:CUDA_PATH\libnvvp;$env:CUDNNPATH;$env:PATH"

$RootDir = Resolve-Path "$PSScriptRoot\.."
$WebDir = Join-Path $RootDir "apps\web"

if (!(Test-Path $WebDir)) {
    Write-Host "Frontend directory not found: $WebDir"
    exit 1
}

Set-Location $WebDir

if (!(Test-Path "package.json")) {
    Write-Host "package.json not found in frontend directory."
    exit 1
}

if (!(Test-Path "node_modules")) {
    Write-Host "node_modules not found. Installing dependencies..."
    npm install
}

Write-Host "Starting frontend..."
Write-Host "Frontend URL: http://localhost:5173/"

npm run dev