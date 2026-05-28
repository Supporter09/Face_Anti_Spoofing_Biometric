$ErrorActionPreference = "Stop"

$ROOT_DIR = Resolve-Path (Join-Path $PSScriptRoot "..")
$WEB_DIR = Join-Path $ROOT_DIR "apps\web"

Set-Location $WEB_DIR

if (-not $env:VITE_API_BASE_URL) {
    $env:VITE_API_BASE_URL = "http://127.0.0.1:8000"
}

Write-Host "Starting frontend with VITE_API_BASE_URL=$env:VITE_API_BASE_URL"

npm run dev -- --host 127.0.0.1 --port 5173