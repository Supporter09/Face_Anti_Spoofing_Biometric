$ErrorActionPreference = "Stop"

$ROOT_DIR = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ROOT_DIR

$PythonPath = Join-Path $ROOT_DIR ".venv\Scripts\python.exe"
$UvicornPath = Join-Path $ROOT_DIR ".venv\Scripts\uvicorn.exe"

if (-not (Test-Path $PythonPath)) {
    Write-Host "Missing .venv. Create it and install dependencies first."
    Write-Host "Run:"
    Write-Host "  python -m venv .venv"
    Write-Host "  .\.venv\Scripts\activate"
    Write-Host "  pip install -r requirements.txt"
    Write-Host "  pip install -e `".[ml]`""
    Write-Host "  pip install psycopg2-binary python-dotenv"
    exit 1
}

if (-not (Test-Path $UvicornPath)) {
    Write-Host "uvicorn.exe not found in .venv."
    Write-Host "Run:"
    Write-Host "  .\.venv\Scripts\activate"
    Write-Host "  pip install uvicorn"
    exit 1
}

# ==============================
# Liveness model config
# ==============================

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

# ==============================
# Authentication database config
# ==============================

if (-not $env:DATABASE_URL) {
    # Change these values if your PostgreSQL username/password/database are different.
    $DB_USER = "face_auth"
    $DB_PASSWORD = "face_auth_pass"
    $DB_HOST = "localhost"
    $DB_PORT = "5432"
    $DB_NAME = "face_anti_spoofing"

    $env:DATABASE_URL = "postgresql://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:${DB_PORT}/${DB_NAME}"
}

# Face authentication threshold
if (-not $env:FACE_AUTH_THRESHOLD) {
    $env:FACE_AUTH_THRESHOLD = "0.50"
}

# ==============================
# Optional debug mode
# ==============================

# Debug mode: set LIVENESS_DEBUG_DIR to save frames + model info per scan
# Example:
# $env:LIVENESS_DEBUG_DIR="C:\tmp\fas_debug"
# .\scripts\run_backend_demo.ps1

# ==============================
# Print config
# ==============================

Write-Host "Starting backend with:"
Write-Host "  ROOT_DIR=$ROOT_DIR"
Write-Host "  LIVENESS_MODEL_PATH=$env:LIVENESS_MODEL_PATH"
Write-Host "  LIVENESS_SPOOF_THRESHOLD=$env:LIVENESS_SPOOF_THRESHOLD"
Write-Host "  LIVENESS_LIVE_THRESHOLD=$env:LIVENESS_LIVE_THRESHOLD"
Write-Host "  FACE_AUTH_THRESHOLD=$env:FACE_AUTH_THRESHOLD"
Write-Host "  DATABASE_URL=$env:DATABASE_URL"

if ($env:LIVENESS_DEBUG_DIR) {
    Write-Host "  LIVENESS_DEBUG_DIR=$env:LIVENESS_DEBUG_DIR  (debug ON, max 300 frames)"
}

# ==============================
# Optional: quick import check
# ==============================

Write-Host ""
Write-Host "Checking Python dependencies..."

& $PythonPath -c "import psycopg2; print('  psycopg2 ok')"

Write-Host ""
Write-Host "Running backend..."
Write-Host ""

& $UvicornPath services.api.app:app --reload --host 127.0.0.1 --port 8000