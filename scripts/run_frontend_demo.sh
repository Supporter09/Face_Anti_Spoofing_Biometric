#!/usr/bin/env bash
set -euo pipefail

export CUDA_PATH="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0"
export CUDNNPATH="C:/Program Files/NVIDIA/CUDNN/v9.13/bin/13.0"
export PATH="$CUDA_PATH/bin:$CUDA_PATH/bin/x64:$CUDA_PATH/lib64:$CUDNNPATH:$PATH"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEB_DIR="${ROOT_DIR}/apps/web"
cd "${WEB_DIR}"

export VITE_API_BASE_URL="${VITE_API_BASE_URL:-http://127.0.0.1:8000}"
echo "Starting frontend with VITE_API_BASE_URL=${VITE_API_BASE_URL}"

exec npm run dev -- --host 127.0.0.1 --port 5173
