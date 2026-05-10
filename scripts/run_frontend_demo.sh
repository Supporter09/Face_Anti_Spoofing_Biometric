#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEB_DIR="${ROOT_DIR}/apps/web"
cd "${WEB_DIR}"

export VITE_API_BASE_URL="${VITE_API_BASE_URL:-http://127.0.0.1:8000}"
echo "Starting frontend with VITE_API_BASE_URL=${VITE_API_BASE_URL}"

exec npm run dev -- --host 127.0.0.1 --port 5173
