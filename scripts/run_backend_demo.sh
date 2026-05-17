#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ ! -x ".venv/bin/python" ]]; then
  echo "Missing .venv. Create it and install dependencies first."
  exit 1
fi

MODEL_PATH="${LIVENESS_MODEL_PATH:-${ROOT_DIR}/Kaggle_Outputs/mobilenetv2_fas_training/mobilenetv2_fas_scripted.pt}"
if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "Model checkpoint not found: ${MODEL_PATH}"
  exit 1
fi

export LIVENESS_MODEL_PATH="${MODEL_PATH}"
export LIVENESS_SPOOF_THRESHOLD="${LIVENESS_SPOOF_THRESHOLD:-0.35}"
export LIVENESS_LIVE_THRESHOLD="${LIVENESS_LIVE_THRESHOLD:-0.85}"

echo "Starting backend with:"
echo "  LIVENESS_MODEL_PATH=${LIVENESS_MODEL_PATH}"
echo "  LIVENESS_SPOOF_THRESHOLD=${LIVENESS_SPOOF_THRESHOLD}"
echo "  LIVENESS_LIVE_THRESHOLD=${LIVENESS_LIVE_THRESHOLD}"

exec .venv/bin/uvicorn services.api.app:app --reload --host 127.0.0.1 --port 8000
