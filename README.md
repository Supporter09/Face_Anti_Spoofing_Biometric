# Face Anti-Spoofing Biometric

RGB-only face liveness MVP with a web demo, notebook-first training workflow, and a deferred paper track.

## Current Focus

1. Integrate real backend face detection and liveness inference.
2. Build reusable preprocessing, training, and evaluation code for notebooks.
3. Delay paper writing until stable benchmark results are available.

## Detection and Landmarking

Yes, the real-time face detection and landmark points (eyes, nose, mouth corners) are handled by a pretrained third-party detector (`insightface`), and this runs in the backend.

## Liveness Model Scope

The liveness model only processes face crops after detection, resized to model input size (`80x80` in this scaffold).

## What Exists Now

- FastAPI liveness API scaffold with health and inference endpoints.
- Detector integration point using pretrained `insightface` (`bbox + 5 landmarks`).
- TorchScript liveness model loader with score-to-label mapping (`live/spoof/uncertain`).
- Reusable preprocessing/training/evaluation modules in `src/fas/`.
- React webcam demo scaffold wired to API contract.
- Two notebook tracks:
- Local/reusable notebooks that import `src/fas/` modules
- Kaggle self-contained notebooks with all code embedded in notebook cells

## Notebook Tracks

Reusable/local track:

- `notebooks/01_data_prep_celeba_spoof.ipynb`
- `notebooks/02_train_minifasnet.ipynb`
- `notebooks/03_eval_minifasnet.ipynb`
- `notebooks/04_demo_inference.ipynb`

Kaggle self-contained track:

- `notebooks/kaggle_full_01_data_prep.ipynb`
- `notebooks/kaggle_full_02_train.ipynb`
- `notebooks/kaggle_full_03_eval.ipynb`
- `notebooks/kaggle_full_04_backend_sanity.ipynb`

## Backend Configuration

Environment variables:

- `LIVENESS_MODEL_PATH`: path to TorchScript liveness checkpoint (`.pt`)
- `LIVENESS_LIVE_THRESHOLD`: currently configured in code defaults (`0.9`)
- `LIVENESS_SPOOF_THRESHOLD`: currently configured in code defaults (`0.3`)

## Quick Start

### Backend baseline

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/uvicorn services.api.app:app --reload
```

### Backend ML dependencies

```bash
.venv/bin/pip install -e '.[ml]'
```

### Frontend

```bash
cd apps/web
npm install
npm run dev
```

The frontend expects the API at `http://127.0.0.1:8000` by default.

## Verification

```bash
.venv/bin/python -m pytest tests/api -q
.venv/bin/python -m compileall src services/api
.venv/bin/python services/api/benchmark_smoke.py
cd apps/web && npm run typecheck
```
