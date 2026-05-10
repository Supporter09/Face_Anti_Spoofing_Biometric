# Progress Log

## Current State

- Active feature: model validation and backend threshold calibration
- Status: local MVP runtime wired (backend + frontend + scripted model) and ready for webcam demo tests (May 10, 2026)
- Branch target: `feat/liveness-mvp-scaffold`

## Completed

- Integrated service pipeline structure: decode -> detect -> crop -> liveness score -> label.
- Added pretrained detector adapter (`insightface`) with bbox and landmark extraction.
- Added TorchScript liveness loader and score conversion logic.
- Added reusable preprocessing/training/eval modules under `src/fas/`.
- Added tests for live/spoof decision mapping and API contract safety.
- Produced first Kaggle training artifacts in `Kaggle_Outputs/celeba_spoof_training_full/`:
  - `best_model.pt`
  - `best_model_scripted.pt`
  - `history.json`
  - `run_summary.json` (`best_val_acc=0.9863875404530744`)
- Produced first Kaggle evaluation artifacts in `Kaggle_Outputs/celeba_spoof_eval_full_03/`:
  - `predictions.csv`
  - `threshold_metrics.csv`
  - `best_threshold.json` (best ACER threshold from single-threshold sweep: `0.1`)
- Ran backend-style Kaggle sanity notebook with `insightface` detection + landmarks + scripted liveness model.
- Added env-driven backend threshold support (`LIVENESS_LIVE_THRESHOLD`, `LIVENESS_SPOOF_THRESHOLD`) with validation and tests.
- Bootstrapped local `.venv` with `.[dev,ml]` dependencies and installed web dependencies.
- Added demo startup scripts:
  - `scripts/run_backend_demo.sh`
  - `scripts/run_frontend_demo.sh`
- Verified `/health` endpoint and frontend dev server startup locally.

## In Progress

- Threshold policy selection for production-like behavior (`live`, `spoof`, `uncertain`).
- Error analysis of hard live samples scored as spoof and hard spoof samples scored as live.

## Next

1. Run real webcam sessions and record outcomes by condition (normal light, low light, phone-screen replay, printed photo).
2. Measure live false-reject vs spoof false-accept tradeoff for current candidate thresholds (`spoof=0.10`, `live=0.80`).
3. Save benchmark/eval evidence under `reports/` and update threshold recommendation.
4. Launch second training iteration focused on hard examples and class overlap reduction.
