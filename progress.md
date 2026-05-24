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
- Hybrid context + challenge liveness implementation is halted at Task 10: notebook 07 is prepared and committed; Kaggle full training is waiting for human upload/run.

## Completed This Session

- Phase 1 backend helpers and pose completed:
  - `pad_to_square_then_resize`
  - context crop prep support
  - 5-landmark head pose estimation
- Phase 2 backend integration completed:
  - `FaceDetection.context_crop_bgr`
  - detector context crop output
  - pose fields in response schema
  - service uses context crop and populates pose
  - `/v1/liveness/frame` endpoint
  - backend smoke with generated blank frame returned pose fields (`pose_ok=false`, no face)
- Task 10 notebook preparation completed:
  - `notebooks/kaggle_full_07_train_context_mobilenetv2.ipynb`

## Next

1. Upload `notebooks/kaggle_full_07_train_context_mobilenetv2.ipynb` to Kaggle with the CelebA-Spoof dataset attached.
2. Run the notebook end-to-end with full data (`limit=None`).
3. Download outputs into `Kaggle_Outputs/context_mobilenetv2_224/`.
4. Resume from Task 10 Step 16, then continue Phase 4 after the model artifacts exist.
