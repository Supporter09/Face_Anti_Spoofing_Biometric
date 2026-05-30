# Progress Log

## Current State

- Active feature: hybrid context + challenge liveness
- Status: context MobileNetV2 training artifacts are available; backend defaults and frontend session UI are wired for the new flow (May 24, 2026)
- Branch target: `feat/fas-context-challenge`

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

- Manual webcam validation of the new session UI.
- Tier 2 webcam eval collection and reporting.
- Error analysis of any real webcam false accepts/false rejects from the context + challenge flow.

## Completed This Session

- Added debug capture API route `POST /v1/liveness/frame/debug` that stores request frame + response metadata per `session_id` under `reports/liveness_debug_frames` (override root with `LIVENESS_DEBUG_CAPTURE_ROOT`).
- Updated runtime liveness model input size default to `224` to match the context MobileNetV2 training/inference contract.
- Updated debug artifacts to save:
  - overlay frame with score + bbox + 5 landmarks
  - context-padded crop used by backend model pipeline
  - resized model input image and JSON metadata
- Wired frontend session flow to support debug capture mode via URL flags:
  - `?capture_debug=1`
  - `&capture_session=<name>`
  When enabled, frame submissions go to `/v1/liveness/frame/debug`.
- Added API test coverage for debug route persistence:
  - `tests/api/test_frame_endpoint.py::test_frame_debug_endpoint_saves_request_and_metadata`
- Verified:
  - `python -m pytest tests/api -q`
  - `cd apps/web && npm run typecheck`
  - `cd apps/web && npm run lint`

- Switched `scripts/run_backend_demo.sh` default model to `Kaggle_Outputs/context_mobilenetv2_224/mobilenetv2_context_scripted.pt`.
- Set demo thresholds to bracket the trained decision threshold: spoof `0.3`, live `0.5`, with `T_PASSIVE=0.4` used in browser fusion.
- Added Vitest/jsdom/testing-library setup for `apps/web`.
- Added frontend session module:
  - `types.ts`
  - `fusion.ts` with passive/challenge fusion and unit tests
  - `useSession.ts` state machine for countdown, forward, randomized turns, evaluation, and result
  - `SessionView.tsx`
  - `ResultView.tsx`
- Rewrote `App.tsx` so the new session UI is the default and the previous MVP UI remains available at `?legacy=1`.
- Added standalone webcam collection tool: `scripts/collect_webcam_eval.py`.
- Added Vietnamese webcam collection protocol: `docs/WEBCAM_COLLECTION_PROTOCOL.md`.
- Added Tier 1 CelebA-Spoof comparison report: `reports/tier1_celeba_spoof_comparison.md`.
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

1. Run manual webcam sessions against the new browser flow and record any UX/pose threshold issues.
2. Collect Tier 2 webcam eval data with `scripts/collect_webcam_eval.py`.
3. Populate Tier 2 frame-level and session-level reports after collection.
4. Keep the legacy UI available for regression checks via `http://127.0.0.1:5173?legacy=1`.
