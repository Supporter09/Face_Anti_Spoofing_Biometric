# Progress Log

## Current State

- Active feature: real detector and liveness inference integration
- Status: backend pipeline shape implemented, model checkpoint wiring pending
- Branch target: `feat/liveness-mvp-scaffold`

## Completed

- Integrated service pipeline structure: decode -> detect -> crop -> liveness score -> label.
- Added pretrained detector adapter (`insightface`) with bbox and landmark extraction.
- Added TorchScript liveness loader and score conversion logic.
- Added reusable preprocessing/training/eval modules under `src/fas/`.
- Added tests for live/spoof decision mapping and API contract safety.

## In Progress

- Wiring a real trained checkpoint from notebook training output.
- Populating training and eval notebooks with concrete dataset-specific cells.

## Next

1. Install `.[ml]` dependencies and run backend with actual images.
2. Produce first train/eval artifacts from notebooks.
3. Connect trained checkpoint via `LIVENESS_MODEL_PATH` and validate end-to-end webcam inference.
