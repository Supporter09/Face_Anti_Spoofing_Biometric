# Session Handoff

## Restart Path

1. Read `AGENTS.md` and `docs/RESEARCH_PLAN.md`.
2. Run `./init.sh`.
3. Install ML extras: `.venv/bin/pip install -e '.[ml]'`.
4. Use checkpoint from `Kaggle_Outputs/celeba_spoof_training_full/best_model_scripted.pt`.
5. Set `LIVENESS_MODEL_PATH` to the scripted checkpoint path.
6. Start backend: `./scripts/run_backend_demo.sh`.
7. Start frontend (new terminal): `./scripts/run_frontend_demo.sh`.
8. Open `http://127.0.0.1:5173` and run webcam tests.
9. Continue from `progress.md`.

## Current Focus

- Hybrid context + challenge liveness implementation is halted at Task 10.
- Notebook 07 is prepared and committed: `notebooks/kaggle_full_07_train_context_mobilenetv2.ipynb`.
- Next human action: upload notebook 07 to Kaggle, attach CelebA-Spoof, run full data, then download outputs to `Kaggle_Outputs/context_mobilenetv2_224/`.

## Known Risks

- `insightface` and `torch` are optional extras and may not be installed by default.
- Detector initialization depends on runtime availability of ONNX runtime backend.
- Current fixed thresholds (`live>=0.9`, `spoof<=0.3`) produce many live samples labeled spoof on eval data.
- Single-threshold best ACER from current run (`0.1`) still indicates meaningful class overlap; more training/error analysis is needed.
- Frontend package installs may fail without writable npm cache unless `HOME`/`npm_config_cache` are redirected.
- Local notebook smoke was not run: `jupyter`/`nbconvert` is not installed locally and the Kaggle CelebA-Spoof path is not mounted.
- Sandboxed local socket access blocks uvicorn/client smoke tests; use escalated commands for local API smoke verification.
