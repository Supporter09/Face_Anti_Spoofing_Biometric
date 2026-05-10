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

- Calibrate threshold policy using `Kaggle_Outputs/celeba_spoof_eval_full_03/threshold_metrics.csv`.
- Validate backend decisions with real webcam frames using loaded scripted model.
- Reduce false-reject pressure on live users while controlling spoof-accepted risk.

## Known Risks

- `insightface` and `torch` are optional extras and may not be installed by default.
- Detector initialization depends on runtime availability of ONNX runtime backend.
- Current fixed thresholds (`live>=0.9`, `spoof<=0.3`) produce many live samples labeled spoof on eval data.
- Single-threshold best ACER from current run (`0.1`) still indicates meaningful class overlap; more training/error analysis is needed.
- Frontend package installs may fail without writable npm cache unless `HOME`/`npm_config_cache` are redirected.
