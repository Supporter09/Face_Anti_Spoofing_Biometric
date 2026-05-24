# Session Handoff

## Restart Path

1. Read `AGENTS.md` and `docs/RESEARCH_PLAN.md`.
2. Run `./init.sh`.
3. Install ML extras: `.venv/bin/pip install -e '.[ml]'`.
4. Use checkpoint from `Kaggle_Outputs/context_mobilenetv2_224/mobilenetv2_context_scripted.pt`.
5. `scripts/run_backend_demo.sh` uses this checkpoint by default; override `LIVENESS_MODEL_PATH` only for legacy comparisons.
6. Start backend: `./scripts/run_backend_demo.sh`.
7. Start frontend (new terminal): `./scripts/run_frontend_demo.sh`.
8. Open `http://127.0.0.1:5173` and run webcam tests.
9. Open `http://127.0.0.1:5173?legacy=1` only when checking the old MVP UI.
10. Continue from `progress.md`.

## Current Focus

- Hybrid context + challenge liveness frontend is implemented locally.
- Context model artifact: `Kaggle_Outputs/context_mobilenetv2_224/mobilenetv2_context_scripted.pt`.
- Trained passive threshold: `T_PASSIVE = 0.4`.
- Next human action: run real webcam sessions through the default UI and collect Tier 2 eval data.

## Known Risks

- `insightface` and `torch` are optional extras and may not be installed by default.
- Detector initialization depends on runtime availability of ONNX runtime backend.
- Browser session UX has not been manually validated with a real camera after the React rewrite.
- Active challenge thresholds may need adjustment after real webcam pose traces.
- Frontend package installs may fail without writable npm cache unless `HOME`/`npm_config_cache` are redirected.
- Local notebook smoke was not run: `jupyter`/`nbconvert` is not installed locally and the Kaggle CelebA-Spoof path is not mounted.
- Sandboxed local socket access blocks uvicorn/client smoke tests; use escalated commands for local API smoke verification.
