# Session Handoff

## Restart Path

1. Read `AGENTS.md` and `docs/RESEARCH_PLAN.md`.
2. Run `./init.sh`.
3. Install ML extras: `.venv/bin/pip install -e '.[ml]'`.
4. Set `LIVENESS_MODEL_PATH` once checkpoint exists.
5. Start API: `.venv/bin/uvicorn services.api.app:app --reload`.
6. Continue from `progress.md`.

## Current Focus

- Replace fallback liveness score (`0.5`) with real model checkpoint inference.
- Build and validate CelebA-Spoof preprocessing/training/eval workflow.

## Known Risks

- `insightface` and `torch` are optional extras and may not be installed by default.
- Detector initialization depends on runtime availability of ONNX runtime backend.
- Notebooks are scaffolded and require dataset-specific path and label mapping work.
