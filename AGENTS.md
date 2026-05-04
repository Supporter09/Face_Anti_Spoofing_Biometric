# AGENTS.md

This repository builds an RGB-only face liveness MVP, then extends it into biometric authentication.

## Startup Workflow

Before writing code:
1. Read this file and `docs/ARCHITECTURE.md`.
2. Read `docs/PRODUCT.md` and `docs/RESEARCH_PLAN.md` for scope.
3. Run `./init.sh` to see the expected toolchain and verification commands.
4. Read `feature_list.json`, `progress.md`, and `session_handoff.md`.

## Working Rules

- Work on one feature area at a time.
- Keep training workflows notebook-first, but move reusable logic into `src/`.
- Do not claim completion without running the relevant verification commands.
- Keep large datasets and model weights out of git.
- Update progress and handoff files before ending a session.

## Required Artifacts

- `feature_list.json`: active feature tracker.
- `progress.md`: current execution status.
- `session_handoff.md`: restart path and blockers.
- `reports/`: generated benchmark and evaluation evidence.

## Definition of Done

A feature is done when:
- [ ] Implementation is complete.
- [ ] Verification has passed.
- [ ] Evidence is captured in `reports/` or docs.
- [ ] Restart instructions remain accurate.

## Verification Commands

- `python -m pytest tests/api -q`
- `python -m compileall src services/api`
- `python services/api/benchmark_smoke.py`
- `cd apps/web && npm run lint`
- `cd apps/web && npm run typecheck`

## End of Session

1. Update `progress.md`.
2. Update `feature_list.json` status.
3. Record blockers and next steps in `session_handoff.md`.
4. Keep the repository restartable from a clean checkout.
