#!/usr/bin/env bash
set -euo pipefail

printf 'Project: Face Anti-Spoofing Biometric\n'
printf 'Python: '; python3 --version || true
printf 'Node: '; node --version || true
printf 'npm: '; npm --version || true
printf '\nVerification commands:\n'
printf '  python -m pytest tests/api -q\n'
printf '  python -m compileall src services/api\n'
printf '  python services/api/benchmark_smoke.py\n'
printf '  (cd apps/web && npm run lint)\n'
printf '  (cd apps/web && npm run typecheck)\n'
