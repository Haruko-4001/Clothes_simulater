#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python -m pytest tests -q
python -m app.main --mode mock --frames 30 --headless --log var/logs/smoke.jsonl >/dev/null
echo "smoke: PASS (30 mock frames)"
