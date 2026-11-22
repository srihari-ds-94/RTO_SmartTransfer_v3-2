#!/usr/bin/env bash
# Checks the model connection, then starts the UI at http://127.0.0.1:8000. Run:  bash run.sh
cd "$(dirname "$0")"
PY=.venv/bin/python
[ -x "$PY" ] || { echo "No .venv yet. Run first:  bash setup.sh"; exit 1; }
$PY -m rto_copilot.check || echo "(Check failed: the page still opens for case checks, chat needs the fixes above.)"
echo "Opening http://127.0.0.1:8000  (Ctrl+C to stop)"
(sleep 3; open http://127.0.0.1:8000 2>/dev/null || true) &
$PY -m rto_copilot.api
