#!/usr/bin/env bash
# One-time setup on macOS/Linux: creates .venv and installs everything. Run:  bash setup.sh
set -e
cd "$(dirname "$0")"
rm -rf .venv
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[notebooks,dev]"
.venv/bin/python -m ipykernel install --user --name rto-v3 --display-name "RTO v3 (.venv)"
echo
echo "Setup done. Next: paste your key into .env (LLM_API_KEY=), then run:  bash run.sh"
