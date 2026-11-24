#!/usr/bin/env bash
# One-time setup on macOS/Linux: creates .venv and installs everything. Run:  bash setup.sh
# Needs Python 3.10 or newer. The macOS system python3 is 3.9, so this looks for a newer one.
set -e
cd "$(dirname "$0")"

PY=""
for c in python3.13 python3.12 python3.11 python3.10 \
         /opt/homebrew/bin/python3.13 /opt/homebrew/bin/python3.12 /opt/homebrew/bin/python3.11 /opt/homebrew/bin/python3.10 \
         /usr/local/bin/python3.13 /usr/local/bin/python3.12 /usr/local/bin/python3.11 /usr/local/bin/python3.10 \
         python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; then
    PY="$c"; break
  fi
done

if [ -z "$PY" ]; then
  echo "No Python 3.10 or newer found (python3 here is $(python3 --version 2>&1))."
  echo "Install one, then run this script again:"
  echo "  Mac with Homebrew:   brew install python@3.12"
  echo "  Mac without Homebrew: download Python 3.12 from https://www.python.org/downloads/macos/"
  exit 1
fi
echo "Using $PY ($($PY --version))"

rm -rf .venv
"$PY" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[notebooks,dev]"
.venv/bin/python -m ipykernel install --user --name rto-v3 --display-name "RTO v3 (.venv)"
echo
echo "Setup done. Next: paste your key into .env (LLM_API_KEY=), then run:  bash run.sh"
