#!/usr/bin/env bash
# Downloads a free open model (from Hugging Face / the Ollama library) and makes a 16k-context copy for the copilot.
# Usage:  bash setup_ollama.sh                      -> Llama 3.1 8B  (named llama3.1-16k)
#         bash setup_ollama.sh qwen2.5:7b qwen25-16k  -> any other model, then set OLLAMA_MODEL=qwen25-16k in .env
#         bash setup_ollama.sh hf.co/bartowski/Llama-3.1-8B-Instruct-GGUF:Q4_K_M llama3.1-16k   (straight from Hugging Face)
set -e
BASE="${1:-llama3.1:8b}"
NAME="${2:-llama3.1-16k}"
command -v ollama >/dev/null || { echo "Install Ollama first:  brew install --cask ollama   (or https://ollama.com/download), open the app once, then run this again."; exit 1; }
ollama list >/dev/null 2>&1 || { echo "Ollama is not running. Open the Ollama app, wait a few seconds, then run this again."; exit 1; }
ollama pull "$BASE"
# Ollama answers with a 2048-token window by default. One copilot turn sends about 8000 tokens, so raise it.
TMP="$(mktemp)"
printf 'FROM %s\nPARAMETER num_ctx 16384\nPARAMETER temperature 0\n' "$BASE" > "$TMP"
ollama create "$NAME" -f "$TMP"
rm -f "$TMP"
echo
echo "Ready: model '$NAME'. Restart the UI (bash run.sh) and pick it in the Model dropdown."
