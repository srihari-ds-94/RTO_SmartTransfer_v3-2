# Using a second model

The copilot talks to any model that speaks the OpenAI chat-completions format. Three are built in; pick one in the
**Model** dropdown on the page. A model shows "(not ready)" until its setup is done, and the reason appears under the dropdown.

| Dropdown name | Where it runs | Cost | Needs |
|---|---|---|---|
| Qwen3-8B (course endpoint) | upGrad VM07 | free for the team | `LLM_API_KEY` in `.env` |
| Llama 3.1 8B (local, free) | your Mac, through Ollama | free | `bash setup_ollama.sh` once |
| Hugging Face router | Hugging Face servers | free accounts get a very small monthly credit (about $0.10), so use it for short demos only | `HF_TOKEN` in `.env` |

## Option A: open model from Hugging Face, running on your Mac (recommended, truly free)

1. Install Ollama: `brew install --cask ollama`, then open the Ollama app once. Keep it open.
2. In the project folder run `bash setup_ollama.sh`. It downloads Llama 3.1 8B (about 5 GB) and makes a copy named
   `llama3.1-16k` with a larger context window. The copy matters: Ollama's default window is 2048 tokens and one
   copilot turn sends about 8000, so without it the model silently drops most of the sources.
3. `.env` already has `OLLAMA_MODEL=llama3.1-16k`. Run `bash run.sh`. The check prints a line for each model.
4. Open the page, choose **Llama 3.1 8B (local, free)** in the Model box, and ask a question.

To use a model straight from a Hugging Face page, give the script its GGUF address:

    bash setup_ollama.sh hf.co/bartowski/Llama-3.1-8B-Instruct-GGUF:Q4_K_M llama3.1-16k

Any other model: `bash setup_ollama.sh qwen2.5:7b qwen25-16k`, then add it to `models.json` (copy `docs/models.example.json`).
An 8B model needs about 6 GB of free memory.

## Option B: Hugging Face router (hosted, no download)

1. Make a free account at huggingface.co, then create a token at Settings > Access Tokens with the permission
   "Make calls to Inference Providers".
2. In `.env` set `HF_TOKEN=hf_...` and, if you want another model, `HF_MODEL=<model id>`.
   Check on the model's Hugging Face page that an Inference Provider serves it, or the call returns 404.
3. Restart `bash run.sh`. When the free credit is used up the router returns HTTP 402 and the page shows it.

## Add any other model

Create `models.json` in the project folder (a list, see `docs/models.example.json`). Fields: `id`, `label`,
`base_url` (ends in /v1), `model`, and either `api_key_env` (the name of a `.env` variable) or nothing for local models.

## Test it properly for the mentor

Run the same questions on both models and compare. Use v1 so the rows match the saved Qwen3 results:

    RTO_EVAL_SET=v1 rto-agent-eval --split dev  --model-id llama31-8b-local --fresh-cache
    RTO_EVAL_SET=v1 rto-agent-eval --split test --model-id llama31-8b-local --fresh-cache

Each run is saved under `results/agent/` with the model name. Open `notebooks/04_Agentic_Copilot.ipynb`, section 9,
for the side-by-side table with 95% intervals. The test split runs once only, after you have chosen settings on dev.
With about 34 test rows the margin is about 17 points, so a smaller gap between models is not a real difference.
A local run takes a while (several model calls per question), so try `--limit 10` first.

## If something fails

| Message | Meaning |
|---|---|
| Ollama is not running | open the Ollama app, then run the script again |
| model 'llama3.1-16k' not pulled yet | run `bash setup_ollama.sh` |
| HF_TOKEN is not set | add the token to `.env`, restart |
| HTTP 402 | free Hugging Face credit used up, use Option A |
| HTTP 404 | that model id is not served by an Inference Provider, check the model page |
| Answers are malformed or very short | smaller models follow the JSON format less reliably, which is a finding worth reporting |
