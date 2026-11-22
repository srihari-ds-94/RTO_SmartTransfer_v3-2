# RTO SmartTransfer Copilot v3

An agentic RAG copilot for used-vehicle ownership transfer inside one West Bengal RTO (Track B capstone, group Synapse6).
It checks documents, explains rules with citations, calculates deadlines and prepares drafts. It never sends, files,
pays, signs or approves anything. Demo data only; not legal advice.

v3 is a cleaned-up, installable version of the Week 1 to 4 work, with an extended evaluation set (v2, 196 rows), demo
notebooks and a minimal web UI. Read `docs/REVIEW_FINDINGS.md` for what changed and why.

## Quick start (Mac or Linux)

```bash
bash setup.sh        # once: creates .venv and installs everything
# open .env and paste your team key after LLM_API_KEY=   (URL, model and certificate are already set)
bash run.sh          # checks the model connection, then opens http://127.0.0.1:8000
```

`rto-check` (or `.venv/bin/python -m rto_copilot.check`) tests the key, certificate and URL on their own.
Notebooks: open `notebooks/00_Overview.ipynb` in VS Code and pick the `.venv` kernel. `RUN_LIVE = True` is already set
and switches itself off while the key is missing. Tests: `.venv/bin/python -m pytest`.

Without a key the page still shows case checks and deadlines. Asking the copilot needs the model.
The first question downloads the retrieval models (about 1.3 GB, once), so it is slow the first time.

Keep `.env` and `certs/` inside the team. The team key allows one active request at a time, so do not run the UI and an
evaluation together.

## Layout

| Path | What is in it |
|---|---|
| `src/rto_copilot/` | The package: `config`, `llm`, `scoring`, `baseline`, `retrieval`, `rag`, `tools`, `agent`, `guardrails`, `serving`, `tracing`, `api`, `web/index.html` |
| `data/cases/` | `synthetic_cases.csv` (148 cases), `synthetic_cases_100.csv` (the original 100) |
| `data/eval/` | `eval_set_v1.csv` (frozen, SHA-256), `eval_set_v2.csv` (v1 + 98 new rows, draft until reviewed) |
| `data/registries/`, `data/spec/` | Rule and source registries, specification, knowledge pack |
| `data/corpus/`, `data/index/` | Parsed official sources and the prebuilt retrieval index |
| `notebooks/` | 00 Overview, 01 Eval set, 02 Baseline, 03 RAG, 04 Agent, 05 Evaluation and RAI, 06 UI |
| `results/` | Saved Week 1 to 4 runs the notebooks read |
| `scripts/` | `build_eval_v2.py`, `build_notebooks.py`, `build_corpus.py`, `fetch_sources.py` |
| `docs/` | `REVIEW_FINDINGS.md`, `EVAL_V2_CHANGELOG.md`, UI screenshot |

## Command line

| Command | Purpose |
|---|---|
| `rto-evalset --version v2` | Validate an eval set (no leakage, no near-duplicates, v1 untouched). `--freeze` after team review |
| `rto-baseline`, `rto-rag`, `rto-agent-eval` | Week 1, 2 and 3 runs; `--split dev`, `--cohort v1\|v2` |
| `rto-report`, `rto-rai`, `rto-diagnose` | Week 4 report, responsible-AI checks, failure diagnosis |
| `rto-serve` | The web app |

Choose the eval set with `RTO_EVAL_SET=v1|v2` (default v2). The test split only runs on a frozen set.

## Evaluation discipline

1. v1 never changes. v2 contains every v1 row byte for byte plus new rows tagged `cohort=v2`.
2. Dev rows use train and val cases; test rows use test cases only.
3. Tune on dev. Run the v2 test split once, after the freeze, and report it with a 95% interval.
4. End-to-end accuracy is the AND of six checks. Always read it next to the component rates (`component_report`).

## Docker

```bash
cp .env.example .env && docker compose up --build      # http://localhost:8000
```

## Status

Offline pieces are tested: scorer, tools, agent control flow (scripted model), serving layer, API, eval-set validator,
notebooks. Live model runs on v2 are still to do; see the last section of `docs/REVIEW_FINDINGS.md`.
