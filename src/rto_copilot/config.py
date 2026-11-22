"""One place for every path and environment setting.

Nothing else in the package builds a path from ``__file__`` or edits ``sys.path``. To move the data or the results,
set the environment variable named next to each path below.

    RTO_HOME           project root (default: two levels above this file)
    RTO_EVAL_SET       eval set version to use, "v1" or "v2" (default v2)
    COPILOT_STATE_DIR  sessions, traces and the review log (default <root>/state)
    LLM_CACHE_DIR      cache of raw LLM responses (default <root>/.cache)
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("RTO_HOME") or Path(__file__).resolve().parents[2])
DATA = ROOT / "data"

CASES = DATA / "cases" / "synthetic_cases.csv"            # DEMO-001..100 (v1) plus DEMO-101..160 (v2 extension)
CASES_V1 = DATA / "cases" / "synthetic_cases_100.csv"     # the original file, kept unchanged for the audit trail
SOURCES = DATA / "registries" / "source_registry.csv"
RULES = DATA / "registries" / "rule_registry.csv"
KNOWLEDGE = DATA / "spec" / "knowledge_pack.txt"
SPEC = DATA / "spec" / "COPILOT_SPECIFICATION.md"
CORPUS_DOCS = DATA / "corpus" / "documents.jsonl"
CORPUS_MANIFEST = DATA / "corpus" / "sources_manifest.csv"
INDEX = DATA / "index"
PRICING = DATA / "pricing.json"

EVAL_VERSION = os.environ.get("RTO_EVAL_SET", "v2")


def eval_path(version: str | None = None) -> Path:
    return DATA / "eval" / f"eval_set_{version or EVAL_VERSION}.csv"


def freeze_path(version: str | None = None) -> Path:
    return DATA / "eval" / f"eval_set_{version or EVAL_VERSION}.sha256"


EVAL = eval_path()
FREEZE_RECORD = freeze_path()

RESULTS = ROOT / "results"
STATE = Path(os.environ.get("COPILOT_STATE_DIR") or ROOT / "state")
LLM_CACHE = Path(os.environ.get("LLM_CACHE_DIR") or ROOT / ".cache")
WEB = Path(__file__).resolve().parent / "web"

REQUIRED_LLM_ENV = ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL")


def missing_llm_env() -> list[str]:
    return [v for v in REQUIRED_LLM_ENV if not os.environ.get(v)]


def load_dotenv(path: Path | None = None) -> bool:
    """Read KEY=VALUE lines from .env into os.environ (never overrides a variable that is already set).

    Notebooks call this so a student can keep the key in .env instead of typing it into a cell."""
    path = path or ROOT / ".env"
    if not path.exists():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    cert = os.environ.get("LLM_CA_CERT", "")
    if cert and not Path(cert).is_absolute():
        os.environ["LLM_CA_CERT"] = str(ROOT / cert)   # relative paths are relative to the project, not the terminal
    return True
