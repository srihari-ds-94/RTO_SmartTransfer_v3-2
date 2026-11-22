"""rto-check: confirm .env, the certificate and the model endpoint work, before starting the UI."""
from __future__ import annotations

import json
import os
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path

from . import config


def main() -> None:
    config.load_dotenv()
    ok = True

    def line(good: bool, text: str) -> None:
        nonlocal ok
        ok = ok and good
        print(("PASS  " if good else "FAIL  ") + text)

    miss = config.missing_llm_env()
    line(not miss, "LLM_BASE_URL, LLM_API_KEY, LLM_MODEL set" + (f" (missing: {', '.join(miss)}; edit .env)" if miss else ""))
    cert = os.environ.get("LLM_CA_CERT", "")
    line(bool(cert) and Path(cert).exists(), f"certificate file found: {cert or 'LLM_CA_CERT not set'}")
    try:
        import sentence_transformers  # noqa: F401
        line(True, "sentence-transformers installed")
    except ImportError:
        line(False, "sentence-transformers missing: run  .venv/bin/python -m pip install -e \".[notebooks,dev]\"")
    if miss or not cert or not Path(cert).exists():
        sys.exit(1)
    ctx = ssl.create_default_context(cafile=cert)
    base = os.environ["LLM_BASE_URL"].rstrip("/")
    key = os.environ["LLM_API_KEY"]
    try:
        req = urllib.request.Request(base + "/models", headers={"Authorization": f"Bearer {key}"})
        with urllib.request.urlopen(req, timeout=30, context=ctx) as r:
            ids = [m.get("id") for m in json.load(r).get("data", [])]
        line(True, f"endpoint reachable, models: {ids}")
        body = json.dumps({"model": os.environ["LLM_MODEL"], "max_tokens": 16,
                           "messages": [{"role": "user", "content": "Say OK."}]}).encode()
        req = urllib.request.Request(base + "/chat/completions", data=body, headers={
            "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120, context=ctx) as r:
            text = json.load(r)["choices"][0]["message"]["content"]
        line(True, f"chat call worked, reply: {text.strip()[:60]!r}")
    except urllib.error.HTTPError as e:
        hint = {401: "key wrong or not for this URL", 429: "someone else is using the team key, wait and retry",
                503: "server busy, retry"}.get(e.code, "")
        line(False, f"HTTP {e.code} {hint}")
    except Exception as e:
        line(False, f"{type(e).__name__}: {e} (wrong URL, no network or VPN needed, or wrong certificate)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
