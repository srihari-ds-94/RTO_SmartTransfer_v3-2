"""rto-check: confirm the setup works before starting the UI. Tests every configured model with one tiny call.

    rto-check                      all models
    rto-check --model-id hf-router one model
"""
from __future__ import annotations

import argparse
import json
import ssl
import sys
import urllib.error
import urllib.request

from . import config, models


def _chat(p: models.Profile) -> str:
    ctx = ssl.create_default_context(cafile=p.ca_cert) if p.ca_cert else None
    body = json.dumps({"model": p.model, "max_tokens": 16, "messages": [{"role": "user", "content": "Say OK."}]}).encode()
    req = urllib.request.Request(p.base_url.rstrip("/") + "/chat/completions", data=body, headers={
        "Authorization": f"Bearer {p.api_key or 'ollama'}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180, context=ctx) as r:
        return json.load(r)["choices"][0]["message"]["content"].strip()[:60]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id")
    args = ap.parse_args()
    config.load_dotenv()
    results = []

    def line(good: bool | None, text: str) -> None:
        print(("PASS  " if good else "FAIL  " if good is False else "SKIP  ") + text)

    try:
        import sentence_transformers  # noqa: F401
        line(True, "sentence-transformers installed")
    except ImportError:
        line(False, "sentence-transformers missing: run  .venv/bin/python -m pip install -e \".[notebooks,dev]\"")
        results.append(False)

    profiles = [models.get_profile(args.model_id)] if args.model_id else models.load_profiles()
    for p in profiles:
        ok, why = p.ready()
        if not ok:
            line(None if not args.model_id else False, f"{p.id}: not ready ({why})")
            if args.model_id:
                results.append(False)
            continue
        try:
            line(True, f"{p.id}: {p.model} replied {_chat(p)!r}")
            results.append(True)
        except urllib.error.HTTPError as e:
            hint = {401: "key wrong or not for this URL", 403: "token lacks permission", 404: "model not found at this URL",
                    402: "free credit used up", 429: "busy or rate limited, wait and retry",
                    503: "server busy, retry"}.get(e.code, "")
            line(False, f"{p.id}: HTTP {e.code} {hint}")
            results.append(False)
        except Exception as e:
            line(False, f"{p.id}: {type(e).__name__}: {e}")
            results.append(False)
    if not any(r for r in results):
        sys.exit(1)
    sys.exit(0 if all(r for r in results) else 1)


if __name__ == "__main__":
    main()
