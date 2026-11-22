"""Make every answer checkable: resolve each citation to its source, show the passage, and measure word overlap.

``source_overlap`` is the share of the answer's content words that also appear in the cited passages (ROUGE-1
precision against the evidence). It is a hint, not proof: answers also quote case facts that are not in the law text,
and a paraphrase can score low. A low score or a missing citation means "read the source before relying on this".
"""
from __future__ import annotations

import csv
from functools import lru_cache

from . import config
from .textmetrics import tokens

CITING = {"ANSWER", "ANSWER_WITH_CAVEAT"}
_STOP = set("the a an and or of to in on for is are be by with as at it this that from if not no any must may can "
            "will shall should you your we our they their there then than has have had was were been which who "
            "what when where how also only into per under over after before within".split())


@lru_cache(maxsize=1)
def registry() -> dict[str, dict]:
    with config.SOURCES.open(encoding="utf-8-sig", newline="") as f:
        return {r["source_id"]: r for r in csv.DictReader(f)}


def content_words(text: str) -> set[str]:
    return {t for t in tokens(text) if t not in _STOP and len(t) > 2}


def resolve(final: dict, query: str, search=None) -> dict:
    """Return {'citations': [...], 'source_overlap': float|None, 'level': 'high'|'medium'|'low'|'none'|'n/a'}."""
    reg = registry()
    behavior = final.get("behavior", "")
    out, passages = [], []
    for c in final.get("citations") or []:
        if not isinstance(c, dict):
            continue
        sid = str(c.get("source_id", ""))
        meta = reg.get(sid)
        snippet = ""
        if search is not None and meta:
            try:
                snippet = search.get_source(sid, query)["results"][0]["text"]
            except Exception:
                snippet = ""
        if snippet:
            passages.append(snippet)
        out.append({"source_id": sid, "locator": c.get("locator", ""), "in_registry": bool(meta),
                    "title": meta["title"] if meta else "Not in the source registry",
                    "url": meta["source_url"] if meta else "", "status": meta["status"] if meta else "",
                    "snippet": snippet[:700]})
    if behavior not in CITING:
        return {"citations": out, "source_overlap": None, "level": "n/a"}
    if not out:
        return {"citations": out, "source_overlap": None, "level": "none"}
    ans = content_words(str(final.get("answer", "")))
    ev = set().union(*(content_words(p) for p in passages)) if passages else set()
    overlap = round(len(ans & ev) / len(ans), 3) if ans and ev else None
    level = "low" if overlap is None or overlap < 0.35 else ("medium" if overlap < 0.55 else "high")
    if any(not c["in_registry"] for c in out):
        level = "low"
    return {"citations": out, "source_overlap": overlap, "level": level}
