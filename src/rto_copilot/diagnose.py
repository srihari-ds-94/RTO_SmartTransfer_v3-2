"""Retrieval-vs-generation diagnostic for the latest live RAG run, with the Week 1 S1_FEW_SHOT answers alongside.

Every dev row that fails end to end under RAG_RETRIEVED gets one cause:
    RETRIEVAL_MISS            no gold source in the context, and the same row passes when the gold sources are given
    RETRIEVAL_AND_GENERATION  no gold source in the context, and it still fails with the gold sources given
    GENERATION_MISS           a gold source was in the context, the generator still failed
    GENERATION_NO_RETRIEVAL   the row needs no corpus source (refusal, clarification, case-only); behaviour/format failed

    rto-diagnose                       # latest live run: writes results/<run>/diagnostic.csv and prints the summary
    rto-diagnose <run folder>          # a specific run
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

from . import baseline as rb
from . import config
from .rag import RESULTS
from .retrieval import EQUIVALENT


def load(path: Path) -> dict[str, dict]:
    return {r["query_id"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())}


def week1_s1() -> dict[str, dict]:
    run = sorted(p for p in (config.RESULTS / "baseline").glob("*") if p.is_dir() and "mock" not in p.name)[-1]
    rows = {r["query_id"]: r for r in rb.read_csv(rb.EVAL)}
    known = {r["source_id"] for r in rb.read_csv(rb.SOURCES)}
    out = {}
    for old in map(json.loads, (run / "S1_FEW_SHOT.jsonl").read_text(encoding="utf-8").splitlines()):
        out[old["query_id"]] = rb.score(rows[old["query_id"]], rb.parse_json(old["raw"]), known)
    return out


def lenient_pass(r: dict, row: dict) -> bool:
    """End to end, but a legally equivalent corpus source (EQUIVALENT) also satisfies the gold-citation check."""
    other_checks = (r["behavior_ok"] and r["include_all"] and not r["must_not_violation"] and r["findings_ok"]
                    and r["drafts_ok"])
    if not other_checks:
        return False
    if r["citations_ok"]:
        return True
    gold = set(rb.split_list(row["gold_source_ids"]))
    accepted = gold.union(*(EQUIVALENT.get(s, set()) for s in gold))
    return row["expected_behavior"] in rb.CITING_BEHAVIORS and bool(set(r["cited_sources"]) & accepted)


def cause(r: dict, g: dict | None) -> str:
    if r["end_to_end"]:
        return "PASS"
    if r["retrieval_hit"] is None:
        return "GENERATION_NO_RETRIEVAL"
    if r["retrieval_hit"]:
        return "GENERATION_MISS"
    return "RETRIEVAL_MISS" if g and g["end_to_end"] else "RETRIEVAL_AND_GENERATION"


def main() -> int:
    if len(sys.argv) > 1:
        run = RESULTS / sys.argv[1]
    else:
        run = sorted(p for p in RESULTS.glob("*_rag_*") if p.is_dir() and "mock" not in p.name
                     and (p / "RAG_GOLD.jsonl").exists())[-1]
    ret, gold, s1 = load(run / "RAG_RETRIEVED.jsonl"), load(run / "RAG_GOLD.jsonl"), week1_s1()
    eval_rows = {r["query_id"]: r for r in rb.read_csv(rb.EVAL)}
    rows = []
    for qid, r in ret.items():
        g = gold.get(qid)
        rows.append({"query_id": qid, "category": r["category"], "expected_behavior": r["expected_behavior"],
                     "week1_s1_pass": s1[qid]["end_to_end"] if qid in s1 else "",
                     "rag_retrieved_pass": r["end_to_end"], "rag_gold_pass": g["end_to_end"] if g else "",
                     "rag_retrieved_pass_lenient_citation": lenient_pass(r, eval_rows[qid]),
                     "retrieval_hit_top5": "" if r["retrieval_hit"] is None else r["retrieval_hit"],
                     "cause": cause(r, g), "failure_tags": r["failure_tags"],
                     "gold_sources": ";".join(r["gold_sources"]), "context_sources": ";".join(r["context_sources"]),
                     "cited_sources": ";".join(r["cited_sources"])})
    with (run / "diagnostic.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    rate = lambda key: sum(bool(x[key]) for x in rows if x[key] != "") / n
    print(f"Run {run.name}: {n} dev rows")
    print(f"  end-to-end  Week 1 S1 {rate('week1_s1_pass'):.3f} | RAG retrieved {rate('rag_retrieved_pass'):.3f} "
          f"| RAG gold sources {rate('rag_gold_pass'):.3f} "
          f"| RAG retrieved, lenient citation {rate('rag_retrieved_pass_lenient_citation'):.3f}")
    print("  cause of RAG_RETRIEVED failures:")
    for k, v in Counter(x["cause"] for x in rows).most_common():
        print(f"    {k:<26}{v}")
    print("  failure tags among GENERATION_MISS rows:")
    for k, v in Counter(t for x in rows if x["cause"] == "GENERATION_MISS"
                        for t in rb.split_list(x["failure_tags"])).most_common():
        print(f"    {k:<26}{v}")
    print("  end-to-end by category (S1 / retrieved / gold):")
    for cat in sorted({x["category"] for x in rows}):
        sub = [x for x in rows if x["category"] == cat]
        cnt = lambda key: sum(bool(x[key]) for x in sub if x[key] != "")
        print(f"    {cat:<12} n={len(sub):<3} {cnt('week1_s1_pass')} / {cnt('rag_retrieved_pass')} / {cnt('rag_gold_pass')}")
    print(f"Saved {run / 'diagnostic.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
