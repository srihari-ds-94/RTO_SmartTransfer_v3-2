"""Citation-grounded RAG answers, scored with the Week 1 scorer so the numbers compare with the Week 1 baseline.

The prompt is the Week 1 S1_FEW_SHOT prompt with one change: the hand-written knowledge pack is gone. Each question
gets the top retrieved official sources from corpus/ instead, and the copilot may cite only those (plus the two
project sources, scope policy and case dataset, which are not legal texts and so are not in the corpus).

Two retrieval conditions separate retrieval failures from generation failures:
    retrieved   top-K sources from the retriever (the real system)
    gold        the gold sources are placed in the context first, then retrieved ones fill the rest
                (an upper bound: what the generator does when retrieval is perfect)

    rto-rag --mock oracle                       # pipeline self-test, must reach 100%
    rto-rag                                           # dev split, both conditions, live endpoint
    rto-rag --conditions retrieved --limit 5
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

from . import baseline as rb
from . import config
from .retrieval import (DEFAULT_CONFIG, DEFAULT_METHOD, METHODS, NOT_RETRIEVABLE, Retriever, case_aware_query,
                        load_cases, load_docs)

RESULTS = config.RESULTS / "rag"
TOP_K = 5             # chunks given with the question being answered (see --chunks / --per-source)
PER_SOURCE = 1        # chunks allowed from one source
EXEMPLAR_K = 2        # sources given with each few-shot example (kept short for the 8192-token limit)
EXEMPLAR_WORDS = 60
CONDITIONS = ["retrieved", "gold"]

PROJECT_SOURCES = """[SRC-SPEC-SCOPE] Copilot scope and refusal policy (project policy, not law).
[SRC-PACK-DATASET] Synthetic case records with demo finding codes (not legal rules)."""


def chunk_block(chunk: dict, words: int | None = None) -> str:
    body = chunk["text"][len(chunk["title"]) + 2:]
    if words:
        body = " ".join(body.split()[:words]) + " ..."
    return f"[{chunk['source_id']}] {chunk['title']} ({chunk['locator']})\n{body}"


class RagContext(rb.Context):
    """Week 1 context, with the knowledge pack replaced by retrieved chunks."""

    def __init__(self, retriever: Retriever, method: str, case_aware: bool):
        super().__init__()
        self.ret = retriever
        self.method = method
        self.case_aware = case_aware
        self.case_rows = load_cases()
        rule_lines = [
            f"{r['rule_id']} | window_days={r['window_days'] or '-'} | conflict_group={r['conflict_group'] or '-'} "
            f"| status={r['rule_status']} | {r['statement']}"
            for r in self.rules
        ]
        self.static = f"PROJECT SOURCES (always available):\n{PROJECT_SOURCES}\n\nRULES:\n" + "\n".join(rule_lines)
        self.first_chunk = {}
        for i, c in enumerate(retriever.chunks):
            self.first_chunk.setdefault(c["source_id"], i)

    def search_text(self, row: dict) -> str:
        if not self.case_aware:
            return row["query"]
        return case_aware_query(row["query"], self.case_rows.get(row["case_id"]))

    def context_chunks(self, row: dict, k: int, condition: str, per_source: int = 1) -> list[dict]:
        """The best-ranked chunks, at most per_source from one source, k in total (gold chunks first if asked)."""
        ranking = self.ret.search(self.search_text(row), self.method)
        picked: list[int] = []
        count: dict[str, int] = {}

        def take(i: int) -> None:
            s = self.ret.chunks[i]["source_id"]
            if i not in picked and count.get(s, 0) < per_source:
                picked.append(i)
                count[s] = count.get(s, 0) + 1

        if condition == "gold":
            for s in rb.split_list(row["gold_source_ids"]):
                if s not in NOT_RETRIEVABLE and s in self.first_chunk:
                    own = [i for i in ranking if self.ret.chunks[i]["source_id"] == s] or [self.first_chunk[s]]
                    for i in own[:per_source]:
                        take(i)
        for i in ranking:
            if len(picked) >= k:
                break
            take(i)
        return [self.ret.chunks[i] for i in picked]

    def rag_user_message(self, row: dict, k: int, condition: str, words: int | None = None,
                         per_source: int = 1) -> tuple[str, list[str]]:
        chunks = self.context_chunks(row, k, condition, per_source)
        sources = "\n\n".join(chunk_block(c, words) for c in chunks)
        return f"SOURCES (retrieved from the official corpus):\n{sources}\n\n{self.user_message(row)}", \
            list(dict.fromkeys(c["source_id"] for c in chunks))


def exemplar_messages(ctx: RagContext, rows_by_id: dict) -> list[dict]:
    msgs = []
    for qid in rb.EXEMPLAR_IDS:
        row = rows_by_id[qid]
        content, _ = ctx.rag_user_message(row, EXEMPLAR_K, "gold", EXEMPLAR_WORDS)
        msgs.append({"role": "user", "content": content})
        msgs.append({"role": "assistant", "content": json.dumps(rb.gold_response(row))})
    return msgs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS, choices=CONDITIONS)
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--cohort", choices=["v1", "v2"], help="only rows added in that eval version")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--method", default=DEFAULT_METHOD, choices=METHODS)
    ap.add_argument("--plain-query", action="store_true", help="do not add case-record facts to the search query")
    ap.add_argument("--chunks", type=int, default=TOP_K, help="chunks in the context of the question")
    ap.add_argument("--per-source", type=int, default=PER_SOURCE, help="max chunks from one source")
    ap.add_argument("--mock", choices=["oracle", "naive"])
    ap.add_argument("--sleep", type=float, default=1.0)
    args = ap.parse_args()

    if args.split == "test":
        rb.check_test_split_allowed()
    if args.mock:
        client = rb.MockClient(args.mock)
    else:
        missing = [v for v in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL") if not os.environ.get(v)]
        if missing:
            return sys.exit(f"Set {', '.join(missing)} (or use --mock oracle).")
        client = rb.EndpointClient(os.environ["LLM_BASE_URL"], os.environ["LLM_API_KEY"], os.environ["LLM_MODEL"],
                                   False, args.sleep, os.environ.get("LLM_CA_CERT"), False)

    docs = load_docs()
    ret = Retriever(args.config, docs, use_reranker="rerank" in args.method)
    ctx = RagContext(ret, args.method, case_aware=not args.plain_query)
    all_rows = rb.read_csv(rb.EVAL)
    rows_by_id = {r["query_id"]: r for r in all_rows}
    rows = rb.select_rows(all_rows, args.split, args.cohort)
    if args.limit:
        rows = rows[:args.limit]
    known_sources = set(ctx.sources) | {d["source_id"] for d in docs}
    exemplars = exemplar_messages(ctx, rows_by_id)
    system = {"role": "system", "content": ctx.system(rb.SYSTEM_PROMPT)}

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_rag_" + re.sub(r"[^A-Za-z0-9.-]+", "-", client.model)
    out_dir = RESULTS / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps({
        "split": args.split, "config": args.config, "method": args.method, "case_aware_query": not args.plain_query,
        "chunks": args.chunks, "per_source": args.per_source, "exemplar_chunks": EXEMPLAR_K}, indent=2), encoding="utf-8")
    summaries = []
    for condition in args.conditions:
        name = f"RAG_{condition.upper()}"
        scored = []
        with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for i, row in enumerate(rows, 1):
                content, given = ctx.rag_user_message(row, args.chunks, condition, per_source=args.per_source)
                gold = set(rb.split_list(row["gold_source_ids"])) - NOT_RETRIEVABLE
                stats = {"calls": 0, "latency_s": 0.0, "tokens": 0, "malformed": 0}
                try:
                    obj, raw = rb.call_json(client, [system] + exemplars + [{"role": "user", "content": content}],
                                            0.0, row, stats)
                except Exception as e:
                    obj, raw = None, f"ERROR: {e}"
                cited = sorted({str(c.get("source_id")) for c in (obj or {}).get("citations") or [] if isinstance(c, dict)})
                result = {"query_id": row["query_id"], "category": row["category"], "agent_id": row["agent_id"],
                          "expected_behavior": row["expected_behavior"], "gold_sources": sorted(gold),
                          "context_sources": given, "retrieval_hit": bool(gold & set(given)) if gold else None,
                          "cited_sources": cited, "cited_outside_context": sorted(set(cited) - set(given)
                                                                                    - {"SRC-SPEC-SCOPE", "SRC-PACK-DATASET"}),
                          **rb.score(row, obj, known_sources), **stats, "raw": raw}
                scored.append(result)
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                if i % 10 == 0 or i == len(rows):
                    print(f"{name}: {i}/{len(rows)}", flush=True)
        summary = rb.summarise(name, scored)
        with_gold = [s for s in scored if s["retrieval_hit"] is not None]
        summary["retrieval_hit_rate"] = round(sum(s["retrieval_hit"] for s in with_gold) / len(with_gold), 3) if with_gold else 0
        summary["cited_outside_context_rate"] = round(sum(bool(s["cited_outside_context"]) for s in scored) / len(scored), 3)
        summaries.append(summary)

    head = list(dict.fromkeys(k for s in summaries for k in s))
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=head, restval=0)
        w.writeheader()
        w.writerows(summaries)
    print(f"\nModel: {client.model} | split: {args.split} | items: {len(rows)} | retriever: {args.config} {args.method} "
          f"{'plain' if args.plain_query else 'case-aware'} query, {args.chunks} chunks, <= {args.per_source} per source")
    for s in summaries:
        print(f"  {s['strategy']:<14} end-to-end {s['end_to_end_accuracy']:.3f}  behavior {s['behavior_accuracy']:.3f}  "
              f"include {s['must_include_coverage']:.3f}  findings {s['finding_code_exact_match']:.3f}  "
              f"citations {s['citation_ok_rate']:.3f}  retrieval hit {s['retrieval_hit_rate']:.3f}")
    print(f"Results: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
