"""Evaluate the agent on the frozen Week 1 eval set with the Week 1 scorer, plus agent metrics and a failure taxonomy.

    rto-agent-eval --limit 3            # smoke test (live endpoint, LLM_* env vars)
    rto-agent-eval                                # dev split (exemplars excluded)
    rto-agent-eval --no-llm-critic      # ablation: rule-check critic only
    rto-agent-eval --fault-rate 0.3     # robustness: 30% of tool calls fail

Outputs in results/agent_<stamp>/: answers.jsonl, taxonomy.csv, summary.json, traces.jsonl (OpenTelemetry spans),
sessions/ (persisted state per task), config.json.

Failure taxonomy (one primary cause per failed row, checked in this order):
    TOOL_CALL_ERROR   a tool failed or returned an error, a required tool was never run, or the planner emitted an
                      invalid tool call
    RETRIEVAL_MISS    the row needs a corpus source and no gold source was in the answer's context
    HALLUCINATION     the first answer cited a source that was not in its context, or the final answer states a
                      date that is not in the question, the case record or the tool results
    REASONING_ERROR   everything else: wrong behaviour, missing facts, wrong wording with the right context
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from . import config, llm, tracing
from . import tools as T
from .agent import Agent, Limits, PROJECT_SOURCES, Session, rb

RESULTS = config.RESULTS / "agent"
ISO = re.compile(r"\b20\d\d-\d\d-\d\d\b")
DEADLINE_RULES = {"R-SELLER-REPORT-SAME-STATE", "R-BUYER-REPORT-ACT", "R-BUYER-APPLY-PARIVAHAN"}


def taxonomy(row: dict, final: dict, tr, scored: dict) -> tuple[str, list[str]]:
    if scored["end_to_end"]:
        return "PASS", []
    flags = []
    tool_errors = [c for c in tr.tool_calls if "error" in c and not c.get("skipped")]
    ran = {c["tool"] for c in tr.tool_calls if "result" in c}
    case_needed = bool(row["case_id"]) or bool(T.CASE_ID.search(row["query"]))
    if tool_errors or any("invalid tool step" in e["error"] for e in tr.errors):
        flags.append("TOOL_CALL_ERROR")
    elif case_needed and "check_case" not in ran and "lookup_case" not in ran:
        flags.append("TOOL_CALL_ERROR")
    elif set(rb.split_list(row["gold_rule_ids"])) & DEADLINE_RULES and (row["reference_date"] or case_needed) \
            and "compute_deadlines" not in ran and row["expected_behavior"] in ("ANSWER", "ANSWER_WITH_CAVEAT"):
        flags.append("TOOL_CALL_ERROR")
    gold_corpus = set(rb.split_list(row["gold_source_ids"])) - PROJECT_SOURCES
    if gold_corpus and row["expected_behavior"] in ("ANSWER", "ANSWER_WITH_CAVEAT", "REFUSE_OUT_OF_SCOPE") \
            and not gold_corpus & set(tr.context_sources):
        flags.append("RETRIEVAL_MISS")
    first_issues = tr.critic_rounds[0]["issues"] if tr.critic_rounds else []
    allowed_dates = set(ISO.findall(row["query"] + " " + (row["reference_date"] or "")))
    allowed_dates |= set(ISO.findall(json.dumps([c.get("result") for c in tr.tool_calls], default=str)))
    stated = set(ISO.findall(str(final.get("answer", "")) + json.dumps(final.get("drafts") or [])))
    if any(i.startswith("[CITATION] cite only") for i in first_issues) or (stated - allowed_dates):
        flags.append("HALLUCINATION")
    flags.append("REASONING_ERROR")
    return flags[0], flags


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--cohort", choices=["v1", "v2"], help="only rows added in that eval version")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", nargs="*", help="query_ids to run")
    ap.add_argument("--no-llm-critic", action="store_true")
    ap.add_argument("--fault-rate", type=float, default=0.0)
    ap.add_argument("--usd-per-1m-tokens", type=float, default=0.0,
                    help="price used for the cost column; the course endpoint is free, so 0 by default")
    ap.add_argument("--model-id", help="model profile from models.py / models.json (default: the course endpoint in .env)")
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--fresh-cache", action="store_true",
                    help="start with an empty response cache so every LLM call is live and wall latency is real")
    args = ap.parse_args()

    config.load_dotenv()
    if args.split == "test":
        rb.check_test_split_allowed()
    if args.model_id:
        from . import models
        prof = models.get_profile(args.model_id)
        ok, why = prof.ready()
        if not ok:
            sys.exit(f"Model '{prof.id}' is not ready: {why}")
        client = prof.client(args.sleep)
    else:
        missing = config.missing_llm_env()
        if missing:
            sys.exit(f"Set {', '.join(missing)}.")
        client = rb.EndpointClient(os.environ["LLM_BASE_URL"], os.environ["LLM_API_KEY"], os.environ["LLM_MODEL"],
                                   False, args.sleep, os.environ.get("LLM_CA_CERT"), False)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = (f"_{args.model_id}" if args.model_id else "") + ("_rulecritic" if args.no_llm_critic else "") + (f"_faults{args.fault_rate:g}" if args.fault_rate else "")
    out = RESULTS / f"agent_{stamp}{tag}"
    (out / "sessions").mkdir(parents=True)
    tracing.setup(out / "traces.jsonl")
    if args.fresh_cache:
        fresh = config.LLM_CACHE / f"fresh_{stamp}"
        fresh.mkdir(parents=True)
        llm.set_cache_dir(fresh)

    search = T.CorpusSearch()
    limits = Limits()
    agent = Agent(client, search, limits, llm_critic=not args.no_llm_critic, fault_rate=args.fault_rate)
    rows = rb.select_rows(rb.read_csv(config.EVAL), args.split, args.cohort)
    if args.only:
        rows = [r for r in rows if r["query_id"] in set(args.only)]
    if args.limit:
        rows = rows[:args.limit]
    known = agent.known_sources | {c["source_id"] for c in search.ret.chunks}
    (out / "config.json").write_text(json.dumps({
        "split": args.split, "cohort": args.cohort, "eval_set": config.EVAL.name, "items": len(rows), "model": client.model, "llm_critic": not args.no_llm_critic,
        "fault_rate": args.fault_rate, "fresh_cache": args.fresh_cache, "limits": limits.__dict__, "retriever": "w128 hybrid_rerank_rrf, case-aware, "
        "8 chunks, <=2 per source", "usd_per_1m_tokens": args.usd_per_1m_tokens}, indent=2), encoding="utf-8")

    results = []
    with (out / "answers.jsonl").open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows, 1):
            session = Session(session_id=f"eval-{row['query_id']}", active_case_id=row["case_id"] or None)
            t0 = time.time()
            final, tr = agent.run(row["query"], row["reference_date"] or None, session, row)
            wall = round(time.time() - t0, 2)
            session.save(out / "sessions")
            scored = rb.score(row, final, known)
            primary, flags = taxonomy(row, final, tr, scored)
            rec = {"query_id": row["query_id"], "category": row["category"], "expected_behavior": row["expected_behavior"],
                   **scored, "failure_type": primary, "failure_flags": ";".join(flags), "completed": tr.completed,
                   "stop_reason": tr.stop_reason, "tool_calls": len(tr.tool_calls),
                   "tool_errors": sum("error" in c and not c.get("skipped") for c in tr.tool_calls),
                   "llm_calls": tr.llm_calls, "critic_rounds": len(tr.critic_rounds), "revisions": max(len(tr.critic_rounds) - 1, 0),
                   "repaired": ";".join(tr.repaired), "prompt_tokens": tr.prompt_tokens,
                   "completion_tokens": tr.completion_tokens, "tokens": tr.tokens, "llm_latency_s": tr.latency_s,
                   "wall_latency_s": wall, "cost_usd": round(tr.tokens * args.usd_per_1m_tokens / 1e6, 6),
                   "plan": tr.plan, "tools": [{k: c[k] for k in ("tool", "args", "error") if k in c} for c in tr.tool_calls],
                   "critic": tr.critic_rounds, "errors": tr.errors, "context_sources": tr.context_sources, "final": final}
            results.append(rec)
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            f.flush()
            print(f"{i}/{len(rows)} {row['query_id']:<8} {'PASS' if scored['end_to_end'] else primary:<16} "
                  f"behavior {scored['predicted_behavior']:<22} tools {rec['tool_calls']} llm {tr.llm_calls} "
                  f"tokens {tr.tokens} {wall}s", flush=True)

    n = len(results)
    mean = lambda k: round(sum(float(r[k]) for r in results) / n, 3)
    walls = sorted(r["wall_latency_s"] for r in results)
    summary = {
        "items": n, "end_to_end_accuracy": mean("end_to_end"), "behavior_accuracy": mean("behavior_ok"),
        "must_include_coverage": mean("include_coverage"), "finding_code_exact_match": mean("findings_ok"),
        "citation_ok_rate": mean("citations_ok"), "task_completion_rate": mean("completed"),
        "avg_tool_calls": mean("tool_calls"), "avg_llm_calls": mean("llm_calls"), "avg_revisions": mean("revisions"),
        "tool_error_rate": round(sum(r["tool_errors"] for r in results) / max(sum(r["tool_calls"] for r in results), 1), 3),
        "avg_tokens": mean("tokens"), "avg_prompt_tokens": mean("prompt_tokens"),
        "avg_completion_tokens": mean("completion_tokens"), "avg_llm_latency_s": mean("llm_latency_s"),
        "avg_wall_latency_s": mean("wall_latency_s"), "p50_wall_latency_s": statistics.median(walls),
        "p95_wall_latency_s": walls[min(n - 1, int(round(0.95 * (n - 1))))], "avg_cost_usd": mean("cost_usd"),
        "stop_reasons": dict(Counter(r["stop_reason"] for r in results)),
        "failure_taxonomy": dict(Counter(r["failure_type"] for r in results)),
        "failure_flags": dict(Counter(f for r in results for f in r["failure_flags"].split(";") if f)),
        "failure_tags": dict(Counter(t for r in results for t in rb.split_list(r["failure_tags"]))),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (out / "taxonomy.csv").open("w", newline="", encoding="utf-8") as f:
        cols = ["query_id", "category", "expected_behavior", "predicted_behavior", "end_to_end", "failure_type",
                "failure_flags", "failure_tags", "completed", "stop_reason", "tool_calls", "tool_errors", "llm_calls",
                "revisions", "repaired", "tokens", "wall_latency_s"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(json.dumps(summary, indent=2))
    print(f"Results: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
