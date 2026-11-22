"""Summarise an OpenTelemetry trace file written by tracing.py.

    python trace_report.py results/agent_<stamp>              # per-span table -> trace_summary.csv
    python trace_report.py results/agent_<stamp> --tree DDL-04 # span tree of one task
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def load(run: Path) -> list[dict]:
    return [json.loads(line) for line in (run / "traces.jsonl").read_text(encoding="utf-8").splitlines()]


def summary(spans: list[dict]) -> list[dict]:
    by = defaultdict(list)
    for s in spans:
        by[s["name"]].append(s)
    rows = []
    for name, ss in sorted(by.items()):
        d = sorted(x["duration_ms"] for x in ss)
        rows.append({"span": name, "count": len(ss), "errors": sum(x["status"] == "ERROR" for x in ss),
                     "avg_ms": round(statistics.mean(d), 1), "p95_ms": d[min(len(d) - 1, int(round(0.95 * (len(d) - 1))))],
                     "total_s": round(sum(d) / 1000, 1)})
    return rows


def tree(spans: list[dict], query_id: str) -> None:
    root = next((s for s in spans if s["name"] == "agent.turn" and s["attributes"].get("query_id") == query_id), None)
    if not root:
        print(f"no trace for {query_id}")
        return
    kids = defaultdict(list)
    for s in spans:
        if s["trace_id"] == root["trace_id"] and s["parent_id"]:
            kids[s["parent_id"]].append(s)

    def show(s, depth):
        a = s["attributes"]
        extra = {k: a[k] for k in ("behavior", "prompt_tokens", "completion_tokens", "error", "issues", "stop_reason",
                                   "tool_calls", "tokens") if k in a and a[k] not in ("", "[]")}
        print(f"{'  ' * depth}{s['name']:<24} {s['duration_ms']:>9.1f} ms  {s['status']:<5} {json.dumps(extra)[:160]}")
        for k in sorted(kids[s["span_id"]], key=lambda x: x["start_ns"]):
            show(k, depth + 1)
    show(root, 0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--tree")
    args = ap.parse_args()
    spans = load(args.run)
    if args.tree:
        tree(spans, args.tree)
        return 0
    rows = summary(spans)
    with (args.run / "trace_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{'span':<24}{'count':>7}{'errors':>8}{'avg ms':>10}{'p95 ms':>10}{'total s':>9}")
    for r in rows:
        print(f"{r['span']:<24}{r['count']:>7}{r['errors']:>8}{r['avg_ms']:>10}{r['p95_ms']:>10}{r['total_s']:>9}")
    print(f"{len(spans)} spans, {sum(1 for s in spans if s['name'] == 'agent.turn')} tasks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
