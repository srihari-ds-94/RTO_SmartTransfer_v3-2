"""Break down one baseline run: accuracy by category, behaviour confusion, failure tags and example failures.

    python analyze_results.py                       # latest run in results/
    python analyze_results.py results/<run_folder>
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from . import config

RESULTS = config.RESULTS / "baseline"
STRATEGIES = ["S0_ZERO_SHOT", "S1_FEW_SHOT", "S2_SELF_CONSISTENCY", "S3_DECOMPOSITION"]


def load(run):
    return {s: [json.loads(l) for l in (run / f"{s}.jsonl").open(encoding="utf-8")]
            for s in STRATEGIES if (run / f"{s}.jsonl").exists()}


def pct(rows, key):
    return f"{100 * sum(bool(r[key]) for r in rows) / len(rows):.0f}%" if rows else "-"


def main():
    if len(sys.argv) > 1:
        run = Path(sys.argv[1])
    else:
        runs = [p for p in RESULTS.iterdir() if p.is_dir() and not p.name.split("_", 2)[-1].startswith("mock")]
        run = max(runs, key=lambda p: p.name)
    data = load(run)
    print(f"Run: {run.name}\n")

    print("End-to-end accuracy by category")
    cats = sorted({r["category"] for rows in data.values() for r in rows})
    print("  " + "category".ljust(10) + "".join(s.split("_")[0].ljust(8) for s in data) + "n")
    for c in cats:
        cells = [pct([r for r in rows if r["category"] == c], "end_to_end") for rows in data.values()]
        n = sum(r["category"] == c for r in next(iter(data.values())))
        print("  " + c.ljust(10) + "".join(x.ljust(8) for x in cells) + str(n))

    print("\nBehaviour accuracy by expected behaviour")
    behs = sorted({r["expected_behavior"] for rows in data.values() for r in rows})
    for b in behs:
        cells = [pct([r for r in rows if r["expected_behavior"] == b], "behavior_ok") for rows in data.values()]
        n = sum(r["expected_behavior"] == b for r in next(iter(data.values())))
        print("  " + b.ljust(24) + "".join(x.ljust(8) for x in cells) + str(n))

    for name, rows in data.items():
        print(f"\n=== {name} ===")
        print("  failure tags:", dict(Counter(t for r in rows for t in r["failure_tags"].split(";") if t).most_common()))
        confusion = Counter((r["expected_behavior"], r["predicted_behavior"]) for r in rows if not r["behavior_ok"])
        print("  behaviour confusions (expected -> predicted):")
        for (g, p), k in confusion.most_common(8):
            print(f"    {g} -> {p or '(none)'}: {k}")
        calls = sum(r["calls"] for r in rows) / len(rows)
        lat = sum(r["latency_s"] for r in rows) / len(rows)
        tok = sum(r["tokens"] for r in rows) / len(rows)
        print(f"  avg calls {calls:.2f}, avg latency {lat:.1f}s, avg tokens {tok:.0f}")

    best = max(data, key=lambda s: sum(r["end_to_end"] for r in data[s]))
    print(f"\nRisky failures in {best} (should refuse/escalate/ask, but answered):")
    for r in data[best]:
        if any(t in r["failure_tags"] for t in ("MISSED_REFUSAL_OR_ESCALATION", "MISSED_CLARIFICATION", "FORBIDDEN_PHRASE")):
            print(f"  {r['query_id']}: expected {r['expected_behavior']}, got {r['predicted_behavior']} [{r['failure_tags']}]")

    by_q = defaultdict(int)
    for rows in data.values():
        for r in rows:
            by_q[r["query_id"]] += r["end_to_end"]
    never = sorted(q for q, v in by_q.items() if v == 0)
    print(f"\nQueries no strategy got fully right: {len(never)} of {len(by_q)}")


if __name__ == "__main__":
    main()
