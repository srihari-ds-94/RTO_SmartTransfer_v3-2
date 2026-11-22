"""Week 4 evaluation and observability report over saved agent runs (no LLM calls).

    python week4_report.py                                    # the three Week 3 reference runs
    python week4_report.py --runs <run_dir> [<run_dir> ...]   # any runs written by run_agent_eval.py
    python week4_report.py --runs A B --labels "Qwen3-8B" "Llama-3.1-8B"

Reads answers.jsonl, config.json and traces.jsonl from each run and writes results/week4_<stamp>/:
    metrics.csv               one row per run: accuracy with 95% Wilson interval, completion, tool calls, cost, latency
    by_category.csv           per run x category: n, end-to-end, behaviour accuracy
    taxonomy_by_category.csv  per run x category: count of each failure type
    stage_latency.csv         per run x stage (planner, tools, compose, critic): avg ms per query and share of the turn
    llm_calls.csv             per run x LLM call type: calls, avg prompt/completion tokens, cost
    tool_calls.csv            per run x tool: calls, errors, avg ms
    per_query.csv             per run x query: outcome, failure type, tokens, cost, latency by stage
    summary.json              everything above for the headline run, plus the pricing used
    dashboard.html            self-contained charts and tables
    chart_*.png               the dashboard charts as separate images

Cost uses pricing.json (USD per 1M input and output tokens, per model). The course endpoint is free; the price
is a reference for what the same traffic would cost on a paid host.
"""
from __future__ import annotations

import argparse
import base64
import csv
import html
import io
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from . import config as cfg

AGENT_RUNS = cfg.RESULTS / "agent"
DEFAULT_RUNS = [
    (AGENT_RUNS / "w3_test_final", "TEST split (final config)"),
    (AGENT_RUNS / "w3_dev_final", "Dev: final (rule critic)"),
    (AGENT_RUNS / "w3_dev_llm_critic", "Dev ablation: LLM critic"),
    (AGENT_RUNS / "w3_dev_faults0.3", "Dev robustness: 30% tool faults"),
]
STAGES = ["planner", "executor.tools", "executor.compose", "critic"]
FAILURES = ["TOOL_CALL_ERROR", "RETRIEVAL_MISS", "HALLUCINATION", "REASONING_ERROR"]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3)


def pct(values: list[float], q: float) -> float:
    v = sorted(values)
    return v[min(len(v) - 1, int(round(q * (len(v) - 1))))] if v else 0.0


def mean(values) -> float:
    values = list(values)
    return round(statistics.mean(values), 3) if values else 0.0


def load_run(path: Path) -> dict:
    answers = [json.loads(x) for x in (path / "answers.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    spans = [json.loads(x) for x in (path / "traces.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    roots = {s["trace_id"]: s for s in spans if s["name"] == "agent.turn"}
    by_trace = defaultdict(list)
    for s in spans:
        if s["trace_id"] in roots:
            by_trace[s["trace_id"]].append(s)
    traces = {}
    for tid, root in roots.items():
        qid = root["attributes"].get("query_id")
        if qid:
            traces[qid] = {"root": root, "spans": by_trace[tid]}
    return {"path": path, "answers": answers, "config": config, "traces": traces}


def price_for(model: str, pricing: dict) -> tuple[float, float]:
    p = pricing.get(model) or {}
    return float(p.get("usd_per_1m_input", 0.0)), float(p.get("usd_per_1m_output", 0.0))


def analyse(run: dict, label: str, pricing: dict) -> dict:
    a, traces = run["answers"], run["traces"]
    n = len(a)
    pin, pout = price_for(run["config"].get("model", ""), pricing)
    cost = lambda pt, ct: pt * pin / 1e6 + ct * pout / 1e6

    per_query, stage_ms = [], defaultdict(list)
    llm_calls, tool_calls = defaultdict(list), defaultdict(list)
    for r in a:
        t = traces.get(r["query_id"])
        stages = {}
        if t:
            root = t["root"]
            for st in STAGES:
                stages[st] = sum(s["duration_ms"] for s in t["spans"] if s["name"] == st and s["parent_id"] == root["span_id"])
            stages["other"] = max(root["duration_ms"] - sum(stages.values()), 0.0)
            for st, ms in stages.items():
                stage_ms[st].append(ms)
            for s in t["spans"]:
                if s["name"].startswith("llm."):
                    llm_calls[s["name"]].append(s)
                elif s["name"].startswith("tool."):
                    tool_calls[s["name"]].append(s)
        per_query.append({
            "run": label, "query_id": r["query_id"], "category": r["category"],
            "expected_behavior": r["expected_behavior"], "predicted_behavior": r["predicted_behavior"],
            "end_to_end": int(bool(r["end_to_end"])), "failure_type": r["failure_type"], "completed": int(bool(r["completed"])),
            "tool_calls": r["tool_calls"], "llm_calls": r["llm_calls"], "prompt_tokens": r["prompt_tokens"],
            "completion_tokens": r["completion_tokens"], "cost_usd": round(cost(r["prompt_tokens"], r["completion_tokens"]), 6),
            "wall_latency_s": r["wall_latency_s"], **{f"{k}_ms": round(v, 1) for k, v in stages.items()},
        })

    passes = sum(q["end_to_end"] for q in per_query)
    lo, hi = wilson(passes, n)
    walls = [r["wall_latency_s"] for r in a]
    costs = [q["cost_usd"] for q in per_query]
    total_tools = sum(r["tool_calls"] for r in a)
    turn_ms = [traces[r["query_id"]]["root"]["duration_ms"] for r in a if r["query_id"] in traces]
    tax = Counter(r["failure_type"] for r in a)
    metrics = {
        "run": label, "folder": run["path"].name, "model": run["config"].get("model"), "split": run["config"].get("split"),
        "llm_critic": run["config"].get("llm_critic"), "fault_rate": run["config"].get("fault_rate"),
        "fresh_cache": run["config"].get("fresh_cache"), "items": n,
        "end_to_end_accuracy": round(passes / n, 3), "e2e_ci95_low": lo, "e2e_ci95_high": hi,
        "behavior_accuracy": mean(float(r["behavior_ok"]) for r in a),
        "must_include_coverage": mean(float(r["include_coverage"]) for r in a),
        "citation_ok_rate": mean(float(r["citations_ok"]) for r in a),
        "task_completion_rate": mean(float(bool(r["completed"])) for r in a),
        "avg_tool_calls": mean(r["tool_calls"] for r in a), "avg_llm_calls": mean(r["llm_calls"] for r in a),
        "tool_error_rate": round(sum(r["tool_errors"] for r in a) / max(total_tools, 1), 3),
        "avg_prompt_tokens": round(mean(r["prompt_tokens"] for r in a)), "avg_completion_tokens": round(mean(r["completion_tokens"] for r in a)),
        "avg_cost_usd_per_query": round(statistics.mean(costs), 6), "cost_usd_per_1000_queries": round(1000 * statistics.mean(costs), 3),
        "cost_usd_per_correct_answer": round(sum(costs) / passes, 6) if passes else None,
        "avg_wall_latency_s": mean(walls), "p50_wall_latency_s": pct(walls, 0.5), "p95_wall_latency_s": pct(walls, 0.95),
        **{f"n_{k}": tax.get(k, 0) for k in ["PASS"] + FAILURES},
    }

    stage_rows = []
    total_turn = sum(turn_ms) or 1.0
    for st in STAGES + ["other"]:
        v = stage_ms.get(st, [])
        stage_rows.append({"run": label, "stage": st, "avg_ms_per_query": round(sum(v) / max(len(turn_ms), 1), 1),
                           "p95_ms": round(pct(v, 0.95), 1), "share_of_turn": round(sum(v) / total_turn, 3)})

    llm_rows = []
    for name, ss in sorted(llm_calls.items()):
        pt = [s["attributes"].get("prompt_tokens", 0) for s in ss]
        ct = [s["attributes"].get("completion_tokens", 0) for s in ss]
        llm_rows.append({"run": label, "llm_call": name, "calls": len(ss), "calls_per_query": round(len(ss) / n, 3),
                         "avg_prompt_tokens": round(mean(pt)), "avg_completion_tokens": round(mean(ct)),
                         "avg_ms": round(mean(s["duration_ms"] for s in ss), 1),
                         "cost_usd_per_query": round(cost(sum(pt), sum(ct)) / n, 6)})

    tool_rows = []
    for name, ss in sorted(tool_calls.items()):
        tool_rows.append({"run": label, "tool": name.removeprefix("tool."), "calls": len(ss),
                          "errors": sum(s["status"] == "ERROR" for s in ss),
                          "avg_ms": round(mean(s["duration_ms"] for s in ss), 1), "p95_ms": round(pct([s["duration_ms"] for s in ss], 0.95), 1)})

    cat_rows, tax_rows = [], []
    by_cat = defaultdict(list)
    for r in a:
        by_cat[r["category"]].append(r)
    for cat, rs in sorted(by_cat.items()):
        k = sum(bool(r["end_to_end"]) for r in rs)
        c_lo, c_hi = wilson(k, len(rs))
        cat_rows.append({"run": label, "category": cat, "n": len(rs), "end_to_end": round(k / len(rs), 3),
                         "ci95_low": c_lo, "ci95_high": c_hi, "behavior_accuracy": mean(float(r["behavior_ok"]) for r in rs)})
        ct = Counter(r["failure_type"] for r in rs)
        tax_rows.append({"run": label, "category": cat, "n": len(rs), **{k: ct.get(k, 0) for k in ["PASS"] + FAILURES}})

    return {"label": label, "metrics": metrics, "stages": stage_rows, "llm": llm_rows, "tools": tool_rows,
            "categories": cat_rows, "taxonomy": tax_rows, "per_query": per_query}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def charts(results: list[dict]) -> dict[str, str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def encode(fig) -> str:
        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", dpi=110)
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode()

    out = {}
    labels = [r["label"] for r in results]
    m = [r["metrics"] for r in results]

    fig, ax = plt.subplots(figsize=(7, 3.2))
    acc = [x["end_to_end_accuracy"] for x in m]
    err = [[x["end_to_end_accuracy"] - x["e2e_ci95_low"] for x in m], [x["e2e_ci95_high"] - x["end_to_end_accuracy"] for x in m]]
    ax.barh(labels, acc, xerr=err, color="#3b6ea5", capsize=4)
    ax.set_xlim(0, 1)
    ax.set_xlabel("End-to-end accuracy (95% Wilson interval)")
    ax.invert_yaxis()
    out["accuracy"] = encode(fig)

    fig, ax = plt.subplots(figsize=(7, 3.2))
    left = [0.0] * len(results)
    colours = {"planner": "#8fb3d9", "executor.tools": "#e3a857", "executor.compose": "#3b6ea5", "critic": "#7fb77e", "other": "#cccccc"}
    for st in STAGES + ["other"]:
        vals = [next(s["avg_ms_per_query"] for s in r["stages"] if s["stage"] == st) / 1000 for r in results]
        ax.barh(labels, vals, left=left, color=colours[st], label=st)
        left = [a + b for a, b in zip(left, vals)]
    ax.set_xlabel("Average seconds per query, by stage")
    ax.legend(fontsize=8, loc="lower right")
    ax.invert_yaxis()
    out["latency"] = encode(fig)

    head = results[0]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    cats = [t["category"] for t in head["taxonomy"]]
    bottom = [0] * len(cats)
    tcol = {"PASS": "#7fb77e", "TOOL_CALL_ERROR": "#b04a4a", "RETRIEVAL_MISS": "#e3a857", "HALLUCINATION": "#8e5ea2", "REASONING_ERROR": "#3b6ea5"}
    for k in ["PASS"] + FAILURES:
        vals = [t[k] for t in head["taxonomy"]]
        ax.bar(cats, vals, bottom=bottom, color=tcol[k], label=k)
        bottom = [a + b for a, b in zip(bottom, vals)]
    ax.set_ylabel("Queries")
    ax.set_title(f"Outcome by category: {head['label']}", fontsize=10)
    ax.legend(fontsize=7, ncol=3)
    out["taxonomy"] = encode(fig)
    return out


def table(rows: list[dict], cols: list[str] | None = None) -> str:
    if not rows:
        return "<p>No data.</p>"
    cols = cols or list(rows[0])
    head = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(r.get(c, '')))}</td>" for c in cols) + "</tr>" for r in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def dashboard(path: Path, results: list[dict], pricing: dict, imgs: dict[str, str]) -> None:
    head = results[0]
    hm = head["metrics"]
    cards = [
        ("End-to-end accuracy", f"{hm['end_to_end_accuracy']:.1%}", f"95% CI {hm['e2e_ci95_low']:.1%} to {hm['e2e_ci95_high']:.1%}, n={hm['items']}"),
        ("Task completion", f"{hm['task_completion_rate']:.1%}", f"tool error rate {hm['tool_error_rate']:.1%}"),
        ("Tool calls per task", f"{hm['avg_tool_calls']:.2f}", f"LLM calls {hm['avg_llm_calls']:.2f}"),
        ("Cost per query", f"${hm['avg_cost_usd_per_query']:.5f}", f"${hm['cost_usd_per_1000_queries']:.2f} per 1,000 queries"),
        ("Latency per query", f"{hm['avg_wall_latency_s']:.1f} s", f"p50 {hm['p50_wall_latency_s']:.1f} s, p95 {hm['p95_wall_latency_s']:.1f} s"),
    ]
    card_html = "".join(f"<div class='card'><div class='k'>{html.escape(k)}</div><div class='v'>{html.escape(v)}</div>"
                        f"<div class='s'>{html.escape(s)}</div></div>" for k, v, s in cards)
    metric_cols = ["run", "items", "end_to_end_accuracy", "e2e_ci95_low", "e2e_ci95_high", "behavior_accuracy",
                   "citation_ok_rate", "task_completion_rate", "avg_tool_calls", "avg_llm_calls", "tool_error_rate",
                   "avg_prompt_tokens", "avg_completion_tokens", "avg_cost_usd_per_query", "avg_wall_latency_s",
                   "p95_wall_latency_s", "fresh_cache"] + [f"n_{k}" for k in ["PASS"] + FAILURES]
    price_rows = [{"model": k, "usd_per_1m_input": v["usd_per_1m_input"], "usd_per_1m_output": v["usd_per_1m_output"],
                   "source": v.get("source", "")} for k, v in pricing.items()]
    worst = sorted((q for q in head["per_query"]), key=lambda q: -q["wall_latency_s"])[:8]
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>RTO SmartTransfer Copilot: Week 4 evaluation</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;margin:24px;color:#222;max-width:1200px}}h1{{font-size:22px}}h2{{font-size:17px;margin-top:28px}}
.cards{{display:flex;gap:12px;flex-wrap:wrap}}.card{{border:1px solid #ddd;border-radius:6px;padding:10px 14px;min-width:190px}}
.k{{font-size:12px;color:#666}}.v{{font-size:22px;font-weight:600}}.s{{font-size:11px;color:#666}}
table{{border-collapse:collapse;font-size:12px;margin:6px 0}}td,th{{border:1px solid #ddd;padding:3px 6px;text-align:left}}th{{background:#f3f5f8}}
.note{{font-size:12px;color:#555}}img{{max-width:100%}}</style></head><body>
<h1>RTO SmartTransfer Copilot: Week 4 evaluation and observability</h1>
<p class="note">Generated {datetime.now():%Y-%m-%d %H:%M} from saved runs. Headline run: {html.escape(head['label'])} ({html.escape(hm['folder'])}), model {html.escape(str(hm['model']))}, {html.escape(str(hm['split']))} split.</p>
<div class="cards">{card_html}</div>
<h2>Runs compared</h2>{table([r['metrics'] for r in results], metric_cols)}
<img src="data:image/png;base64,{imgs['accuracy']}">
<h2>Latency by stage (from OpenTelemetry spans)</h2><img src="data:image/png;base64,{imgs['latency']}">
{table([s for r in results for s in r['stages']])}
<h2>Tool calls</h2>{table([t for r in results for t in r['tools']])}
<h2>LLM calls and cost</h2>{table([x for r in results for x in r['llm']])}
<p class="note">Pricing (reference only; the course endpoint is free):</p>{table(price_rows)}
<h2>Failure taxonomy by category: {html.escape(head['label'])}</h2><img src="data:image/png;base64,{imgs['taxonomy']}">
{table(head['taxonomy'])}
<h2>Accuracy by category: {html.escape(head['label'])}</h2>{table(head['categories'])}
<h2>Slowest queries: {html.escape(head['label'])}</h2>
{table(worst, ['query_id', 'category', 'end_to_end', 'failure_type', 'tool_calls', 'llm_calls', 'wall_latency_s', 'planner_ms', 'executor.tools_ms', 'executor.compose_ms', 'critic_ms'])}
<p class="note">Latency is only comparable between runs with fresh_cache=True; cached LLM responses return instantly.</p>
</body></html>"""
    path.write_text(doc, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", type=Path)
    ap.add_argument("--labels", nargs="*")
    ap.add_argument("--pricing", type=Path, default=cfg.PRICING)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    if args.runs:
        labels = args.labels or [p.name for p in args.runs]
        if len(labels) != len(args.runs):
            sys.exit("--labels must match --runs one to one.")
        runs = list(zip(args.runs, labels))
    else:
        runs = DEFAULT_RUNS
    missing = [str(p) for p, _ in runs if not (p / "answers.jsonl").exists()]
    if missing:
        sys.exit(f"No answers.jsonl in: {', '.join(missing)}")
    pricing = json.loads(args.pricing.read_text(encoding="utf-8"))

    results = [analyse(load_run(p), label, pricing) for p, label in runs]
    out = args.out or cfg.RESULTS / "report" / f"report_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "metrics.csv", [r["metrics"] for r in results])
    write_csv(out / "by_category.csv", [x for r in results for x in r["categories"]])
    write_csv(out / "taxonomy_by_category.csv", [x for r in results for x in r["taxonomy"]])
    write_csv(out / "stage_latency.csv", [x for r in results for x in r["stages"]])
    write_csv(out / "llm_calls.csv", [x for r in results for x in r["llm"]])
    write_csv(out / "tool_calls.csv", [x for r in results for x in r["tools"]])
    write_csv(out / "per_query.csv", [x for r in results for x in r["per_query"]])
    head = results[0]
    (out / "summary.json").write_text(json.dumps({
        "headline": head["metrics"], "stages": head["stages"], "llm_calls": head["llm"], "tools": head["tools"],
        "by_category": head["categories"], "runs": [r["metrics"] for r in results], "pricing": pricing}, indent=2), encoding="utf-8")
    imgs = charts(results)
    for name, b64 in imgs.items():
        (out / f"chart_{name}.png").write_bytes(base64.b64decode(b64))
    dashboard(out / "dashboard.html", results, pricing, imgs)

    for r in results:
        m = r["metrics"]
        print(f"{r['label']:<30} e2e {m['end_to_end_accuracy']:.3f} [{m['e2e_ci95_low']:.3f}, {m['e2e_ci95_high']:.3f}]  "
              f"complete {m['task_completion_rate']:.3f}  tools {m['avg_tool_calls']:.2f}  "
              f"${m['avg_cost_usd_per_query']:.5f}/query  {m['avg_wall_latency_s']:.1f}s avg  {m['p95_wall_latency_s']:.1f}s p95")
    print("Stages (headline run):", ", ".join(f"{s['stage']} {s['avg_ms_per_query'] / 1000:.1f}s ({s['share_of_turn']:.0%})" for s in head["stages"]))
    print(f"Results: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
