"""Trace dashboard: one self-contained HTML page built from the real OpenTelemetry traces of the agent.

    rto-dashboard                         # saved runs in results/agent  ->  dashboard/trace_dashboard.html
    rto-dashboard --runs results/agent/w3_dev_final --out my.html

The page needs no server: open the file in a browser. When the web app is running it is also served at /dashboard,
where an extra "Live" run shows the questions asked in the UI during this session (state/traces.jsonl).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from . import config

TEMPLATE = config.WEB / "dashboard.html"
LABELS = {
    "w3_dev_final": "Dev, final configuration",
    "w3_test_final": "Test, held-out split",
    "w3_dev_llm_critic": "Dev, ablation: LLM critic",
    "w3_dev_faults0.3": "Dev, 30% injected tool faults",
}
_KEEP = {"query", "query_id", "behavior", "llm_calls", "tool_calls", "tokens", "stop_reason", "completed", "repaired",
         "plan", "args", "issues", "round", "revision", "chunks", "exemplars", "attempt", "prompt_tokens",
         "completion_tokens", "json_valid", "error", "steps", "attempts", "skipped", "est_prompt_tokens"}


def _trim(attrs: dict) -> dict:
    out = {}
    for k, v in attrs.items():
        if k in _KEEP or k.startswith("gen_ai"):
            out[k] = v if not isinstance(v, str) or len(v) <= 600 else v[:600] + "..."
    return out


def read_spans(path: Path) -> list[dict]:
    if not path.exists():
        return []
    spans = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            spans.append(json.loads(line))
        except ValueError:
            continue  # a half-written last line while the app is running
    return spans


def structure(spans: list[dict], answers: dict[str, dict] | None = None, limit: int | None = None) -> list[dict]:
    """Group spans into one record per query (one agent.turn root each), oldest first."""
    answers = answers or {}
    by_trace = defaultdict(list)
    for s in spans:
        by_trace[s["trace_id"]].append(s)
    traces = []
    for tid, ss in by_trace.items():
        root = next((s for s in ss if s["name"] == "agent.turn"), None)
        if root is None:
            continue
        t0 = root["start_ns"]
        a = root["attributes"]
        ans = answers.get(a.get("query_id") or "", {})
        final = ans.get("final") or {}
        traces.append({
            "id": tid[:8], "query_id": a.get("query_id"), "query": a.get("query", ""), "behavior": a.get("behavior"),
            "ms": round(root["duration_ms"], 1), "tokens": a.get("tokens", 0), "llm_calls": a.get("llm_calls", 0),
            "tool_calls": a.get("tool_calls", 0), "stop_reason": a.get("stop_reason"), "start_ns": t0,
            "category": ans.get("category"), "expected": ans.get("expected_behavior"),
            "passed": ans.get("end_to_end"), "failure": ans.get("failure_type"), "tags": ans.get("failure_tags"),
            "plan": ans.get("plan"), "critic": ans.get("critic"), "tools": ans.get("tools"),
            "answer": final.get("answer"), "citations": final.get("citations"), "drafts": len(final.get("drafts") or []),
            "spans": [{"i": s["span_id"][:6], "p": (s["parent_id"] or "")[:6], "n": s["name"],
                       "s": round((s["start_ns"] - t0) / 1e6, 1), "d": round(s["duration_ms"], 1), "e": s["status"] == "ERROR",
                       "a": _trim(s["attributes"])} for s in sorted(ss, key=lambda x: x["start_ns"])],
        })
    traces.sort(key=lambda t: t["start_ns"])
    for t in traces:
        del t["start_ns"]
    return traces[-limit:] if limit else traces


def load_run(folder: Path, label: str | None = None) -> dict:
    answers = {}
    f = folder / "answers.jsonl"
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            answers[r["query_id"]] = r
    cfgf = folder / "config.json"
    model = json.loads(cfgf.read_text()).get("model") if cfgf.exists() else None
    return {"id": folder.name, "label": label or LABELS.get(folder.name, folder.name), "model": model,
            "traces": structure(read_spans(folder / "traces.jsonl"), answers)}


def live_run(limit: int = 60) -> dict:
    return {"id": "live", "label": "Live: questions asked in the UI this session", "model": None, "live": True,
            "traces": structure(read_spans(config.STATE / "traces.jsonl"), None, limit)}


def default_runs() -> list[Path]:
    root = config.RESULTS / "agent"
    named = [root / n for n in LABELS if (root / n / "traces.jsonl").exists()]
    other = [p for p in sorted(root.glob("agent_*")) if (p / "traces.jsonl").exists()]
    return named + other


def build_html(runs: list[dict], live: bool = False) -> str:
    data = json.dumps({"runs": runs, "live": live}, separators=(",", ":")).replace("</", "<\\/")
    return TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/null", data)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", type=Path, help="run folders containing traces.jsonl (default: saved runs)")
    ap.add_argument("--out", type=Path, default=config.ROOT / "dashboard" / "trace_dashboard.html")
    args = ap.parse_args()
    folders = args.runs or default_runs()
    runs = [load_run(Path(p)) for p in folders]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build_html(runs), encoding="utf-8")
    n = sum(len(r["traces"]) for r in runs)
    print(f"Wrote {args.out} ({len(runs)} runs, {n} traced queries, {args.out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
