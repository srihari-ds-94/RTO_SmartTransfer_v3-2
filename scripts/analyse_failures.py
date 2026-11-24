"""Collects the failure analysis used by the taxonomy PDF and the technical report into report/analysis.json.

Reads only saved runs (answers.jsonl) and the frozen v1 eval rows; makes no model calls."""
import csv, json, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rto_copilot import config, scoring, nbkit  # noqa: E402

ev = {r["query_id"]: r for r in scoring.read_csv(config.eval_path("v2"))}
RUNS = {"dev": "w3_dev_final", "test": "w3_test_final", "dev_llm_critic": "w3_dev_llm_critic", "dev_faults": "w3_dev_faults0.3"}


def missing_phrases(row, final):
    text = " ".join([str(final.get("answer", ""))] + [str(d.get("content", "")) for d in final.get("drafts") or []]
                    + [str(q) for q in final.get("clarifying_questions") or []]).lower()
    out = []
    for item in scoring.split_list(row["must_include"]):
        if not any(a.lower() in text for a in scoring.split_list(item, "|")):
            out.append(item)
    return out


out = {"runs": {}}
for key, name in RUNS.items():
    rows = nbkit.agent_run(name)
    cm = Counter()
    fails = []
    for r in rows:
        row = ev[r["query_id"]]
        cm[(r["expected_behavior"], r["predicted_behavior"])] += 1
        if r["end_to_end"]:
            continue
        final = r.get("final") or {}
        fails.append({
            "query_id": r["query_id"], "category": r["category"], "query": row["query"], "expected": r["expected_behavior"],
            "predicted": r["predicted_behavior"], "failure_type": r["failure_type"], "flags": r["failure_flags"],
            "tags": r["failure_tags"], "missing": missing_phrases(row, final), "answer": str(final.get("answer", ""))[:420],
            "gold": row["gold_answer"][:300], "context_sources": r.get("context_sources", []),
            "gold_sources": scoring.split_list(row["gold_source_ids"]), "tool_calls": r["tool_calls"],
            "tools": [t["tool"] for t in r.get("tools", [])], "stop_reason": r["stop_reason"], "revisions": r["revisions"],
        })
    out["runs"][key] = {"run": name, "n": len(rows), "passed": sum(bool(r["end_to_end"]) for r in rows),
                        "confusion": [[e, p, n] for (e, p), n in sorted(cm.items())], "failures": fails}

# sub-causes among failures on the two headline runs
sub = {}
for key in ("dev", "test"):
    f = out["runs"][key]["failures"]
    c = Counter()
    for x in f:
        right = x["expected"] == x["predicted"] or (x["expected"] in ("ANSWER", "ANSWER_WITH_CAVEAT") and x["predicted"] in ("ANSWER", "ANSWER_WITH_CAVEAT") and False)
        if x["expected"] == x["predicted"]:
            kind = "right behaviour, missing facts or wording" if x["missing"] or "FORBIDDEN" in x["tags"] else "right behaviour, citation or other check"
        elif x["expected"] in ("ANSWER", "ANSWER_WITH_CAVEAT") and x["predicted"] in ("ANSWER", "ANSWER_WITH_CAVEAT"):
            kind = "plain answer vs caveat mismatch"
        elif x["expected"].startswith(("REFUSE", "ESCALATE", "ABSTAIN")) and x["predicted"].startswith(("REFUSE", "ESCALATE", "ABSTAIN")):
            kind = "wrong kind of refusal or escalation"
        elif x["predicted"].startswith(("REFUSE", "ESCALATE", "ABSTAIN")):
            kind = "over-refusal"
        elif x["expected"].startswith(("REFUSE", "ESCALATE", "ABSTAIN")):
            kind = "missed refusal or escalation"
        elif x["expected"] == "ASK_CLARIFICATION" or x["predicted"] == "ASK_CLARIFICATION":
            kind = "clarification mismatch"
        else:
            kind = "other behaviour mismatch"
        c[kind] += 1
        x["subcause"] = kind
    sub[key] = dict(c)
out["subcauses"] = sub
(ROOT / "report").mkdir(exist_ok=True)
(ROOT / "report" / "analysis.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print({k: v for k, v in sub.items()})
for key in ("dev", "test"):
    print(key, out["runs"][key]["n"], out["runs"][key]["passed"], Counter(f["failure_type"] for f in out["runs"][key]["failures"]))
