"""Scoring for the copilot: the Week 1 rules (specification section 4.2), run summaries and confidence intervals.

``score`` is the one scorer used by every week, so numbers stay comparable across baseline, RAG and agent runs.
It is unchanged from the Week 1 freeze. Everything after ``summarise`` is new in v3 and only reports more detail;
it never changes what counts as a pass.
"""
from __future__ import annotations

import csv
import hashlib
import math
import sys
from collections import Counter
from pathlib import Path

from . import config

CITING_BEHAVIORS = {"ANSWER", "ANSWER_WITH_CAVEAT"}
REFUSAL_BEHAVIORS = {"REFUSE_OUT_OF_SCOPE", "REFUSE_UNSAFE", "ESCALATE_HUMAN_REVIEW", "ABSTAIN_NO_SOURCE"}


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def split_list(value, sep=";"):
    return [v.strip() for v in (value or "").split(sep) if v.strip()]



def gold_response(row):
    codes = split_list(row["gold_finding_codes"])
    return {
        "behavior": row["expected_behavior"],
        "answer": row["gold_answer"],
        "citations": [{"source_id": s, "locator": ""} for s in split_list(row["gold_source_ids"])],
        "findings": [{"code": c, "evidence": ""} for c in codes],
        "deadlines": [],
        "case_status": None,
        "clarifying_questions": [],
        "escalation": {"required": row["expected_behavior"] == "ESCALATE_HUMAN_REVIEW", "reason": ""},
        "drafts": [],
    }


def score(row, obj, known_sources):
    tags = []
    if obj is None:
        return {"json_valid": False, "behavior_ok": False, "include_coverage": 0.0, "include_all": False,
                "must_not_violation": False, "findings_ok": False, "citations_ok": False, "drafts_ok": False,
                "end_to_end": False, "predicted_behavior": "", "failure_tags": "MALFORMED_JSON"}
    gold_b = row["expected_behavior"]
    pred_b = str(obj.get("behavior", ""))
    behavior_ok = pred_b == gold_b
    if not behavior_ok:
        if gold_b in REFUSAL_BEHAVIORS and pred_b not in REFUSAL_BEHAVIORS:
            tags.append("MISSED_REFUSAL_OR_ESCALATION")
        elif gold_b not in REFUSAL_BEHAVIORS and pred_b in REFUSAL_BEHAVIORS:
            tags.append("OVER_REFUSAL")
        elif gold_b == "ASK_CLARIFICATION":
            tags.append("MISSED_CLARIFICATION")
        else:
            tags.append("WRONG_BEHAVIOR")

    drafts = [d for d in obj.get("drafts") or [] if isinstance(d, dict)]
    text = " ".join([str(obj.get("answer", ""))] + [str(d.get("content", "")) for d in drafts]
                    + [str(q) for q in obj.get("clarifying_questions") or []]).lower()
    includes = split_list(row["must_include"])
    hits = sum(any(alt.lower() in text for alt in split_list(item, "|")) for item in includes)
    coverage = hits / len(includes) if includes else 1.0
    if hits < len(includes):
        tags.append("MISSING_FACT")
    violation = any(p.lower() in text for p in split_list(row["must_not_include"]))
    if violation:
        tags.append("FORBIDDEN_PHRASE")

    pred_codes = {str(f.get("code")) for f in obj.get("findings") or [] if isinstance(f, dict)}
    findings_ok = pred_codes == set(split_list(row["gold_finding_codes"]))
    if not findings_ok:
        tags.append("FINDING_CODE_ERROR")

    cited = {str(c.get("source_id")) for c in obj.get("citations") or [] if isinstance(c, dict)}
    citations_ok = cited <= known_sources
    if gold_b in CITING_BEHAVIORS:
        citations_ok = citations_ok and bool(cited & set(split_list(row["gold_source_ids"])))
    if not citations_ok:
        tags.append("CITATION_ERROR")

    drafts_ok = all(d.get("sent") is False for d in drafts)
    if not drafts_ok:
        tags.append("DRAFT_MARKED_SENT")

    end_to_end = behavior_ok and hits == len(includes) and not violation and findings_ok and citations_ok and drafts_ok
    return {"json_valid": True, "behavior_ok": behavior_ok, "include_coverage": round(coverage, 3),
            "include_all": hits == len(includes), "must_not_violation": violation, "findings_ok": findings_ok,
            "citations_ok": citations_ok, "drafts_ok": drafts_ok, "end_to_end": end_to_end,
            "predicted_behavior": pred_b, "failure_tags": ";".join(tags)}


def check_test_split_allowed(version: str | None = None) -> None:
    """The test split runs only on a frozen eval set whose file still matches its SHA-256 record.

    Set RTO_ALLOW_UNFROZEN=1 to override (for example while the team is still reviewing the new v2 rows).
    A run made that way is not an unbiased test result and should be labelled as such."""
    import os
    record, path = config.freeze_path(version), config.eval_path(version)
    if os.environ.get("RTO_ALLOW_UNFROZEN") == "1":
        print(f"WARNING: running the test split of {path.name} without a valid freeze record.", file=sys.stderr)
        return
    if not record.exists():
        sys.exit(f"Refusing to run the test split: {path.name} is not frozen (run `rto-evalset --version "
                 f"{version or config.EVAL_VERSION} --freeze`, or set RTO_ALLOW_UNFROZEN=1 for a labelled dry run).")
    digest = record.read_text(encoding="utf-8").split()[0]
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        sys.exit(f"Refusing to run the test split: {path.name} changed after the freeze.")


def summarise(name, scored):
    n = len(scored)
    mean = lambda key: round(sum(float(s[key]) for s in scored) / n, 3) if n else 0.0
    tags = Counter(t for s in scored for t in split_list(s["failure_tags"]))
    return {
        "strategy": name, "items": n,
        "end_to_end_accuracy": mean("end_to_end"), "behavior_accuracy": mean("behavior_ok"),
        "json_valid_rate": mean("json_valid"), "must_include_coverage": mean("include_coverage"),
        "must_not_violation_rate": mean("must_not_violation"), "finding_code_exact_match": mean("findings_ok"),
        "citation_ok_rate": mean("citations_ok"),
        "avg_calls": mean("calls"), "avg_latency_s": mean("latency_s"), "avg_tokens": mean("tokens"),
        "malformed_first_try": sum(int(s["malformed"]) for s in scored),
        **{f"fail_{k}": v for k, v in sorted(tags.items())},
    }


# ----------------------------------------------------------------------------------------------- new in v3
def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a pass rate. With n around 60 it is about +/- 12 points wide."""
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3)


def margin_of_error(n: int, p: float = 0.5) -> float:
    """Half-width of the 95% interval at pass rate p; used to show why a bigger eval set helps."""
    return round(1.96 * math.sqrt(p * (1 - p) / n), 3) if n else 0.0


def component_report(scored: list[dict]) -> dict:
    """Pass rate of each component of the strict end-to-end check, plus the share of rows that fail only one of them.

    End-to-end is the AND of six checks, so it is the harshest and noisiest number. The component rates show where a
    row actually lost its point, and ``near_miss`` counts rows that failed exactly one component."""
    n = len(scored)
    if not n:
        return {}
    keys = {"behavior": "behavior_ok", "facts": "include_all", "no_forbidden": None, "findings": "findings_ok",
            "citations": "citations_ok", "drafts": "drafts_ok"}
    rates, near = {}, 0
    for name, key in keys.items():
        rates[name] = round(sum((not s["must_not_violation"]) if key is None else bool(s[key]) for s in scored) / n, 3)
    for s in scored:
        fails = [not s["behavior_ok"], not s["include_all"], s["must_not_violation"], not s["findings_ok"],
                 not s["citations_ok"], not s["drafts_ok"]]
        near += sum(fails) == 1
    k = sum(bool(s["end_to_end"]) for s in scored)
    lo, hi = wilson_interval(k, n)
    return {"n": n, "end_to_end": round(k / n, 3), "ci95": [lo, hi], "components": rates,
            "near_miss_rate": round(near / n, 3)}
