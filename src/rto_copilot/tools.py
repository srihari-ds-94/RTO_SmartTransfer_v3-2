"""Deterministic tools for the agent. The LLM may explain their output but never overwrites it (spec section 2).

    lookup_case(case_id)                         case record (synthetic), without the gold columns
    check_case(case_id)                          finding codes, case_status, form path, escalation triggers
    loan_procedure(scenario)                     forms, consent and registry rules per loan scenario (no case needed)
    resolve_date(text, today)                    'yesterday', '3 days ago', '2026-10-02' -> ISO date
    compute_deadlines(transfer_date, today, role) due dates from rule_registry.csv (day of transfer excluded)
    search_corpus(query, case_id)                Week 2 retriever over the official corpus
    get_source(source_id, query)                 best chunk of one source, e.g. the source a registry rule cites

Every tool returns a JSON-serialisable dict or raises ToolError with a message the agent can show the model.
"""
from __future__ import annotations

import csv
import re
from datetime import date, timedelta

from . import config

CASES = config.CASES
RULES = config.RULES
HIDDEN_CASE_FIELDS = {"expected_findings", "split"}
CASE_ID = re.compile(r"\bDEMO-\d{3}\b")
ISO_DATE = re.compile(r"\b(20\d\d)-(\d\d)-(\d\d)\b")

# spec section 1: copilot path per loan scenario
FORM_PATH = {
    "NO_LOAN": "No hypothecation: seller files Form 29, buyer applies in Form 30 with the supporting documents.",
    "LOAN_TERMINATED": "Loan repaid: Form 35 signed by owner and financier with the financier's due-clearance "
                       "certificate removes the hypothecation; then Form 29 and Form 30.",
    "LOAN_ACTIVE_WITH_CONSENT": "Loan running with the financier's written consent: the transfer is recorded under "
                                "continuation of the hypothecation (financier consent on Form 30, MVA s.51(4)). "
                                "West Bengal availability is not officially confirmed, so the case needs human review.",
    "LOAN_ACTIVE_PENDING": "Loan running and the financier's written consent is not yet given: no transfer entry "
                           "without it (MVA s.51(4)). Either obtain written consent or repay and file Form 35. "
                           "Never marked ready; a draft request to the financier can be prepared.",
}
SCENARIO_RULES = {
    "LOAN_TERMINATED": ["R-HP-TERMINATION-FORM35"],
    "LOAN_ACTIVE_WITH_CONSENT": ["R-FINANCIER-CONSENT", "R-CONSENT-CONTINUATION"],
    "LOAN_ACTIVE_PENDING": ["R-FINANCIER-CONSENT", "R-HP-TERMINATION-FORM35"],
}
# who normally provides each document
DOCUMENTS = [("has_rc", "MISSING_RC", "registration certificate (RC)", "seller"),
             ("has_insurance", "MISSING_INSURANCE", "insurance certificate", "seller"),
             ("has_puc", "MISSING_PUC", "PUC (pollution under control) certificate", "seller"),
             ("has_form_30", "MISSING_FORM30", "Form 30", "buyer")]


class ToolError(Exception):
    pass


def _cases() -> dict[str, dict]:
    if not hasattr(_cases, "cache"):
        with CASES.open(newline="", encoding="utf-8-sig") as f:
            _cases.cache = {r["case_id"]: r for r in csv.DictReader(f)}
    return _cases.cache


def _rules() -> list[dict]:
    if not hasattr(_rules, "cache"):
        with RULES.open(newline="", encoding="utf-8-sig") as f:
            _rules.cache = list(csv.DictReader(f))
    return _rules.cache


def _case_id(case_id) -> str:
    if not isinstance(case_id, str) or not CASE_ID.fullmatch(case_id.strip().upper()):
        raise ToolError(f"case_id must look like DEMO-123, got {case_id!r}")
    return case_id.strip().upper()


def _iso(value, name: str) -> date:
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        raise ToolError(f"{name} must be an ISO date YYYY-MM-DD, got {value!r}") from None


def lookup_case(case_id: str) -> dict:
    cid = _case_id(case_id)
    rec = _cases().get(cid)
    if rec is None:
        return {"found": False, "case_id": cid, "message": f"No case record exists for {cid}."}
    return {"found": True, "case_id": cid, "record": {k: v for k, v in rec.items() if k not in HIDDEN_CASE_FIELDS}}


def _scenario_rules(scenario: str) -> list[dict]:
    by_id = {r["rule_id"]: r for r in _rules()}
    return [{"rule_id": rid, "statement": by_id[rid]["statement"], "rule_status": by_id[rid]["rule_status"],
             "source_id": by_id[rid]["source_id"], "locator": by_id[rid]["locator"]}
            for rid in SCENARIO_RULES.get(scenario, []) if rid in by_id]


def loan_procedure(scenario: str = "all") -> dict:
    """Spec section 1 path and registry rules for a loan scenario, for questions without a case record."""
    name = str(scenario or "all").strip().upper()
    names = list(FORM_PATH) if name in ("ALL", "UNKNOWN") else [name]
    if any(n not in FORM_PATH for n in names):
        raise ToolError(f"scenario must be one of {list(FORM_PATH)} or 'all', got {scenario!r}")
    return {"scenarios": [{"scenario": n, "path": FORM_PATH[n], "rules": _scenario_rules(n)} for n in names],
            "note": "Repossession by the financier after default (Form 36) is out of scope."}


def check_case(case_id: str) -> dict:
    """Spec rule 3 finding codes and section 1 case status. Uses only the case record, never expected_findings."""
    looked = lookup_case(case_id)
    if not looked["found"]:
        return looked
    rec = looked["record"]
    findings, missing = [], []
    for col, code, name, party in DOCUMENTS:
        if rec.get(col) == "0":
            who = rec["seller_name"] if party == "seller" else rec["buyer_name"]
            findings.append({"code": code, "evidence": f"{col}=0"})
            missing.append({"item": name, "provided_by": party, "name": who})
    if rec.get("has_form_29") == "0":
        missing.append({"item": "Form 29", "provided_by": "seller", "name": rec["seller_name"]})
    if rec.get("has_rc") == "1" and rec["engine_number_declared"] != rec["engine_number_rc"]:
        findings.append({"code": "ENGINE_MISMATCH", "evidence": f"declared {rec['engine_number_declared']} vs RC "
                                                                f"{rec['engine_number_rc']}"})
    if rec.get("financier_status") == "pending":
        findings.append({"code": "FINANCIER_REVIEW_REQUIRED", "evidence": "financier_status=pending"})

    escalation = []
    if any(f["code"] == "ENGINE_MISMATCH" for f in findings):
        escalation.append("engine number on the declaration differs from the RC")
    if rec["seller_name"].strip().lower() == rec["buyer_name"].strip().lower():
        escalation.append(f"buyer and seller have the same name ({rec['seller_name']})")

    ready = not findings and not escalation and rec["scenario"] in ("NO_LOAN", "LOAN_TERMINATED")
    rules = _scenario_rules(rec["scenario"])
    return {"found": True, "case_id": rec["case_id"], "scenario": rec["scenario"],
            "financier_status": rec["financier_status"], "findings": findings, "missing_items": missing,
            "case_status": "READY_FOR_MANUAL_SUBMISSION" if ready else "REVIEW_REQUIRED",
            "status_note": "Ready for manual submission is not an approval; the RTO decides." if ready else
                           "Needs review before submission.",
            "form_path": FORM_PATH.get(rec["scenario"], ""), "rules": rules, "escalation_triggers": escalation,
            "parties": {"seller": rec["seller_name"], "buyer": rec["buyer_name"]},
            "vehicle": {"registration_number": rec["registration_number"], "chassis_number": rec["chassis_number"],
                        "engine_number_declared": rec["engine_number_declared"],
                        "engine_number_rc": rec["engine_number_rc"]},
            "sale_date": rec["sale_date"]}


_WORD_NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "ten": 10}


def resolve_date(text: str, today: str) -> dict:
    """Resolve a transfer-date expression against TODAY. Vague or future dates are errors (ask the user)."""
    now = _iso(today, "today")
    t = str(text or "").strip().lower()
    if not t:
        raise ToolError("no date given")
    m = ISO_DATE.search(t)
    if m:
        d = _iso(m.group(0), "date")
        how = "explicit date"
    elif "day before yesterday" in t:
        d, how = now - timedelta(days=2), "day before yesterday"
    elif "yesterday" in t:
        d, how = now - timedelta(days=1), "yesterday"
    elif re.search(r"\btoday\b", t):
        d, how = now, "today"
    else:
        m = re.search(r"\b(\d+|one|two|three|four|five|six|seven|ten)\s+days?\s+ago\b", t)
        if not m:
            raise ToolError(f"cannot resolve {text!r} to a single date; ask the user for the exact transfer date")
        n = int(m.group(1)) if m.group(1).isdigit() else _WORD_NUMBERS[m.group(1)]
        d, how = now - timedelta(days=n), f"{n} days ago"
    if d > now:
        raise ToolError(f"transfer date {d.isoformat()} is in the future (today is {now.isoformat()})")
    return {"date": d.isoformat(), "resolved_from": how, "today": now.isoformat()}


def compute_deadlines(transfer_date: str, today: str | None = None, role: str = "both") -> dict:
    """Due date = transfer date + window_days (R-DAY-COUNT: day of transfer excluded)."""
    start = _iso(transfer_date, "transfer_date")
    now = _iso(today, "today") if today else None
    role = (role or "both").lower()
    if role not in ("seller", "buyer", "both"):
        role = "both"
    out = []
    for r in _rules():
        if not r["window_days"]:
            continue
        party = "seller" if "SELLER" in r["rule_id"] else "buyer"
        if role != "both" and party != role:
            continue
        due = start + timedelta(days=int(r["window_days"]))
        row = {"rule_id": r["rule_id"], "party": party, "window_days": int(r["window_days"]),
               "due_date": due.isoformat(), "conflict_group": r["conflict_group"] or None,
               "rule_status": r["rule_status"], "statement": r["statement"], "source_id": r["source_id"],
               "locator": r["locator"]}
        if now:
            row["days_left"] = (due - now).days
            row["timing"] = "overdue" if now > due else "open"
        out.append(row)
    groups: dict[str, list[dict]] = {}
    for row in out:
        if row["conflict_group"]:
            groups.setdefault(row["conflict_group"], []).append(row)
    conflicts = []
    for g, rows in groups.items():
        if len(rows) > 1:
            for row in rows:
                row["status"] = "conflict"
            earliest = min(rows, key=lambda x: x["due_date"])
            conflicts.append({"group": g, "rule_ids": [x["rule_id"] for x in rows],
                              "due_dates": [x["due_date"] for x in rows], "plan_to": earliest["due_date"],
                              "note": "The sources give different windows; show both dates and plan to the earlier one."})
    for row in out:
        row.setdefault("status", row.get("timing", "open"))
    day_rule = next((r for r in _rules() if r["rule_id"] == "R-DAY-COUNT"), {})
    return {"transfer_date": start.isoformat(), "today": now.isoformat() if now else None,
            "day_count": "day of transfer excluded (R-DAY-COUNT, team decision)",
            "day_count_source": {"source_id": day_rule.get("source_id"), "locator": day_rule.get("locator")},
            "deadlines": out, "conflicts": conflicts}


class CorpusSearch:
    """search_corpus tool: the Week 2 final retriever (w128, case-aware query, hybrid + reranker fused by RRF)."""

    def __init__(self, method: str = "hybrid_rerank_rrf", config: str = "w128", chunks: int = 8, per_source: int = 2):
        from . import retrieval  # imported here so tools can be used without sentence-transformers installed
        self.r = retrieval
        self.ret = retrieval.Retriever(config, use_reranker="rerank" in method)
        self.method, self.chunks, self.per_source = method, chunks, per_source

    def __call__(self, query: str, case_id: str | None = None, k: int | None = None) -> dict:
        if not isinstance(query, str) or not query.strip():
            raise ToolError("query must be a non-empty string")
        case = _cases().get(case_id) if case_id else None
        text = self.r.case_aware_query(query, case)
        k = max(1, min(int(k or self.chunks), 10))
        picked, count = [], {}
        for i in self.ret.search(text, self.method):
            c = self.ret.chunks[i]
            if count.get(c["source_id"], 0) < self.per_source:
                picked.append(c)
                count[c["source_id"]] = count.get(c["source_id"], 0) + 1
            if len(picked) >= k:
                break
        return {"query": text, "results": [self._out(c) for c in picked]}

    @staticmethod
    def _out(c: dict) -> dict:
        return {"source_id": c["source_id"], "title": c["title"], "locator": c["locator"],
                "text": c["text"][len(c["title"]) + 2:]}

    def get_source(self, source_id: str, query: str | None = None) -> dict:
        """get_source tool: the best chunk of one registered source (e.g. the source a rule cites)."""
        own = [i for i, c in enumerate(self.ret.chunks) if c["source_id"] == source_id]
        if not own:
            raise ToolError(f"source {source_id!r} is not in the corpus")
        best = own[0]
        if query:
            scores = self.ret.emb[own] @ self.ret.embedder.encode([self.r.QUERY_PREFIX + query],
                                                                   normalize_embeddings=True)[0]
            best = own[int(scores.argmax())]
        return {"results": [self._out(self.ret.chunks[best])]}
