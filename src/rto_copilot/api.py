"""Minimal web layer: FastAPI + one static page.

    rto-serve                      # http://127.0.0.1:8000
    RTO_HOST=0.0.0.0 RTO_PORT=8080 rto-serve

The page works without an LLM key for case lookup (deterministic checks and deadlines). Asking the copilot needs
LLM_BASE_URL, LLM_API_KEY and LLM_MODEL (see .env.example). Nothing here sends, files, pays or approves anything.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path
from datetime import date

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from . import config, tools, evidence, models, scoring, textmetrics, dashboard

_DECISIONS = ("approved_for_manual_use", "rejected", "needs_changes")


class AskIn(BaseModel):
    session_id: str = Field(min_length=1, max_length=64)
    query: str = Field(max_length=2000)
    case_id: str | None = None
    today: str | None = None
    model_id: str | None = None   # which model profile answers (see /api/models)
    eval_id: str | None = None   # demo: score this answer against the eval set's reference answer


class ReviewIn(BaseModel):
    session_id: str = Field(min_length=1, max_length=64)
    reviewer: str = Field(min_length=1, max_length=80)
    decision: str
    draft: dict
    note: str = Field(default="", max_length=500)


def _eval_rows() -> dict[str, dict]:
    """Dev rows of the current eval set (the test split is never offered in the UI)."""
    with config.EVAL.open(encoding="utf-8-sig", newline="") as f:
        return {r["query_id"]: r for r in csv.DictReader(f) if r["split"] == "dev"}


def _cases() -> list[dict]:
    with config.CASES.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def create_app(copilot=None) -> FastAPI:
    """`copilot` can be injected (tests, notebooks). Otherwise it is built lazily from the environment."""
    app = FastAPI(title="RTO SmartTransfer Copilot", version="3.0.0", docs_url=None, redoc_url=None)
    state = {"injected": copilot, "built": {}, "errors": {}, "search": None}
    config.load_dotenv()

    def profiles() -> dict:
        return {p.id: p for p in models.load_profiles()}

    def pick(model_id: str | None):
        """Return (profile, error). Default is the first ready model in the list."""
        ps = profiles()
        if model_id:
            if model_id not in ps:
                return None, f"Unknown model '{model_id}'."
            ok, why = ps[model_id].ready()
            return (ps[model_id], None) if ok else (None, f"{ps[model_id].label} is not ready: {why}")
        for p in ps.values():
            if p.ready()[0]:
                return p, None
        return None, "no model is configured"

    def get_copilot(model_id: str | None = None):
        """(copilot, profile, error). Built lazily, one per model, sharing one retriever."""
        if state["injected"] is not None:
            return state["injected"], None, None
        prof, err = pick(model_id)
        if prof is None:
            return None, None, err
        if prof.id in state["built"]:
            return state["built"][prof.id], prof, None
        if prof.id in state["errors"]:
            return None, prof, state["errors"][prof.id]
        try:
            from .serving import Copilot
            if state["search"] is None:
                state["search"] = tools.CorpusSearch()
            state["built"][prof.id] = Copilot.for_profile(prof, state["search"])
        except Exception as exc:  # retriever or client could not start
            state["errors"][prof.id] = str(exc)
            return None, prof, str(exc)
        return state["built"][prof.id], prof, None

    @app.get("/")
    def index():
        return FileResponse(config.WEB / "index.html")

    @app.get("/api/health")
    def health():
        cert = os.environ.get("LLM_CA_CERT", "")
        ready = [m for m in _model_list() if m["ready"]]
        return {"ok": True, "llm_ready": bool(ready), "llm_error": None if ready else "no model is ready",
                "models_ready": [m["id"] for m in ready], "cases": len(_cases()), "eval_set": config.EVAL_VERSION,
                "missing_env": config.missing_llm_env(), "ca_cert": cert or None,
                "ca_cert_found": bool(cert) and Path(cert).exists()}

    def _model_list() -> list[dict]:
        if state["injected"] is not None:
            m = getattr(state["injected"].agent.client, "model", "model")
            return [{"id": "default", "label": m, "model": m, "kind": "remote", "ready": True, "reason": ""}]
        return [p.public() for p in profiles().values()]

    @app.get("/dashboard")
    def trace_dashboard():
        from fastapi.responses import HTMLResponse
        runs = [dashboard.load_run(p) for p in dashboard.default_runs()]
        return HTMLResponse(dashboard.build_html(runs, live=True))

    @app.get("/api/traces/live")
    def traces_live(limit: int = 60):
        return dashboard.live_run(max(1, min(limit, 300)))

    @app.get("/api/models")
    def model_list():
        items = _model_list()
        default = next((m["id"] for m in items if m["ready"]), None)
        return {"default": default, "models": items}

    @app.get("/api/cases")
    def cases():
        return [{"case_id": c["case_id"], "scenario": c["scenario"], "seller": c["seller_name"],
                 "buyer": c["buyer_name"], "sale_date": c["sale_date"]} for c in _cases()]

    @app.get("/api/evalset")
    def evalset():
        return [{"query_id": r["query_id"], "category": r["category"], "query": r["query"], "case_id": r["case_id"] or None,
                 "reference_date": r["reference_date"] or None} for r in _eval_rows().values()]

    @app.get("/api/case/{case_id}")
    def case(case_id: str, today: str | None = None):
        """Deterministic view: document findings, status and deadlines. No LLM involved."""
        checked = tools.check_case(case_id)
        if not checked.get("found"):
            raise HTTPException(404, checked.get("message", "Unknown case"))
        try:
            deadlines = tools.compute_deadlines(checked["sale_date"], today or date.today().isoformat(), "both")
        except Exception as exc:
            deadlines = {"error": str(exc)}
        return {"check": checked, "deadlines": deadlines}

    @app.post("/api/ask")
    def ask(body: AskIn):
        c, prof, err = get_copilot(body.model_id)
        if c is None:
            return JSONResponse(status_code=503, content={
                "error": "The language model is not configured, so the copilot cannot answer questions yet. "
                         "Case lookup still works.", "detail": err})
        query = body.query.strip()
        if body.case_id and body.case_id.lower() not in query.lower():
            query = f"Case {body.case_id}: {query}"
        row = _eval_rows().get(body.eval_id) if body.eval_id else None
        res = c.ask(body.session_id, query, body.today)
        res["model"] = ({"id": prof.id, "label": prof.label, "model": prof.model} if prof else
                        {"id": "default", "label": c.agent.client.model, "model": c.agent.client.model})
        final = res.get("final") or {}
        try:
            res["evidence"] = evidence.resolve(final, query, getattr(c.agent, "search", None))
        except Exception as exc:
            res["evidence"] = {"citations": [], "source_overlap": None, "level": "n/a", "error": str(exc)}
        if row and res.get("llm_calls", 0) + int(res.get("cache_hit", False)) > 0:
            known = set(c.agent.known_sources) | {x["source_id"] for x in getattr(c.agent.search.ret, "chunks", [])}
            checks = scoring.score(row, final, known)
            res["reference"] = {"query_id": row["query_id"], "gold_answer": row["gold_answer"],
                                "expected_behavior": row["expected_behavior"],
                                "text": textmetrics.text_scores(str(final.get("answer", "")), row["gold_answer"]),
                                "checks": {k: checks[k] for k in ("behavior_ok", "include_all", "findings_ok",
                                                                  "citations_ok", "drafts_ok", "end_to_end")},
                                "must_not_violation": checks["must_not_violation"],
                                "failure_tags": checks["failure_tags"]}
        return res

    @app.post("/api/review")
    def review(body: ReviewIn):
        c = state["injected"] or next(iter(state["built"].values()), None) or get_copilot()[0]
        if c is None:
            raise HTTPException(503, "Copilot not configured")
        if body.decision not in _DECISIONS:
            raise HTTPException(422, f"decision must be one of {', '.join(_DECISIONS)}")
        try:
            return c.record_review(body.session_id, body.reviewer, body.decision, body.draft, body.note)
        except ValueError as exc:
            raise HTTPException(422, str(exc))

    return app


app = create_app()


def main() -> None:
    import uvicorn
    uvicorn.run("rto_copilot.api:app", host=os.environ.get("RTO_HOST", "127.0.0.1"),
                port=int(os.environ.get("RTO_PORT", "8000")))


if __name__ == "__main__":
    main()
