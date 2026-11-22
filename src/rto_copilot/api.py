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

from . import config, tools

_DECISIONS = ("approved_for_manual_use", "rejected", "needs_changes")


class AskIn(BaseModel):
    session_id: str = Field(min_length=1, max_length=64)
    query: str = Field(max_length=2000)
    case_id: str | None = None
    today: str | None = None


class ReviewIn(BaseModel):
    session_id: str = Field(min_length=1, max_length=64)
    reviewer: str = Field(min_length=1, max_length=80)
    decision: str
    draft: dict
    note: str = Field(default="", max_length=500)


def _cases() -> list[dict]:
    with config.CASES.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def create_app(copilot=None) -> FastAPI:
    """`copilot` can be injected (tests, notebooks). Otherwise it is built lazily from the environment."""
    app = FastAPI(title="RTO SmartTransfer Copilot", version="3.0.0", docs_url=None, redoc_url=None)
    state = {"copilot": copilot, "error": None}
    config.load_dotenv()

    def get_copilot():
        if state["copilot"] is None and state["error"] is None:
            config.load_dotenv()
            try:
                from .serving import Copilot
                state["copilot"] = Copilot.from_env()
            except Exception as exc:  # missing env vars or unreachable setup
                state["error"] = str(exc)
        return state["copilot"]

    @app.get("/")
    def index():
        return FileResponse(config.WEB / "index.html")

    @app.get("/api/health")
    def health():
        c = get_copilot()
        cert = os.environ.get("LLM_CA_CERT", "")
        return {"ok": True, "llm_ready": c is not None, "llm_error": state["error"], "cases": len(_cases()),
                "eval_set": config.EVAL_VERSION, "missing_env": config.missing_llm_env(),
                "ca_cert": cert or None, "ca_cert_found": bool(cert) and Path(cert).exists()}

    @app.get("/api/cases")
    def cases():
        return [{"case_id": c["case_id"], "scenario": c["scenario"], "seller": c["seller_name"],
                 "buyer": c["buyer_name"], "sale_date": c["sale_date"]} for c in _cases()]

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
        c = get_copilot()
        if c is None:
            return JSONResponse(status_code=503, content={
                "error": "The language model is not configured, so the copilot cannot answer questions yet. "
                         "Case lookup still works.", "detail": state["error"]})
        query = body.query.strip()
        if body.case_id and body.case_id.lower() not in query.lower():
            query = f"Case {body.case_id}: {query}"
        return c.ask(body.session_id, query, body.today)

    @app.post("/api/review")
    def review(body: ReviewIn):
        c = get_copilot()
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
