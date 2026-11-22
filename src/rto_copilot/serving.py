"""Serving layer between the front-end and the Week 3 agent: input checks, PII redaction, rate limits, one-at-a-time
LLM access, an answer cache, and the human review log.

    copilot = Copilot.from_env()
    result = copilot.ask("session-1", "Check case DEMO-013.", "2026-10-20")

Order of checks for every message:
    1. input validation       empty or longer than MAX_QUERY_CHARS -> rejected, no LLM call
    2. PII redaction          Aadhaar, PAN, phone and e-mail replaced before anything is stored, cached or sent
    3. rate limits            per session (RATE_PER_MIN, burst RATE_BURST) and across all users (GLOBAL_PER_MIN)
    4. answer cache           same redacted question + date + session context -> cached answer, no LLM call
    5. endpoint gate          the course endpoint allows one active request per key, so one agent turn at a time;
                              waits up to GATE_WAIT_S, then answers "busy"
    6. agent turn             planner -> tools (retrieved chunks screened for injection) -> compose -> critic -> repair

Settings come from environment variables (see .env.example). LLM_* variables are read only from the environment.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import config, guardrails, llm, tracing
from . import baseline as rb
from .agent import Agent, Session

STATE = config.STATE


def _env_num(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


class TokenBucket:
    """capacity tokens, refilled at rate_per_min; one token per request."""

    def __init__(self, rate_per_min: float, capacity: float, clock=time.monotonic):
        self.rate, self.capacity, self.clock = rate_per_min / 60.0, capacity, clock
        self.tokens, self.last = capacity, clock()
        self.lock = threading.Lock()

    def take(self) -> tuple[bool, float]:
        """Returns (allowed, seconds to wait before the next token)."""
        with self.lock:
            now = self.clock()
            self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.rate)
            self.last = now
            if self.tokens >= 1:
                self.tokens -= 1
                return True, 0.0
            return False, round((1 - self.tokens) / self.rate, 1) if self.rate else float("inf")


class RateLimiter:
    def __init__(self, per_session_per_min: float, burst: float, global_per_min: float, clock=time.monotonic):
        self.per_session_per_min, self.burst, self.clock = per_session_per_min, burst, clock
        self.global_bucket = TokenBucket(global_per_min, max(global_per_min / 6, 1), clock)
        self.sessions: dict[str, TokenBucket] = {}
        self.lock = threading.Lock()

    def check(self, key: str) -> tuple[bool, str, float]:
        with self.lock:
            bucket = self.sessions.setdefault(key, TokenBucket(self.per_session_per_min, self.burst, self.clock))
        ok, wait = bucket.take()
        if not ok:
            return False, "session", wait
        ok, wait = self.global_bucket.take()
        if not ok:
            bucket.tokens = min(bucket.capacity, bucket.tokens + 1)  # refund: the request was not served
            return False, "global", wait
        return True, "", 0.0


class AnswerCache:
    """In-memory LRU with a time-to-live. Values are the final answer plus the session fields the turn set."""

    def __init__(self, max_items: int, ttl_s: float, clock=time.monotonic):
        self.max_items, self.ttl_s, self.clock = max_items, ttl_s, clock
        self.items: OrderedDict[str, tuple[float, dict]] = OrderedDict()
        self.lock = threading.Lock()
        self.hits = self.misses = 0

    @staticmethod
    def key(query: str, today: str | None, session: Session, model: str) -> str:
        ctx = [" ".join(query.lower().split()), today, session.active_case_id, session.role, session.transfer_date, model]
        return hashlib.sha256(json.dumps(ctx).encode()).hexdigest()

    def get(self, key: str) -> dict | None:
        with self.lock:
            item = self.items.get(key)
            if item and self.clock() - item[0] <= self.ttl_s:
                self.items.move_to_end(key)
                self.hits += 1
                return item[1]
            self.items.pop(key, None)
            self.misses += 1
            return None

    def put(self, key: str, value: dict) -> None:
        with self.lock:
            self.items[key] = (self.clock(), value)
            self.items.move_to_end(key)
            while len(self.items) > self.max_items:
                self.items.popitem(last=False)


@dataclass
class Settings:
    max_query_chars: int = 1000
    rate_per_min: float = 6
    rate_burst: float = 3
    global_per_min: float = 30
    cache_ttl_s: float = 3600
    cache_max_items: int = 256
    gate_wait_s: float = 90

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(int(_env_num("MAX_QUERY_CHARS", 1000)), _env_num("RATE_PER_MIN", 6), _env_num("RATE_BURST", 3),
                   _env_num("GLOBAL_PER_MIN", 30), _env_num("CACHE_TTL_S", 3600), int(_env_num("CACHE_MAX_ITEMS", 256)),
                   _env_num("GATE_WAIT_S", 90))


def refusal(behavior: str, answer: str) -> dict:
    return {"behavior": behavior, "answer": answer, "citations": [], "findings": [], "deadlines": [],
            "case_status": None, "clarifying_questions": [], "escalation": {"required": False, "reason": ""}, "drafts": []}


class Copilot:
    def __init__(self, agent: Agent, settings: Settings | None = None, state_dir: Path = STATE):
        self.agent = agent
        self.settings = settings or Settings()
        self.state_dir = Path(state_dir)
        self.limiter = RateLimiter(self.settings.rate_per_min, self.settings.rate_burst, self.settings.global_per_min)
        self.cache = AnswerCache(self.settings.cache_max_items, self.settings.cache_ttl_s)
        self.gate = threading.Semaphore(1)
        self.review_log = self.state_dir / "review_log.jsonl"

    @classmethod
    def from_env(cls) -> "Copilot":
        from . import tools as T
        missing = config.missing_llm_env()
        if missing:
            raise RuntimeError(f"Set {', '.join(missing)}.")
        client = rb.EndpointClient(os.environ["LLM_BASE_URL"], os.environ["LLM_API_KEY"], os.environ["LLM_MODEL"],
                                   False, 0.0, os.environ.get("LLM_CA_CERT") or None, False)
        STATE.mkdir(parents=True, exist_ok=True)
        tracing.setup(STATE / "traces.jsonl")
        return cls(Agent(client, T.CorpusSearch(), llm_critic=False), Settings.from_env())

    def ask(self, session_id: str, query: str, today: str | None = None) -> dict:
        started = time.time()
        out = {"final": None, "notices": [], "pii_redacted": [], "cache_hit": False, "rate_limited": None,
               "quarantined": [], "tools": [], "llm_calls": 0, "tokens": 0, "stop_reason": None, "latency_s": 0.0}
        query = (query or "").strip()
        if not query:
            out["final"] = refusal("ASK_CLARIFICATION", "Please type a question about your vehicle transfer.")
            return out
        if len(query) > self.settings.max_query_chars:
            out["final"] = refusal("ASK_CLARIFICATION", f"Please keep the question under {self.settings.max_query_chars} "
                                                        "characters.")
            return out
        query, pii = guardrails.redact_pii(query)
        if pii:
            out["pii_redacted"] = pii
            out["notices"].append(f"Removed {', '.join(p.lower() for p in pii)} from your message before processing. "
                                  "Please do not share real identity numbers or contact details; this copilot works on "
                                  "synthetic case data and does not store them.")
        ok, scope, wait = self.limiter.check(session_id)
        if not ok:
            out["rate_limited"] = scope
            who = "this session" if scope == "session" else "all users"
            out["final"] = refusal("ASK_CLARIFICATION", f"Too many requests from {who}. Please wait about {wait:.0f} s "
                                                        "and try again.")
            return out

        session = Session.load(session_id, self.state_dir / "sessions")
        key = AnswerCache.key(query, today, session, self.agent.client.model)
        cached = self.cache.get(key)
        if cached:
            final = json.loads(json.dumps(cached["final"]))
            session.active_case_id, session.role, session.transfer_date = cached["session_after"]
            out.update(cache_hit=True, quarantined=cached["quarantined"], tools=cached["tools"])
        else:
            if not self.gate.acquire(timeout=self.settings.gate_wait_s):
                out["final"] = refusal("ASK_CLARIFICATION", "The copilot is busy with another request. Please try again "
                                                            "in a minute.")
                out["rate_limited"] = "busy"
                return out
            try:
                final, tr = self.agent.run(query, today, session)
            finally:
                self.gate.release()
            session.turns.pop()  # re-added below so cached and live turns are recorded the same way
            tools = [c["tool"] for c in tr.tool_calls]
            out.update(quarantined=tr.quarantined, tools=tools, llm_calls=tr.llm_calls, tokens=tr.tokens,
                       stop_reason=tr.stop_reason)
            if tr.completed and not tr.quarantined:
                self.cache.put(key, {"final": final, "quarantined": tr.quarantined, "tools": tools,
                                     "session_after": [session.active_case_id, session.role, session.transfer_date]})
        if out["quarantined"]:
            out["notices"].append("Some retrieved text looked like hidden instructions and was ignored: "
                                  + ", ".join(sorted({q["source_id"] for q in out["quarantined"]})) + ".")
        session.turns.append({"question": query, "today": today, "behavior": final.get("behavior"),
                              "answer": str(final.get("answer", ""))[:400], "case_id": session.active_case_id})
        session.turns = session.turns[-20:]
        session.save(self.state_dir / "sessions")
        out["final"] = final
        out["latency_s"] = round(time.time() - started, 2)
        return out

    def record_review(self, session_id: str, reviewer: str, decision: str, draft: dict, note: str = "") -> dict:
        """A human reviewer's decision on a draft. This only records the decision; nothing is sent or filed."""
        if decision not in ("approved_for_manual_use", "rejected", "needs_changes"):
            raise ValueError("decision must be approved_for_manual_use, rejected or needs_changes")
        if not reviewer.strip():
            raise ValueError("reviewer name is required")
        rec = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "session": session_id,
               "reviewer": guardrails.redact_pii(reviewer.strip())[0], "decision": decision,
               "draft_type": draft.get("type"), "draft_sha256": hashlib.sha256(str(draft.get("content")).encode()).hexdigest(),
               "note": guardrails.redact_pii(note)[0], "sent": False}
        self.review_log.parent.mkdir(parents=True, exist_ok=True)
        with self.review_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec
