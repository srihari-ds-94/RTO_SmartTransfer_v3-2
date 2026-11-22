"""Talk to the agent turn by turn; state persists in state/sessions/<session>.json between runs.

    rto-chat --session s1 --today 2026-10-20 "Check case DEMO-013."
    rto-chat --session s1 --today 2026-10-20 "When is my deadline as the buyer?"
    rto-chat --demo                         # scripted three-turn conversation, saved to results/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

from . import config, tracing
from . import tools as T
from .agent import Agent, Session, rb

DEMO = [("Check case DEMO-013.", "2026-10-20"),
        ("I am the buyer. When is my deadline?", "2026-10-20"),
        ("Draft a message to the buyer asking for the missing item.", "2026-10-20")]


def client():
    config.load_dotenv()
    missing = config.missing_llm_env()
    if missing:
        sys.exit(f"Set {', '.join(missing)}.")
    return rb.EndpointClient(os.environ["LLM_BASE_URL"], os.environ["LLM_API_KEY"], os.environ["LLM_MODEL"],
                             False, 1.0, os.environ.get("LLM_CA_CERT"), False)


def show(q, final, tr):
    print(f"\nUSER: {q}\nCOPILOT [{final.get('behavior')}]: {final.get('answer')}")
    for d in final.get("deadlines") or []:
        print(f"  deadline {d.get('rule_id')}: {d.get('due_date')} ({d.get('status')})")
    for d in final.get("drafts") or []:
        print(f"  DRAFT (sent={d.get('sent')}): {str(d.get('content'))[:300]}")
    print(f"  tools: {[c['tool'] for c in tr.tool_calls]} | llm calls {tr.llm_calls} | tokens {tr.tokens} | {tr.stop_reason}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?")
    ap.add_argument("--session", default="default")
    ap.add_argument("--today")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--llm-critic", action="store_true", help="add the LLM review to the rule critic (off in the final config)")
    args = ap.parse_args()

    agent = Agent(client(), T.CorpusSearch(), llm_critic=args.llm_critic)
    if args.demo:
        out = config.RESULTS / "demo" / f"multiturn_{datetime.now():%Y%m%d_%H%M%S}"
        out.mkdir(parents=True)
        tracing.setup(out / "traces.jsonl")
        sid = "demo-multiturn"
        transcript = []
        for q, today in DEMO:
            session = Session.load(sid, out / "sessions")   # reload from disk every turn
            final, tr = agent.run(q, today, session)
            session.save(out / "sessions")
            show(q, final, tr)
            transcript.append({"question": q, "today": today, "state_before": {"active_case_id": session.turns[-1]["case_id"]},
                               "answer": final, "tools": [c["tool"] for c in tr.tool_calls], "stop_reason": tr.stop_reason})
        (out / "transcript.json").write_text(json.dumps(transcript, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nSaved {out}")
        return 0
    if not args.question:
        ap.error("give a question or --demo")
    tracing.setup(config.STATE / "traces.jsonl")
    session = Session.load(args.session)
    final, tr = agent.run(args.question, args.today, session)
    session.save()
    show(args.question, final, tr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
