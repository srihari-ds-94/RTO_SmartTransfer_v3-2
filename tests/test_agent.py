"""Offline tests for the agent's control flow and error handling (scripted LLM, no network, no retriever).

    pytest tests/test_agent.py -v
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rto_copilot import tools as T
from rto_copilot.agent import Agent, Limits, Session

FAKE_CHUNK = {"source_id": "SRC-MVA-S50", "title": "Motor Vehicles Act, 1988, section 50", "locator": "s.50",
              "text": "the transferor shall within fourteen days of the transfer report the fact of transfer"}


def fake_search(query, case_id=None, k=None):
    return {"query": query, "results": [FAKE_CHUNK]}


class ScriptedClient:
    """Returns scripted replies per role; a callable reply receives the messages."""
    model = "scripted"

    def __init__(self, planner, executor, critic=None):
        self.replies = {"planner": list(planner), "executor": list(executor), "critic": list(critic or [])}
        self.calls = {"planner": 0, "executor": 0, "critic": 0}

    def complete(self, messages, temperature, row, sample=0):
        sys = messages[0]["content"]
        role = "planner" if sys.startswith("You are the planner") else "critic" if sys.startswith("You are the critic") else "executor"
        self.calls[role] += 1
        queue = self.replies[role]
        reply = queue.pop(0) if len(queue) > 1 else (queue[0] if queue else '{"verdict": "pass", "issues": []}')
        text = reply(messages) if callable(reply) else reply if isinstance(reply, str) else json.dumps(reply)
        return {"text": text, "latency_s": 0.0, "prompt_tokens": 100, "completion_tokens": 50}


def answer(**kw):
    base = {"behavior": "ANSWER", "answer": "ok", "citations": [{"source_id": "SRC-MVA-S50", "locator": "s.50"}],
            "findings": [], "deadlines": [], "case_status": None, "clarifying_questions": [],
            "escalation": {"required": False, "reason": ""}, "drafts": []}
    base.update(kw)
    return base


def plan(**kw):
    base = {"behavior": "ANSWER", "reason": "test", "role": "seller", "case_id": None, "transfer_date": None,
            "steps": [{"tool": "search_corpus", "args": {"query": "seller report"}}]}
    base.update(kw)
    return base


def make(client, **kw):
    return Agent(client, fake_search, kw.pop("limits", Limits()), exemplars=False, **kw)


class DeterministicTools(unittest.TestCase):
    def test_deadlines_with_conflict(self):
        d = T.compute_deadlines("2026-10-20", "2026-11-10")
        dates = {x["rule_id"]: (x["due_date"], x["status"]) for x in d["deadlines"]}
        self.assertEqual(dates["R-SELLER-REPORT-SAME-STATE"], ("2026-11-03", "overdue"))
        self.assertEqual(dates["R-BUYER-REPORT-ACT"], ("2026-11-19", "conflict"))
        self.assertEqual(d["conflicts"][0]["plan_to"], "2026-11-03")

    def test_relative_and_future_dates(self):
        self.assertEqual(T.resolve_date("we sold it yesterday", "2026-11-10")["date"], "2026-11-09")
        self.assertEqual(T.resolve_date("3 days ago", "2026-11-10")["date"], "2026-11-07")
        with self.assertRaises(T.ToolError):
            T.resolve_date("2026-12-01", "2026-11-10")
        with self.assertRaises(T.ToolError):
            T.resolve_date("last week", "2026-11-10")

    def test_case_checks(self):
        self.assertEqual([f["code"] for f in T.check_case("DEMO-009")["findings"]], ["ENGINE_MISMATCH"])
        self.assertTrue(T.check_case("DEMO-069")["escalation_triggers"])
        self.assertFalse(T.check_case("DEMO-150")["found"])
        with self.assertRaises(T.ToolError):
            T.check_case("case 9")

    def test_loan_procedure(self):
        one = T.loan_procedure("LOAN_TERMINATED")["scenarios"]
        self.assertEqual([s["scenario"] for s in one], ["LOAN_TERMINATED"])
        self.assertIn("Form 35", one[0]["path"])
        self.assertEqual(len(T.loan_procedure("all")["scenarios"]), 4)
        with self.assertRaises(T.ToolError):
            T.loan_procedure("REPOSSESSION")


class ErrorHandling(unittest.TestCase):
    def test_malformed_json_is_retried_then_falls_back(self):
        client = ScriptedClient([plan()], ["not json at all", "still not json"])
        final, tr = make(client, llm_critic=False).run("When must the seller report?", "2026-11-10")
        self.assertGreaterEqual(sum(e["error"] == "malformed_json" for e in tr.errors), 2)
        self.assertLessEqual(tr.llm_calls, Limits().max_llm_calls)
        self.assertIn("FALLBACK_ANSWER", tr.repaired)
        self.assertEqual(final["behavior"], "ESCALATE_HUMAN_REVIEW")
        self.assertFalse(tr.completed)

    def test_malformed_planner_uses_default_plan(self):
        client = ScriptedClient(["{broken"], [answer()])
        final, tr = make(client, llm_critic=False).run("Which forms do we need?", None)
        self.assertTrue(any(e["error"] == "planner_fallback" for e in tr.errors))
        self.assertTrue(tr.completed)

    def test_tool_failures_are_caught_and_retried(self):
        client = ScriptedClient([plan()], [answer(citations=[])])
        final, tr = make(client, llm_critic=False, fault_rate=1.0).run("When must the seller report?", None)
        rec = tr.tool_calls[0]
        self.assertTrue(rec["error"].startswith("tool failed"))
        self.assertEqual(rec["attempts"], 2)
        self.assertIsNotNone(final["behavior"])

    def test_invalid_tool_steps_are_dropped(self):
        p = plan(steps=[{"tool": "delete_database", "args": {}}, "garbage", {"tool": "search_corpus", "args": {}}])
        client = ScriptedClient([p], [answer()])
        _, tr = make(client, llm_critic=False).run("Which forms?", None)
        self.assertEqual([c["tool"] for c in tr.tool_calls], ["search_corpus"])
        self.assertEqual(sum("invalid tool step" in e["error"] for e in tr.errors), 2)

    def test_duplicate_calls_and_tool_cap(self):
        steps = [{"tool": "search_corpus", "args": {"query": "same"}}] * 3 + \
                [{"tool": "search_corpus", "args": {"query": f"q{i}"}} for i in range(10)]
        client = ScriptedClient([plan(steps=steps)], [answer()])
        _, tr = make(client, llm_critic=False, limits=Limits(max_tool_calls=4)).run("Which forms?", None)
        self.assertEqual(len([c for c in tr.tool_calls if not c.get("skipped")]), 4)
        self.assertTrue(any("duplicate" in e["error"] for e in tr.errors))
        self.assertTrue(any("cap" in e["error"] for e in tr.errors))

    def test_critic_loop_is_bounded_and_repairs(self):
        p = plan(case_id="DEMO-009", steps=[{"tool": "check_case", "args": {"case_id": "DEMO-009"}}])
        client = ScriptedClient([p], [answer(findings=[])])  # never fixes the findings
        final, tr = make(client, llm_critic=False).run("Verify the documents for case DEMO-009.", None)
        self.assertEqual(len(tr.critic_rounds), Limits().max_critic_rounds + 1)
        self.assertEqual(tr.stop_reason, "critic_unresolved")
        self.assertEqual([f["code"] for f in final["findings"]], ["ENGINE_MISMATCH"])
        self.assertEqual(final["behavior"], "ESCALATE_HUMAN_REVIEW")

    def test_critic_runs_missing_tool(self):
        p = plan(case_id="DEMO-013", steps=[{"tool": "search_corpus", "args": {"query": "Form 30"}}])  # no check_case
        fixed = answer(findings=[{"code": "MISSING_FORM30", "evidence": ""}], case_status="REVIEW_REQUIRED",
                       citations=[{"source_id": "SRC-MVA-S50", "locator": ""},
                                  {"source_id": "SRC-PACK-DATASET", "locator": "DEMO-013"}])
        client = ScriptedClient([p], [answer(), fixed])
        final, tr = make(client, llm_critic=False).run("Is anything missing for case DEMO-013?", None)
        self.assertIn("check_case", [c["tool"] for c in tr.tool_calls])
        self.assertEqual(tr.stop_reason, "done")

    def test_critic_runs_loan_procedure(self):
        client = ScriptedClient([plan()], [answer()])
        _, tr = make(client, llm_critic=False).run("The bank still holds hypothecation. Can we transfer?", None)
        self.assertIn("loan_procedure", [c["tool"] for c in tr.tool_calls])

    def test_draft_and_escalation_wording_repaired(self):
        draft = answer(answer="Dear buyer, please send Form 30.",
                       drafts=[{"type": "message", "content": "Please send Form 30.", "sent": True}])
        final, tr = make(ScriptedClient([plan()], [draft]), llm_critic=False).run("Draft a message to the buyer.", None)
        self.assertIn("draft", final["answer"].lower())
        self.assertFalse(final["drafts"][0]["sent"])
        esc = answer(behavior="ESCALATE_HUMAN_REVIEW", answer="Something looks wrong.", citations=[])
        final, _ = make(ScriptedClient([plan()], [esc]), llm_critic=False).run("Is this RC forged?", None)
        self.assertIn("human reviewer", final["answer"])

    def test_token_budget_stops_the_task(self):
        client = ScriptedClient([plan()], [answer()])
        _, tr = make(client, llm_critic=False, limits=Limits(task_token_budget=1500)).run("Which forms?", None)
        self.assertEqual(tr.stop_reason, "budget")
        self.assertFalse(tr.completed)

    def test_llm_call_cap(self):
        client = ScriptedClient([plan()], ["bad"] * 10)
        _, tr = make(client, llm_critic=False, limits=Limits(max_llm_calls=2)).run("Which forms?", None)
        self.assertLessEqual(tr.llm_calls, 2)
        self.assertIn(tr.stop_reason, ("budget", "done"))

    def test_dropped_connection_is_retried(self):
        client = ScriptedClient([plan()], [answer()])
        real = client.complete
        state = {"failed": False}
        def flaky(messages, *a, **k):
            if not state["failed"]:
                state["failed"] = True
                raise ConnectionError("Remote end closed connection without response")
            return real(messages, *a, **k)
        client.complete = flaky
        from rto_copilot import agent as A
        A.time.sleep, sleep = (lambda s: None), A.time.sleep
        try:
            final, tr = make(client, llm_critic=False).run("Which forms?", None)
        finally:
            A.time.sleep = sleep
        self.assertTrue(any(e["error"].startswith("transient") for e in tr.errors))
        self.assertEqual(tr.stop_reason, "done")

    def test_context_length_rejection_trims_and_retries(self):
        client = ScriptedClient([plan()], [answer()])
        real, state = client.complete, {"rejected": False}
        def strict(messages, *a, **k):
            if not messages[0]["content"].startswith("You are the planner") and not state["rejected"]:
                state["rejected"] = True
                raise RuntimeError("HTTP 400: This model's maximum context length is 8192 tokens. However, your "
                                   "prompt contains at least 7300 input tokens")
            return real(messages, *a, **k)
        client.complete = strict
        agent = make(client, llm_critic=False)
        final, tr = agent.run("Which forms?", None)
        self.assertTrue(any(e["error"] == "context_length_rejected" for e in tr.errors))
        self.assertEqual(tr.stop_reason, "done")
        self.assertGreater(agent.token_scale, 1.0)
        case_plan = plan(case_id="DEMO-013", steps=[{"tool": "check_case", "args": {"case_id": "DEMO-013"}},
                                                    {"tool": "loan_procedure", "args": {"scenario": "all"}}])
        _, tr = make(ScriptedClient([case_plan], [answer()]), llm_critic=False).run("Check case DEMO-013.", None)
        self.assertNotIn("loan_procedure", [c["tool"] for c in tr.tool_calls])
        loan_plan = plan(steps=[{"tool": "loan_procedure", "args": {"scenario": "LOAN_TERMINATED"}}])
        _, tr = make(ScriptedClient([loan_plan], [answer()]), llm_critic=False).run("Loan repaid, what now?", None)
        self.assertEqual([c["tool"] for c in tr.tool_calls][:2], ["loan_procedure", "search_corpus"])

    def test_llm_critic_revise_then_pass(self):
        client = ScriptedClient([plan()], [answer(), answer(answer="revised")],
                                [{"verdict": "revise", "issues": ["unsupported claim"]}, {"verdict": "pass", "issues": []}])
        final, tr = make(client).run("When must the seller report?", None)
        self.assertEqual(final["answer"], "revised")
        self.assertEqual(client.calls["critic"], 2)


class PersistentState(unittest.TestCase):
    def test_follow_up_uses_saved_case(self):
        with tempfile.TemporaryDirectory() as d:
            p1 = plan(case_id="DEMO-013", steps=[{"tool": "check_case", "args": {"case_id": "DEMO-013"}}])
            client = ScriptedClient([p1], [answer(findings=[{"code": "MISSING_FORM30", "evidence": ""}],
                                                  case_status="REVIEW_REQUIRED")])
            s = Session(session_id="t1")
            make(client, llm_critic=False).run("Check case DEMO-013.", "2026-10-20", s)
            s.save(Path(d))

            s2 = Session.load("t1", Path(d))
            self.assertEqual(s2.active_case_id, "DEMO-013")
            seen = {}
            def planner(messages):
                seen["state"] = messages[-1]["content"]
                return json.dumps(plan(case_id=None, role="buyer",
                                       steps=[{"tool": "compute_deadlines", "args": {"transfer_date": "case", "role": "buyer"}}]))
            client2 = ScriptedClient([planner], [lambda m: json.dumps(answer(behavior="ANSWER_WITH_CAVEAT",
                                     findings=[{"code": "MISSING_FORM30", "evidence": ""}], case_status="REVIEW_REQUIRED",
                                     deadlines=[], answer="x"))])
            _, tr = make(client2, llm_critic=False).run("And when is my deadline as the buyer?", "2026-10-20", s2)
            self.assertIn("DEMO-013", seen["state"])
            self.assertIn("compute_deadlines", [c["tool"] for c in tr.tool_calls])
            self.assertEqual(len(s2.turns), 2)
            self.assertEqual(s2.role, "buyer")


if __name__ == "__main__":
    unittest.main()
