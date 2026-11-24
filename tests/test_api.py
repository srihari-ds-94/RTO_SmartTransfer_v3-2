"""Web layer tests: FastAPI TestClient with a scripted model (no network)."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from rto_copilot import serving
from rto_copilot.api import create_app
from test_agent import ScriptedClient, answer, make, plan


class Web(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def client(self, copilot=None):
        return TestClient(create_app(copilot))

    def copilot(self, **kw):
        agent = make(ScriptedClient([plan()], [answer(**kw)]))
        return serving.Copilot(agent, serving.Settings(rate_per_min=60, rate_burst=10), Path(self.tmp.name))

    def test_page_and_cases(self):
        c = self.client(self.copilot())
        page = c.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn("RTO SmartTransfer Copilot", page.text)
        cases = c.get("/api/cases").json()
        self.assertEqual(len(cases), 148)
        self.assertEqual(cases[0]["case_id"], "DEMO-001")

    def test_case_view_needs_no_model(self):
        c = self.client(self.copilot())
        d = c.get("/api/case/DEMO-009?today=2026-11-10").json()
        self.assertEqual([f["code"] for f in d["check"]["findings"]], ["ENGINE_MISMATCH"])
        self.assertEqual(d["check"]["case_status"], "REVIEW_REQUIRED")
        self.assertTrue(d["deadlines"]["deadlines"])
        self.assertEqual(c.get("/api/case/DEMO-999").status_code, 404)

    def test_ask_and_review_log(self):
        draft = {"type": "message", "content": "Dear X, please send Form 30.", "sent": False}
        cp = self.copilot(answer="This is a draft and has not been sent.", drafts=[draft])
        c = self.client(cp)
        res = c.post("/api/ask", json={"session_id": "s1", "query": "Draft a message to the buyer", "case_id": "DEMO-013"})
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual(body["final"]["drafts"][0]["sent"], False)
        rv = c.post("/api/review", json={"session_id": "s1", "reviewer": "Asha", "decision": "approved_for_manual_use",
                                         "draft": draft})
        self.assertEqual(rv.status_code, 200)
        self.assertFalse(rv.json()["sent"])
        line = json.loads(Path(self.tmp.name, "review_log.jsonl").read_text().splitlines()[-1])
        self.assertFalse(line["sent"])
        bad = c.post("/api/review", json={"session_id": "s1", "reviewer": "Asha", "decision": "send_it", "draft": draft})
        self.assertEqual(bad.status_code, 422)

    def test_ask_without_model_is_a_clear_503(self):
        import os
        for k in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
            os.environ.pop(k, None)
        import rto_copilot.config as cfg
        keep = cfg.load_dotenv
        cfg.load_dotenv = lambda *a, **k: None
        try:
            c = self.client(None)
            r = c.post("/api/ask", json={"session_id": "s", "query": "hello"})
            self.assertEqual(r.status_code, 503)
            self.assertIn("not configured", r.json()["error"])
            self.assertFalse(c.get("/api/health").json()["llm_ready"])
        finally:
            cfg.load_dotenv = keep


if __name__ == "__main__":
    unittest.main()


class Config(unittest.TestCase):
    def test_relative_cert_path_is_resolved_against_project_root(self):
        import os
        from rto_copilot import config
        with tempfile.TemporaryDirectory() as d:
            env = Path(d) / ".env"
            env.write_text("LLM_CA_CERT=certs/vm07-ca.crt\n")
            os.environ.pop("LLM_CA_CERT", None)
            self.addCleanup(os.environ.pop, "LLM_CA_CERT", None)
            config.load_dotenv(env)
            self.assertEqual(os.environ["LLM_CA_CERT"], str(config.ROOT / "certs" / "vm07-ca.crt"))
            self.assertTrue(Path(os.environ["LLM_CA_CERT"]).exists())

    def test_health_reports_missing_env(self):
        c = TestClient(create_app())
        h = c.get("/api/health").json()
        self.assertIn("missing_env", h)
        self.assertIn("ca_cert_found", h)


class FakeSearch:
    """Like tools.CorpusSearch: callable, plus get_source and a .ret.chunks list."""
    class ret:  # noqa: N801
        chunks = [{"source_id": "SRC-MVA-S50"}]

    def __call__(self, query, case_id=None, k=None):
        from test_agent import fake_search
        return fake_search(query, case_id, k)

    def get_source(self, source_id, query=None):
        return {"results": [{"source_id": source_id, "text":
                             "the transferor shall within fourteen days of the transfer report the fact of transfer"}]}


class Evidence(unittest.TestCase):
    def app(self, **kw):
        from rto_copilot.agent import Agent, Limits
        agent = Agent(ScriptedClient([plan()], [answer(**kw)]), FakeSearch(), Limits(), exemplars=False)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        cp = serving.Copilot(agent, serving.Settings(rate_per_min=60, rate_burst=10), Path(tmp.name))
        return TestClient(create_app(cp))

    def test_citations_are_resolved_with_title_and_passage(self):
        c = self.app(answer="The seller reports the transfer within fourteen days.")
        r = c.post("/api/ask", json={"session_id": "s1", "query": "How long does the seller have?"}).json()
        ev = r["evidence"]
        self.assertEqual(ev["citations"][0]["source_id"], "SRC-MVA-S50")
        self.assertTrue(ev["citations"][0]["in_registry"])
        self.assertIn("fourteen days", ev["citations"][0]["snippet"])
        self.assertGreaterEqual(ev["source_overlap"], 0.55)
        self.assertEqual(ev["level"], "high")

    def test_unknown_source_is_flagged_and_missing_citation_is_called_out(self):
        from rto_copilot import evidence
        fake = {"behavior": "ANSWER", "answer": "x", "citations": [{"source_id": "SRC-MADE-UP", "locator": "x"}]}
        ev = evidence.resolve(fake, "q", FakeSearch())
        self.assertFalse(ev["citations"][0]["in_registry"])
        self.assertEqual(ev["level"], "low")
        self.assertEqual(evidence.resolve({"behavior": "ANSWER", "answer": "x", "citations": []}, "q")["level"], "none")
        self.assertEqual(evidence.resolve({"behavior": "REFUSE_UNSAFE", "answer": "no"}, "q")["level"], "n/a")
        # the agent already strips unknown ids before the UI sees them; this check is the second line of defence
        c = self.app(citations=[{"source_id": "SRC-MADE-UP", "locator": "x"}])
        r = c.post("/api/ask", json={"session_id": "s2", "query": "q"}).json()
        self.assertNotIn("SRC-MADE-UP", [x["source_id"] for x in r["evidence"]["citations"]])

    def test_eval_question_returns_bleu_rouge_and_checks(self):
        c = self.app()
        rows = c.get("/api/evalset").json()
        self.assertTrue(rows and all(r["query_id"] for r in rows))
        r = c.post("/api/ask", json={"session_id": "s4", "query": rows[0]["query"], "eval_id": rows[0]["query_id"]}).json()
        ref = r["reference"]
        self.assertEqual(set(ref["text"]), {"bleu", "rouge1", "rouge2", "rougeL"})
        self.assertIn("end_to_end", ref["checks"])

    def test_test_split_is_not_offered(self):
        c = self.app()
        ids = {r["query_id"] for r in c.get("/api/evalset").json()}
        import csv
        from rto_copilot import config
        test_ids = {r["query_id"] for r in csv.DictReader(open(config.EVAL, encoding="utf-8-sig")) if r["split"] == "test"}
        self.assertFalse(ids & test_ids)


class TextMetrics(unittest.TestCase):
    def test_known_values(self):
        from rto_copilot import textmetrics as T
        self.assertEqual(T.bleu("a b c d e", "a b c d e"), 1.0)
        self.assertEqual(T.rouge_l("x y z", "a b c"), 0.0)
        self.assertAlmostEqual(T.rouge_n("the cat sat", "the cat ran", 1), 2 / 3, places=4)
        self.assertAlmostEqual(T.rouge_l("the cat sat on mat", "the cat on mat"), 2 * 1 * 0.8 / 1.8, places=4)


class Models(unittest.TestCase):
    def setUp(self):
        import os
        self.keep = dict(os.environ)
        for k in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL", "LLM_CA_CERT", "HF_TOKEN", "MODELS_FILE"):
            os.environ.pop(k, None)
        self.addCleanup(lambda: (os.environ.clear(), os.environ.update(self.keep)))

    def test_readiness_reasons(self):
        import os
        from rto_copilot import models
        ps = {p.id: p for p in models.load_profiles()}
        self.assertEqual(set(ps), {"qwen3-8b", "llama31-8b-local", "hf-router"})
        self.assertIn("LLM_API_KEY", ps["qwen3-8b"].ready()[1])
        self.assertIn("HF_TOKEN", ps["hf-router"].ready()[1])
        os.environ.update(LLM_BASE_URL="https://x/v1", LLM_API_KEY="k", LLM_MODEL="Qwen/Qwen3-8B", HF_TOKEN="hf_x")
        ps = {p.id: p for p in models.load_profiles()}
        self.assertTrue(ps["qwen3-8b"].ready()[0])
        self.assertTrue(ps["hf-router"].ready()[0])
        os.environ["LLM_CA_CERT"] = "certs/does-not-exist.crt"
        self.assertIn("certificate", {p.id: p for p in models.load_profiles()}["qwen3-8b"].ready()[1])

    def test_models_json_adds_a_profile_and_qwen_kwargs_only_for_course(self):
        import os
        from rto_copilot import models
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "m.json"
            f.write_text(json.dumps([{"id": "x1", "label": "X", "base_url": "http://localhost:9/v1", "model": "m"}]))
            os.environ["MODELS_FILE"] = str(f)
            self.assertIn("x1", [p.id for p in models.load_profiles()])
        os.environ.update(LLM_BASE_URL="https://x/v1", LLM_API_KEY="k", LLM_MODEL="Qwen/Qwen3-8B")
        ps = {p.id: p for p in models.load_profiles()}
        self.assertTrue(ps["qwen3-8b"].client().template_kwargs)
        self.assertFalse(ps["hf-router"].client().template_kwargs)

    def test_api_models_and_unknown_model(self):
        c = TestClient(create_app())
        r = c.get("/api/models").json()
        self.assertIsNone(r["default"])
        self.assertEqual(len(r["models"]), 3)
        bad = c.post("/api/ask", json={"session_id": "s", "query": "hi", "model_id": "nope"})
        self.assertEqual(bad.status_code, 503)
        self.assertIn("Unknown model", bad.json()["detail"])

    def test_injected_copilot_reports_its_model(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        agent = make(ScriptedClient([plan()], [answer()]))
        cp = serving.Copilot(agent, serving.Settings(rate_per_min=60, rate_burst=10), Path(tmp.name))
        c = TestClient(create_app(cp))
        self.assertEqual(c.get("/api/models").json()["default"], "default")
        r = c.post("/api/ask", json={"session_id": "m1", "query": "hello"}).json()
        self.assertEqual(r["model"]["id"], "default")


class Dashboard(unittest.TestCase):
    def test_saved_runs_are_structured_and_html_is_self_contained(self):
        from rto_copilot import dashboard
        runs = [dashboard.load_run(p) for p in dashboard.default_runs()]
        self.assertGreaterEqual(len(runs), 4)
        dev = next(r for r in runs if r["id"] == "w3_dev_final")
        self.assertEqual(len(dev["traces"]), 57)
        t = dev["traces"][0]
        self.assertTrue(t["spans"][0]["n"] == "agent.turn" or any(s["n"] == "agent.turn" for s in t["spans"]))
        self.assertIn(t["passed"], (True, False))
        html = dashboard.build_html(runs)
        self.assertNotIn("__DATA__", html)
        self.assertNotIn("</script><script", html.replace("</script>\n</body>", ""))

    def test_live_endpoint_reads_the_trace_file(self):
        import os
        from rto_copilot import config, dashboard, tracing
        with tempfile.TemporaryDirectory() as d:
            keep = config.STATE
            config.STATE = Path(d)
            try:
                tracing.setup(Path(d) / "traces.jsonl")
                with tracing.span("agent.turn", session="s", query_id=None, query="What is Form 29?", behavior="ANSWER",
                                  llm_calls=1, tool_calls=0, tokens=10, stop_reason="done", completed=True, repaired="[]"):
                    with tracing.span("planner"):
                        pass
                c = TestClient(create_app())
                r = c.get("/api/traces/live").json()
                self.assertEqual(r["id"], "live")
                self.assertEqual(r["traces"][0]["query"], "What is Form 29?")
                self.assertEqual(c.get("/dashboard").status_code, 200)
            finally:
                config.STATE = keep
                tracing.setup(None)
