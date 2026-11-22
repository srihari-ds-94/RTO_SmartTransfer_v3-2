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
