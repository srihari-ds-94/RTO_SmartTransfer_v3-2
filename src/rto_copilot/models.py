"""The models the copilot can talk to. Every profile speaks the OpenAI chat-completions dialect.

    qwen3-8b          the course endpoint (LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, LLM_CA_CERT)
    llama31-8b-local  an open model from Hugging Face / Ollama running on this laptop (free, no key)
    hf-router         any chat model on the Hugging Face Inference Providers router (needs HF_TOKEN; free credit is small)

Add or change profiles in ``models.json`` (see docs/MODELS.md). A profile is "ready" when its key is present
(and, for the local one, when Ollama is running and has the model).
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from . import config
from . import llm

_PROBE_TTL_S = 8.0
_probe_cache: dict[str, tuple[float, tuple[bool, str]]] = {}


@dataclass
class Profile:
    id: str
    label: str
    base_url: str
    model: str
    api_key: str = ""
    ca_cert: str = ""
    kind: str = "remote"          # "course" | "local" | "hf" | "remote"
    template_kwargs: bool = True  # send chat_template_kwargs for Qwen3 (only the course endpoint is known to accept it)

    def ready(self) -> tuple[bool, str]:
        if self.kind == "course":
            missing = config.missing_llm_env()
            if missing:
                return False, f"{', '.join(missing)} not set in .env"
        if not self.base_url or not self.model:
            return False, "base URL or model not set"
        if self.kind not in ("local", "course") and not self.api_key:
            return False, {"hf": "HF_TOKEN is not set in .env"}.get(self.kind, "API key is not set")
        if self.ca_cert and not Path(self.ca_cert).exists():
            return False, f"certificate file not found: {self.ca_cert}"
        if self.kind == "local":
            return _probe_ollama(self)
        return True, ""

    def client(self, sleep: float = 0.0):
        return llm.EndpointClient(self.base_url, self.api_key or "ollama", self.model, False, sleep,
                                  self.ca_cert or None, False, template_kwargs=self.template_kwargs)

    def public(self) -> dict:
        ok, why = self.ready()
        return {"id": self.id, "label": self.label, "model": self.model, "kind": self.kind, "ready": ok, "reason": why}


def _probe_ollama(p: Profile) -> tuple[bool, str]:
    hit = _probe_cache.get(p.id)
    if hit and time.time() - hit[0] < _PROBE_TTL_S:
        return hit[1]
    try:
        req = urllib.request.Request(p.base_url.rstrip("/") + "/models")
        with urllib.request.urlopen(req, timeout=1.5) as r:
            ids = {m.get("id") for m in json.load(r).get("data", [])}
        wanted = {p.model, p.model + ":latest"}
        res = (True, "") if ids & wanted else (False, f"model '{p.model}' not pulled yet (run: bash setup_ollama.sh)")
    except Exception:
        res = (False, "Ollama is not running (open the Ollama app, then bash setup_ollama.sh)")
    _probe_cache[p.id] = (time.time(), res)
    return res


def _abs(path: str) -> str:
    return str(config.ROOT / path) if path and not Path(path).is_absolute() else path


def load_profiles() -> list[Profile]:
    """Built from the environment; models.json (if present) adds or replaces profiles by id."""
    env = os.environ.get
    profiles = {p.id: p for p in [
        Profile("qwen3-8b", "Qwen3-8B (course endpoint)", env("LLM_BASE_URL", ""), env("LLM_MODEL", ""),
                env("LLM_API_KEY", ""), _abs(env("LLM_CA_CERT", "")), "course"),
        Profile("llama31-8b-local", "Llama 3.1 8B (local, free)", env("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
                env("OLLAMA_MODEL", "llama3.1-16k"), "", "", "local", False),
        Profile("hf-router", "Hugging Face router", "https://router.huggingface.co/v1",
                env("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct"), env("HF_TOKEN", ""), "", "hf", False),
    ]}
    path = Path(env("MODELS_FILE") or config.ROOT / "models.json")
    if path.exists():
        for item in json.loads(path.read_text(encoding="utf-8")):
            key = os.environ.get(item.get("api_key_env", ""), "") if item.get("api_key_env") else item.get("api_key", "")
            profiles[item["id"]] = Profile(item["id"], item.get("label", item["id"]), item["base_url"], item["model"],
                                           key, _abs(item.get("ca_cert", "")), item.get("kind", "remote"),
                                           bool(item.get("template_kwargs", False)))
    return list(profiles.values())


def get_profile(model_id: str) -> Profile:
    for p in load_profiles():
        if p.id == model_id:
            return p
    raise KeyError(f"Unknown model id '{model_id}'. Known: {', '.join(p.id for p in load_profiles())}")
