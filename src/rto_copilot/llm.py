"""Endpoint client, mock client and JSON handling for the course Qwen endpoint (OpenAI-compatible chat completions).

Standard library only. The endpoint caps input plus output at 8192 tokens and allows one active request per group
key, so calls are sequential. Raw responses are cached on disk (see ``config.LLM_CACHE``) so a re-run of the same
prompt is free and reproducible.
"""
from __future__ import annotations

import hashlib
import json
import re
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import config
from .scoring import gold_response

_cache_dir: Path | None = None


def set_cache_dir(path: Path | None) -> None:
    """Point the response cache somewhere else (a fresh directory gives real, uncached latency)."""
    global _cache_dir
    _cache_dir = Path(path) if path else None


def cache_dir() -> Path:
    return _cache_dir or config.LLM_CACHE


class EndpointClient:
    def __init__(self, base_url, api_key, model, json_mode, sleep, ca_cert=None, thinking=False):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.model = model
        self.json_mode = json_mode
        self.sleep = sleep
        self.thinking = thinking
        self.ssl = ssl.create_default_context(cafile=ca_cert) if ca_cert else None
        cache_dir().mkdir(parents=True, exist_ok=True)

    def complete(self, messages, temperature, row, sample=0):
        payload = {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": 900}
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        if not self.thinking and "qwen3" in self.model.lower():
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        key = hashlib.sha256(json.dumps([payload, sample], sort_keys=True).encode()).hexdigest()
        cached = cache_dir() / f"{key}.json"
        if cached.exists():
            return json.loads(cached.read_text(encoding="utf-8"))
        body = json.dumps(payload).encode()
        for attempt in range(4):
            req = urllib.request.Request(self.url, data=body, method="POST", headers={
                "Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
            start = time.time()
            try:
                with urllib.request.urlopen(req, timeout=180, context=self.ssl) as resp:
                    data = json.loads(resp.read().decode())
                break
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 504) and attempt < 3:
                    time.sleep(2 ** attempt * 2)
                    continue
                raise RuntimeError(f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}") from e
            except urllib.error.URLError:
                if attempt < 3:
                    time.sleep(2 ** attempt * 2)
                    continue
                raise
        usage = data.get("usage") or {}
        result = {
            "text": data["choices"][0]["message"].get("content") or "",
            "latency_s": round(time.time() - start, 2),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
        }
        cached.write_text(json.dumps(result), encoding="utf-8")
        time.sleep(self.sleep)
        return result


class MockClient:
    def __init__(self, mode):
        self.mode = mode
        self.model = f"mock-{mode}"

    def complete(self, messages, temperature, row, sample=0):
        if self.mode == "oracle":
            body = gold_response(row)
        else:
            body = {"behavior": "ANSWER", "answer": "Please contact your RTO.", "citations": [],
                    "findings": [], "drafts": []}
        if messages[0]["content"].startswith("You are the intake step"):
            body = {"behavior": body["behavior"], "reason": "mock"}
        return {"text": json.dumps(body), "latency_s": 0.0, "prompt_tokens": 0, "completion_tokens": 0}


def parse_json(text):
    text = re.sub(r"<think>.*?(</think>|$)", "", text, flags=re.S).strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fence:
        text = fence.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def call_json(client, messages, temperature, row, stats, sample=0):
    result = client.complete(messages, temperature, row, sample)
    stats["calls"] += 1
    stats["latency_s"] += result["latency_s"]
    stats["tokens"] += result["prompt_tokens"] + result["completion_tokens"]
    obj = parse_json(result["text"])
    if obj is None:
        stats["malformed"] += 1
        retry = messages + [{"role": "assistant", "content": result["text"]},
                            {"role": "user", "content": "That was not valid JSON. Reply with the JSON object only."}]
        result = client.complete(retry, 0.0, row, sample)
        stats["calls"] += 1
        stats["latency_s"] += result["latency_s"]
        stats["tokens"] += result["prompt_tokens"] + result["completion_tokens"]
        obj = parse_json(result["text"])
    return obj, result["text"]
