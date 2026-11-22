"""OpenTelemetry tracing to a local JSONL file (one span per line). No external collector or account needed.

The same spans also go to Langfuse, Jaeger or any OTLP/HTTP backend when OTEL_EXPORTER_OTLP_ENDPOINT (or
OTEL_EXPORTER_OTLP_TRACES_ENDPOINT) is set; headers such as Langfuse's Basic auth come from OTEL_EXPORTER_OTLP_HEADERS.
Needs the opentelemetry-exporter-otlp-proto-http package.
"""
from __future__ import annotations

import atexit
import json
import os
import threading
from contextlib import contextmanager
from pathlib import Path

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.trace import Status, StatusCode

from .guardrails import redact_pii


class JsonlSpanExporter(SpanExporter):
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def export(self, spans) -> SpanExportResult:
        with self.lock, self.path.open("a", encoding="utf-8") as f:
            for s in spans:
                f.write(json.dumps({
                    "trace_id": f"{s.context.trace_id:032x}", "span_id": f"{s.context.span_id:016x}",
                    "parent_id": f"{s.parent.span_id:016x}" if s.parent else None, "name": s.name,
                    "start_ns": s.start_time, "end_ns": s.end_time,
                    "duration_ms": round((s.end_time - s.start_time) / 1e6, 1),
                    "status": s.status.status_code.name, "attributes": dict(s.attributes or {}),
                    "events": [{"name": e.name, "attributes": dict(e.attributes or {})} for e in s.events],
                }, ensure_ascii=False, default=str) + "\n")
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass


_provider: TracerProvider | None = None


def setup(path: Path | None) -> None:
    """Send spans to path; with None, spans are created but not exported."""
    global _provider
    _provider = TracerProvider(resource=Resource.create({"service.name": "rto-smarttransfer-copilot"}))
    if path:
        _provider.add_span_processor(SimpleSpanProcessor(JsonlSpanExporter(path)))
    if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") or os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"):
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        _provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        atexit.register(_provider.shutdown)


def tracer():
    if _provider is None:
        setup(None)
    return _provider.get_tracer("copilot")


@contextmanager
def span(name: str, **attributes):
    with tracer().start_as_current_span(name) as s:
        for k, v in attributes.items():
            set_attr(s, k, v)
        try:
            yield s
        except Exception as e:
            s.set_status(Status(StatusCode.ERROR, str(e)[:200]))
            s.record_exception(e)
            raise


def set_attr(s, key: str, value) -> None:
    """Strings are PII-redacted before they are recorded, so traces never hold an Aadhaar, PAN, phone or e-mail."""
    if value is None:
        return
    if not isinstance(value, (str, bool, int, float)):
        value = json.dumps(value, ensure_ascii=False, default=str)[:4000]
    if isinstance(value, str):
        value = redact_pii(value)[0]
    s.set_attribute(key, value)
