# RTO SmartTransfer Copilot v3: FastAPI + one static page, CPU only.
# Build from this folder:  docker compose up --build   (then open http://localhost:8000)
# Only the code, the small data files and the prebuilt index are copied. Results, notebooks, .env and caches stay out.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/opt/hf

WORKDIR /app
COPY requirements.txt /tmp/requirements.txt
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1 \
 && pip install -r /tmp/requirements.txt

# Retrieval models are downloaded at build time so the container runs offline from the Hugging Face Hub.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5', device='cpu'); CrossEncoder('BAAI/bge-reranker-base', device='cpu')"

COPY pyproject.toml /app/
COPY src /app/src
COPY data/cases /app/data/cases
COPY data/registries /app/data/registries
COPY data/eval /app/data/eval
COPY data/spec /app/data/spec
COPY data/corpus/documents.jsonl data/corpus/sources_manifest.csv /app/data/corpus/
COPY data/index /app/data/index
COPY data/pricing.json /app/data/pricing.json
RUN pip install --no-deps .

RUN useradd --create-home --uid 1000 copilot \
 && mkdir -p /data/state /data/llm_cache \
 && chown -R copilot /data /app/data/index
USER copilot

ENV RTO_HOME=/app HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 RTO_HOST=0.0.0.0 RTO_PORT=8000 \
    COPILOT_STATE_DIR=/data/state LLM_CACHE_DIR=/data/llm_cache \
    RATE_PER_MIN=6 RATE_BURST=3 GLOBAL_PER_MIN=30 CACHE_TTL_S=3600 CACHE_MAX_ITEMS=256 MAX_QUERY_CHARS=1000 GATE_WAIT_S=90

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)"
CMD ["rto-serve"]
