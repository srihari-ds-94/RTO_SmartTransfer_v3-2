"""Chunking experiments and hybrid retrieval (BM25 + dense + cross-encoder reranker) over corpus/documents.jsonl.

Retrieval is scored on its own, before any answer is generated: for each dev query with an official gold source,
did the gold source_id appear in the top k? Test-split rows are never used.

    rto-retrieval                          # every chunking config x method x query mode, scored on the dev split
    rto-retrieval --configs w128           # one chunking config only
    rto-retrieval --query "..."            # top hits for one question with the default config
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from . import config as cfg

DOCS = cfg.CORPUS_DOCS
INDEX = cfg.INDEX
RESULTS = cfg.RESULTS / "retrieval"
EVAL = cfg.EVAL
CASES = cfg.CASES

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-base"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
NOT_RETRIEVABLE = {"SRC-SPEC-SCOPE", "SRC-PACK-DATASET"}  # project policy and case data, not corpus documents

# name -> (words per chunk, overlap); None keeps each section / rule / FAQ entry whole
CHUNK_CONFIGS = {"section": None, "w128": (128, 32), "w256": (256, 64), "w384": (384, 96)}
DEFAULT_CONFIG = "w128"
DEFAULT_METHOD = "hybrid"
# hybrid_rerank replaces the hybrid order with the cross-encoder order; hybrid_rerank_rrf fuses the two orders
METHODS = ["bm25", "dense", "hybrid", "hybrid_rerank", "hybrid_rerank_rrf"]
POOL = 50          # candidates taken from each retriever before fusion
RERANK_TOP = 30    # fused candidates passed to the cross-encoder
RRF_K = 60
KS = (1, 3, 5, 10)

# The gold labels name the Parivahan page or the Act; the corpus also holds the CMVR rule / Act section that says
# the same thing. "Lenient" recall accepts these; "strict" recall (the headline number) does not.
EQUIVALENT = {
    "SRC-PARIVAHAN-TO": {"SRC-CMVR-R55", "SRC-MVA-S50"},
    "SRC-MVA-S50": {"SRC-CMVR-R55", "SRC-PARIVAHAN-TO"},
    "SRC-PARIVAHAN-HPT": {"SRC-CMVR-R61", "SRC-MVA-S51"},
    "SRC-PARIVAHAN-HPA": {"SRC-CMVR-R60", "SRC-MVA-S51"},
    "SRC-PARIVAHAN-HPC": {"SRC-MVA-S51"},
    "SRC-MVA-S51": {"SRC-CMVR-R60", "SRC-CMVR-R61"},
    "SRC-PARIVAHAN-NOC": {"SRC-CMVR-R58", "SRC-MVA-S48"},
    "SRC-PARIVAHAN-ADDR": {"SRC-CMVR-R59", "SRC-MVA-S49"},
}

LOAN_TEXT = {
    "NO_LOAN": "no loan or hypothecation on the vehicle",
    "LOAN_TERMINATED": "loan closed, hypothecation to be terminated",
    "LOAN_ACTIVE_WITH_CONSENT": "loan still active under hypothecation, financier has given written consent",
    "LOAN_ACTIVE_PENDING": "loan still active under hypothecation, financier consent pending",
}
DOC_TEXT = {"has_rc": "registration certificate", "has_insurance": "insurance certificate",
            "has_puc": "pollution under control certificate", "has_form_29": "Form 29", "has_form_30": "Form 30"}

STOP = set("a an and are as at be by can do does for from has have how i in is it its may me my of on or our shall "
           "should so that the their them then there this to under was we what when where which who will with you your".split())


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP]


def load_docs() -> list[dict]:
    with DOCS.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_cases() -> dict[str, dict]:
    with CASES.open(newline="", encoding="utf-8-sig") as f:
        return {r["case_id"]: r for r in csv.DictReader(f)}


def case_aware_query(query: str, case: dict | None) -> str:
    """Append what the case record says (never expected_findings) so 'Check case DEMO-053' has content to match."""
    if not case:
        return query
    parts = ["vehicle transfer of ownership", LOAN_TEXT.get(case["scenario"], "")]
    if case.get("financier_status"):
        parts.append(f"financier status {case['financier_status']}")
    missing = [name for col, name in DOC_TEXT.items() if case.get(col) == "0"]
    if missing:
        parts.append("missing " + ", ".join(missing))
    if case.get("engine_number_declared") != case.get("engine_number_rc"):
        parts.append("engine number does not match the registration certificate")
    return query + " | " + "; ".join(p for p in parts if p)


def make_chunks(docs: list[dict], config: str) -> list[dict]:
    spec = CHUNK_CONFIGS[config]
    chunks = []
    for d in docs:
        words = d["text"].split()
        if spec is None or len(words) <= spec[0]:
            windows = [words]
        else:
            size, overlap = spec
            step = size - overlap
            windows = [words[i:i + size] for i in range(0, max(len(words) - overlap, 1), step)]
        for j, w in enumerate(windows):
            # the title is prepended so a window from the middle of s.50 still says it is s.50 of the MVA
            chunks.append({"chunk_id": f"{d['doc_id']}#{j}", "doc_id": d["doc_id"], "source_id": d["source_id"],
                           "title": d["title"], "url": d["url"], "locator": d["locator"],
                           "text": f"{d['title']}. {' '.join(w)}"})
    return chunks


class Retriever:
    def __init__(self, config: str, docs: list[dict] | None = None, use_reranker: bool = True):
        from sentence_transformers import SentenceTransformer
        self.config = config
        self.chunks = make_chunks(docs or load_docs(), config)
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])
        self.embedder = SentenceTransformer(EMBED_MODEL, device="cpu")
        self.emb = self._embeddings()
        self.reranker = None
        if use_reranker:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(RERANK_MODEL, device="cpu", max_length=512)

    def _embeddings(self) -> np.ndarray:
        INDEX.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256("\n".join(c["text"] for c in self.chunks).encode()).hexdigest()[:16]
        path = INDEX / f"{self.config}_{digest}.npy"
        if path.exists():
            return np.load(path)
        emb = self.embedder.encode([c["text"] for c in self.chunks], batch_size=32, normalize_embeddings=True,
                                   show_progress_bar=True)
        np.save(path, emb)
        with (INDEX / f"{self.config}_{digest}.chunks.jsonl").open("w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        return emb

    def bm25_rank(self, query: str) -> list[int]:
        return list(np.argsort(-self.bm25.get_scores(tokenize(query)))[:POOL])

    def dense_rank(self, query: str) -> list[int]:
        q = self.embedder.encode([QUERY_PREFIX + query], normalize_embeddings=True)[0]
        return list(np.argsort(-(self.emb @ q))[:POOL])

    @staticmethod
    def rrf(*rankings: list[int]) -> list[int]:
        scores: dict[int, float] = {}
        for ranking in rankings:
            for r, i in enumerate(ranking):
                scores[i] = scores.get(i, 0.0) + 1.0 / (RRF_K + r + 1)
        return sorted(scores, key=scores.get, reverse=True)

    def hybrid_rank(self, query: str) -> list[int]:
        return self.rrf(self.bm25_rank(query), self.dense_rank(query))

    def rerank(self, query: str, candidates: list[int]) -> list[int]:
        key = (query, tuple(candidates[:RERANK_TOP]))
        cache = self.__dict__.setdefault("_rerank_cache", {})
        if key not in cache:
            top = candidates[:RERANK_TOP]
            scores = self.reranker.predict([(query, self.chunks[i]["text"]) for i in top], batch_size=16)
            cache[key] = [top[i] for i in np.argsort(-scores)]
        return cache[key] + candidates[RERANK_TOP:]

    def search(self, query: str, method: str = DEFAULT_METHOD) -> list[int]:
        if method == "bm25":
            return self.bm25_rank(query)
        if method == "dense":
            return self.dense_rank(query)
        fused = self.hybrid_rank(query)
        if method == "hybrid":
            return fused
        reranked = self.rerank(query, fused)
        return reranked if method == "hybrid_rerank" else self.rrf(fused, reranked)

    def top_sources(self, ranking: list[int]) -> list[str]:
        """Collapse a chunk ranking to source_ids in first-seen order (several chunks of s.50 count once)."""
        seen, out = set(), []
        for i in ranking:
            s = self.chunks[i]["source_id"]
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out


def dev_queries(splits: tuple[str, ...] = ("dev",)) -> list[dict]:
    """Rows with an official corpus gold source. Dev only by default: test rows are never used for tuning."""
    with EVAL.open(newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] in splits]
    out = []
    for r in rows:
        gold = {s for s in r["gold_source_ids"].split(";") if s} - NOT_RETRIEVABLE
        if gold:
            out.append({"query_id": r["query_id"], "category": r["category"], "query": r["query"],
                        "case_id": r["case_id"], "gold": sorted(gold)})
    return out


def score(sources: list[str], gold: list[str]) -> dict:
    g = set(gold)
    row = {f"recall@{k}": len(g & set(sources[:k])) / len(g) for k in KS}
    row.update({f"hit@{k}": float(bool(g & set(sources[:k]))) for k in KS})
    first = next((r for r, s in enumerate(sources[:10], 1) if s in g), None)
    row["mrr@10"] = 1.0 / first if first else 0.0
    # lenient: a gold source counts as found if it or a legally equivalent corpus source is in the top k
    for k in (5, 10):
        top = set(sources[:k])
        row[f"lenient_recall@{k}"] = sum(bool(({s} | EQUIVALENT.get(s, set())) & top) for s in g) / len(g)
    return row


def evaluate(configs: list[str], methods: list[str], query_modes: list[str]) -> int:
    queries = dev_queries()
    cases = load_cases()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS.mkdir(parents=True, exist_ok=True)
    docs = load_docs()
    summary, per_query = [], []
    for config in configs:
        ret = Retriever(config, docs, use_reranker=any("rerank" in m for m in methods))
        print(f"\n{config}: {len(ret.chunks)} chunks", flush=True)
        for mode in query_modes:
            for method in methods:
                rows = []
                for q in queries:
                    text = case_aware_query(q["query"], cases.get(q["case_id"])) if mode == "case_aware" else q["query"]
                    sources = ret.top_sources(ret.search(text, method))
                    s = score(sources, q["gold"])
                    rows.append(s)
                    per_query.append({"config": config, "query_mode": mode, "method": method, **q,
                                      "search_text": text, "top10": sources[:10], **s})
                avg = {k: round(sum(r[k] for r in rows) / len(rows), 3) for k in rows[0]}
                summary.append({"config": config, "chunks": len(ret.chunks), "query_mode": mode, "method": method,
                                "queries": len(rows), **avg})
                print(f"  {mode:<10} {method:<18} recall@5 {avg['recall@5']:.3f}  hit@5 {avg['hit@5']:.3f}  "
                      f"mrr@10 {avg['mrr@10']:.3f}  lenient@5 {avg['lenient_recall@5']:.3f}", flush=True)
    out = RESULTS / f"retrieval_{stamp}"
    out.mkdir()
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    with (out / "per_query.jsonl").open("w", encoding="utf-8") as f:
        for r in per_query:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n{len(queries)} dev queries with an official gold source. Results: {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=",".join(CHUNK_CONFIGS))
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--query-modes", default="plain,case_aware")
    ap.add_argument("--query")
    ap.add_argument("--method", default=DEFAULT_METHOD, choices=METHODS)
    args = ap.parse_args()
    if args.query:
        ret = Retriever(DEFAULT_CONFIG)
        for rank, i in enumerate(ret.search(args.query, args.method)[:5], 1):
            c = ret.chunks[i]
            print(f"{rank}. [{c['source_id']}] {c['title']}\n   {c['text'][len(c['title']) + 2:][:200]}...")
        return 0
    split = lambda s: [x.strip() for x in s.split(",") if x.strip()]
    return evaluate(split(args.configs), split(args.methods), split(args.query_modes))


if __name__ == "__main__":
    sys.exit(main())
