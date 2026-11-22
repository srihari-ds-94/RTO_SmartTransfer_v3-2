"""BLEU and ROUGE for comparing an answer with the reference (gold) answer. Pure Python, no extra packages.

Both metrics only count word overlap. They do not know whether a date is right or a citation is real, so they are a
secondary signal next to the six-check scorer in ``scoring.score`` (which stays the pass/fail rule).

Tokens are lower-case words and numbers. BLEU is sentence-level BLEU-4 with add-one smoothing on the 2 to 4 gram
precisions (Lin and Och 2004) and the usual brevity penalty. ROUGE-1, ROUGE-2 and ROUGE-L are F1 scores without stemming.
"""
from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9]+(?:[./-][a-z0-9]+)*")


def tokens(text: str) -> list[str]:
    return _TOKEN.findall((text or "").lower())


def _ngrams(toks: list[str], n: int) -> Counter:
    return Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))


def bleu(candidate: str, reference: str, max_n: int = 4) -> float:
    c, r = tokens(candidate), tokens(reference)
    if not c or not r:
        return 0.0
    logs = 0.0
    for n in range(1, max_n + 1):
        cg, rg = _ngrams(c, n), _ngrams(r, n)
        match = sum(min(v, rg[g]) for g, v in cg.items())
        total = max(len(c) - n + 1, 0)
        if n == 1:
            if match == 0:
                return 0.0
            p = match / total
        else:
            p = (match + 1) / (total + 1)
        logs += math.log(p) / max_n
    bp = 1.0 if len(c) > len(r) else math.exp(1 - len(r) / len(c))
    return bp * math.exp(logs)


def _f1(match: int, c_len: int, r_len: int) -> float:
    if not match:
        return 0.0
    p, r = match / c_len, match / r_len
    return 2 * p * r / (p + r)


def rouge_n(candidate: str, reference: str, n: int) -> float:
    c, r = _ngrams(tokens(candidate), n), _ngrams(tokens(reference), n)
    return _f1(sum(min(v, r[g]) for g, v in c.items()), sum(c.values()), sum(r.values()))


def _lcs(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def rouge_l(candidate: str, reference: str) -> float:
    c, r = tokens(candidate), tokens(reference)
    return _f1(_lcs(c, r), len(c), len(r)) if c and r else 0.0


def text_scores(candidate: str, reference: str) -> dict:
    return {"bleu": round(bleu(candidate, reference), 4), "rouge1": round(rouge_n(candidate, reference, 1), 4),
            "rouge2": round(rouge_n(candidate, reference, 2), 4), "rougeL": round(rouge_l(candidate, reference), 4)}
