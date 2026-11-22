# Review of the Week 1 to 4 project and what v3 changes

Scope: the teammate's folders `08` to `13` plus the dataset and sources. The original folders were not modified;
everything below is in this v3 copy. Behaviour was preserved on purpose: the scorer is unchanged, so old and new numbers
stay comparable, and all 33 original tests still pass (37 with the new API tests).

## What is already good (keep it)

* **Spec first, then code.** Scope, refusal classes, response contract and scoring rules were written before the agent.
* **A frozen yardstick.** v1 is hashed (SHA-256); the test split is gated behind that hash.
* **One scorer for every week**, so baseline, RAG and agent numbers are directly comparable.
* **Deterministic tools for anything checkable** (dates, finding codes, loan path), with the model limited to planning,
  explaining and drafting.
* **Defence in depth**: PII redaction, injection screening, rate limits, answer cache, human review log, tracing.
* **Honest reporting**: failure taxonomy, ablations (LLM critic, tool faults), confidence intervals in the Week 4 report.

## Structure and best-practice changes made

| Finding in the original | Why it matters | Change in v3 |
|---|---|---|
| Week-numbered folders with `sys.path` edits and `import run_baseline as rb` across folders | Fragile imports, impossible to install or test cleanly | One installable package `src/rto_copilot/` (`pip install -e .`), `pyproject.toml`, console scripts (`rto-baseline`, `rto-agent-eval`, `rto-evalset`, `rto-serve`, ...) |
| Paths hard-coded per file | Every move breaks something | `config.py` is the single source of paths and env settings (`RTO_HOME`, `RTO_EVAL_SET`, `COPILOT_STATE_DIR`) |
| `run_baseline.py` mixed prompts, HTTP client, scorer and CLI in one large file | Hard to read, hard to reuse | Split into `llm.py`, `scoring.py`, `baseline.py`; scorer body untouched |
| Eval validator tied to v1 (60 to 100 rows) | Could not grow the set safely | `evalset.py`: version-aware, v1 immutability check, no duplicate or near-duplicate questions, no dev/test case leakage, margin of error printed |
| Test split already spent, only 34 test rows | A second look at test would be tuning on test | v2 adds a fresh test split built from new test-split cases; `check_test_split_allowed` blocks unfrozen test runs |
| Streamlit front end (heavy install, basic page) | Slow container, hard to restyle | FastAPI + one static HTML page, no build step, works without the LLM for case lookup |
| Dockerfile and compose referenced the teammate's folder layout and a path outside the repo for the CA cert | Not reproducible | New Dockerfile and compose for this layout, non-root user, read-only filesystem, health check on `/api/health` |
| Scripts only, no way to demo step by step | You asked for notebooks | Seven notebooks (00 to 06) that run offline from saved results and mark live cells clearly |
| No `.gitignore`, endpoint IP in `.env.example` | Risk of committing secrets | `.gitignore` added; `.env` stays out of git and out of the image. The IP in `.env.example` is the teammate's; consider replacing it with a placeholder before sharing outside the team |
| No tests for the web layer | New layer, new risks | `tests/test_api.py` covers page, case view without a model, ask, review log and the no-model 503 |

## Evaluation findings (the part to be candid about)

1. **The dev to test drop is partly noise and partly overfitting to dev.** Dev 56.1% (32 of 57), test 32.4% (11 of 34).
   Fisher exact p is about 0.03, so it is borderline real, with overlapping intervals (43% to 68% vs 19% to 49%).
2. **Behaviour accuracy did not drop** (73.7% dev, 73.5% test) and finding codes and drafts are perfect on both. The gap
   is in required facts (60% vs 53%) and citations (97% vs 88%).
3. **The strict end-to-end metric amplifies small misses.** It is the AND of six checks, and half of the failing test
   rows missed exactly one check (15.8% of dev rows). `component_report` now prints this so nobody reads 32% as "the
   system is wrong two times in three".
4. **More rows do not raise accuracy by themselves.** They narrow the interval (test margin about 17 points at n=34,
   about 10 at n=93), give a clean held-out set, and reveal weak categories. Raising accuracy needs the changes below.
5. **Retrieval is not the bottleneck.** Gold context scored the same as retrieved context (17.5% each) in Week 2, so
   effort belongs in answer completeness, not in a better retriever.
6. **The cross-encoder reranker is not a clear win** (helps case-aware queries, hurts plain ones). The fused ranker is the
   safer default.
7. **A second LLM as critic did not beat the rule critic** (47% vs 56%) and costs more. Keep the rule critic.
8. **Safety evidence is real but small.** 10 injection attacks and 4 PII probes are smoke tests, not proof.
9. **Substring `must_include` is brittle.** A correct paraphrase can fail. Alternatives with `|` soften this; a
   judged-faithfulness score (spec 5.3) as a secondary metric would show how much of the 32% is wording.
10. **Latency is dominated by writing the answer** (55% of a turn) and about 9.8K prompt tokens per query. Trimming
    retrieved context is the cheapest speed and cost lever.

## Recommended next steps, in order

1. Team reviews the 98 new rows (checklist in `EVAL_V2_CHANGELOG.md`), then `rto-evalset --version v2 --freeze`.
2. Live run on v2 dev, then **one** run on the v2 test split. Report with intervals. Do not tune on the v2 test split.
3. Over-sample the weak categories when iterating (PROC, HYP, CLAR, DRAFT scored 0 of 12 on the v1 test split, with
   wide intervals).
4. Add alternate phrasings to `must_include` where the team agrees a paraphrase is correct (changes the yardstick, so
   do it before the v2 freeze, never after).
5. Try prompt trimming (fewer, shorter chunks) and measure accuracy, cost per correct answer and latency together.
6. Optional: a judged-faithfulness score as a secondary metric, spot-checked by a human on 20 rows as the spec says.

## Known loose ends

* `CLR-05` (frozen) says cases run to DEMO-100; they now run to DEMO-148. Left as is, harmless for scoring.
* `R-DAY-COUNT` is a team decision and `R-CONSENT-CONTINUATION` is unconfirmed for West Bengal; both are flagged in every
  answer that depends on them.
* Live model runs could not be done while preparing v3 (the endpoint was unreachable from the build environment), so
  every live number in the notebooks is the teammate's saved Week 1 to 4 run, not a fresh one.
