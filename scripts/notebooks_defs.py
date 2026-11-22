"""Notebooks 02 to 06 (see build_notebooks.py for 00 and 01)."""


def build_rest(build, md, code, SETUP):
    return [nb02(build, md, code, SETUP), nb03(build, md, code, SETUP), nb04(build, md, code, SETUP),
            nb05(build, md, code, SETUP), nb06(build, md, code, SETUP)]


# ============================================================================ 02 baseline
def nb02(build, md, code, SETUP):
    return build("02_Prompt_Baseline.ipynb", [
        md("""
# Week 1: the prompt-only baseline

**Question.** Before building anything, how well does the model do with just a good prompt?

Four prompting strategies, same model (Qwen3-8B), same 57 dev rows, no tools and no sources:

| Strategy | Idea |
|---|---|
| S0 zero-shot | Rules in the system prompt, nothing else |
| S1 few-shot | Plus worked examples |
| S2 self-consistency | Three samples, majority vote |
| S3 decomposition | Classify first, then answer |

**Business reading.** This is the "just paste it into a chatbot" option. It sets the bar every later week must clear.
"""),
        code(SETUP),
        code('''
base = pd.read_csv(config.RESULTS / "baseline/week1_qwen3_8b/summary.csv")
base["k"] = (base.end_to_end_accuracy * base["items"]).round().astype(int)
base["label"] = base.strategy.str.replace("_", "\\n", n=1)
base[["strategy", "items", "end_to_end_accuracy", "behavior_accuracy", "must_include_coverage", "citation_ok_rate", "avg_latency_s", "avg_tokens"]].round(3)
'''),
        code('''
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
nbkit.ci_bars(axes[0], base.label, base.k, base["items"], colors=[nbkit.GREY, nbkit.TEAL, nbkit.GREY, nbkit.GREY])
axes[0].set_title("End-to-end accuracy (strict)")
kb = (base.behavior_accuracy * base["items"]).round().astype(int)
nbkit.ci_bars(axes[1], base.label, kb, base["items"], colors=[nbkit.GREY, nbkit.TEAL, nbkit.GREY, nbkit.GREY])
axes[1].set_title("Behaviour accuracy (right kind of response)")
plt.tight_layout(); plt.show()
'''),
        md("""
**Read-out.** Few-shot is best, and the extra machinery (voting, decomposition) did *not* help: more calls, more tokens,
no gain. All four intervals overlap heavily, so do not over-read the ranking. What is solid is the level: roughly one
row in five passes every check. The model does not know what it does not know, and it cannot cite sources it has not been given.
"""),
        md("## Where does a baseline row lose its point?"),
        code('''
fail = base.set_index("strategy")[[c for c in base.columns if c.startswith("fail_")]]
fail.columns = [c.replace("fail_", "").replace("_", " ").title() for c in fail.columns]
fig, ax = plt.subplots(figsize=(10, 4))
fail.T.plot.barh(ax=ax, width=0.8, color=[nbkit.GREY, nbkit.TEAL, nbkit.SKY, nbkit.NAVY])
ax.set_title("Failure tags per strategy (a row can carry several)"); ax.set_xlabel("Rows"); plt.tight_layout(); plt.show()
'''),
        md("""
**Missing fact** dominates: the answer is plausible but misses a required item (a form number, a date, a section).
**Wrong behaviour** (answering when it should clarify or refuse, or the reverse) and **citation errors** (sources that are
wrong or that it never saw) follow. Together they are *grounding* problems,
which is exactly what Week 2 (retrieval) and Week 3 (tools) attack.

## Cost of each strategy
"""),
        code('''
fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(base.avg_tokens, base.end_to_end_accuracy, s=140, color=nbkit.TEAL)
for _, r in base.iterrows():
    ax.annotate(r.strategy.split("_")[0], (r.avg_tokens, r.end_to_end_accuracy), textcoords="offset points", xytext=(8, 6))
ax.set_xlabel("Average tokens per query"); ax.set_ylabel("End-to-end accuracy"); ax.set_title("More tokens did not buy accuracy"); plt.show()
'''),
        md("## A concrete failing row"),
        code('''
v1 = pd.read_csv(config.eval_path("v1")).set_index("query_id")
rows = nbkit.load_jsonl(config.RESULTS / "baseline/week1_qwen3_8b/S1_FEW_SHOT.jsonl")
bad = next(r for r in rows if not r["end_to_end"] and r["failure_tags"] and "MISSING_FACT" in r["failure_tags"])
print("Query    :", v1.loc[bad["query_id"], "query"])
print("Gold     :", v1.loc[bad["query_id"], "gold_answer"][:300])
print("Must have:", v1.loc[bad["query_id"], "must_include"])
print("Tags     :", bad["failure_tags"])
print("Model    :", str(bad["raw"])[:450])
'''),
        md("""
## Live: re-run the baseline (optional)

Needs the endpoint. Uses the v2 dev rows (96 after excluding the few-shot exemplars). Results go to `results/baseline/`.
The test split is gated: it only runs on a frozen eval set.
"""),
        code('''
import subprocess
cmd = [sys.executable, "-m", "rto_copilot.baseline", "--split", "dev", "--cohort", "v2"]
print("Command:", " ".join(cmd), "\\n(cohort v2 = only the new rows; drop --cohort to score all dev rows)")
if RUN_LIVE:
    env = {**os.environ, "RTO_EVAL_SET": "v2", "PYTHONPATH": str(ROOT / "src")}
    print(subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=ROOT).stdout[-3000:])
'''),
    ])


# ============================================================================ 03 RAG
def nb03(build, md, code, SETUP):
    return build("03_RAG_and_Retrieval.ipynb", [
        md("""
# Week 2: retrieval and RAG

**Question.** If we hand the model the official sources, does it answer better? And is retrieval the bottleneck?

**Corpus.** Motor Vehicles Act 1988, Central and West Bengal Motor Vehicles Rules, General Clauses Act, Parivahan pages,
West Bengal FAQ and the forms. Cut into 128-word chunks with 32 overlap (2,308 chunks).

**Retrieval stack.** Keyword search (BM25) and meaning search (bge-small embeddings), fused with reciprocal rank
fusion (RRF), optionally re-scored by a cross-encoder (bge-reranker-base).
"""),
        code(SETUP),
        md("## 1. Which retrieval method works? (46 dev queries with known gold sources)"),
        code('''
ret = pd.read_csv(config.RESULTS / "retrieval/final/summary.csv")
view = ret.pivot(index="method", columns="query_mode", values=["recall@5", "hit@10", "mrr@10"]).round(3)
view
'''),
        code('''
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
order = ["bm25", "dense", "hybrid", "hybrid_rerank", "hybrid_rerank_rrf"]
for ax, metric in zip(axes, ["recall@5", "hit@10"]):
    p = ret.pivot(index="method", columns="query_mode", values=metric).loc[order]
    p.plot.bar(ax=ax, color=[nbkit.GREY, nbkit.TEAL], rot=20); ax.set_title(metric); ax.set_xlabel(""); ax.set_ylim(0, 1)
plt.tight_layout(); plt.show()
'''),
        md("""
**Read-outs.**

* **Hybrid beats either alone.** Keywords catch exact terms like "Form 35"; embeddings catch paraphrase. Fusing them is
  cheap and robust.
* **Case-aware queries help** (the query is expanded with the case's loan scenario and missing documents). Recall@5 moves
  from about 0.57 to 0.62 for hybrid.
* **The reranker is not a clear win.** It helps when the query is case-aware (recall@5 0.65 vs 0.62 for plain hybrid)
  and hurts in plain mode (0.51 vs 0.57). Good example of "measure it, do not assume it". The fused ranker
  (`hybrid_rerank_rrf`) is the safest default because it never fell below plain hybrid on hit@10.
* With 46 queries, differences of a few points are within noise.
"""),
        md("## 2. Does retrieval quality limit answer quality?"),
        code('''
rag = pd.read_csv(config.RESULTS / "rag/week2_qwen3_8b/summary.csv").set_index("strategy")
cols = ["end_to_end_accuracy", "behavior_accuracy", "must_include_coverage", "citation_ok_rate", "retrieval_hit_rate", "cited_outside_context_rate"]
pd.concat([rag[cols], pd.read_csv(config.RESULTS / "baseline/week1_qwen3_8b/summary.csv").set_index("strategy").loc[["S1_FEW_SHOT"], ["end_to_end_accuracy", "behavior_accuracy", "must_include_coverage", "citation_ok_rate"]]]).round(3)
'''),
        md("""
Two surprising facts, both important:

1. **Perfect context (`RAG_GOLD`) scores the same end-to-end as retrieved context (17.5%).** Retrieval is *not* the
   bottleneck. Even when the right sources are in front of it, the model misses required facts.
2. **Citation quality improves with gold context** (72% to 86%), so retrieval does matter for *trust*, just not for
   the strict pass rate.

This is why Week 3 stopped treating the model as the only brain and moved date arithmetic, finding codes and form
paths into deterministic tools.
"""),
        code('''
diag = pd.read_csv(config.RESULTS / "rag/week2_qwen3_8b/diagnostic.csv")
fig, ax = plt.subplots(figsize=(7, 3.2))
diag.cause.value_counts().sort_values().plot.barh(ax=ax, color=nbkit.TEAL)
ax.set_title("Why RAG rows failed (diagnostic)"); ax.set_ylabel(""); plt.show()
'''),
        md("""
## 3. Live in this notebook: keyword search over the real corpus (no model needed)

This runs BM25 over the saved index so you can show how a question lands on sources.
"""),
        code('''
from rank_bm25 import BM25Okapi
from rto_copilot.retrieval import tokenize
chunk_file = next((config.INDEX).glob("*.chunks.jsonl"))
chunks = nbkit.load_jsonl(chunk_file)
bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
def keyword_search(q, k=5):
    scores = bm25.get_scores(tokenize(q)); top = np.argsort(scores)[::-1][:k]
    return pd.DataFrame([{"rank": i + 1, "source_id": chunks[j]["source_id"], "locator": chunks[j]["locator"], "score": round(float(scores[j]), 1),
                          "snippet": chunks[j]["text"][:110].replace("\\n", " ")} for i, j in enumerate(top)])
print(f"{len(chunks):,} chunks loaded from {chunk_file.name}")
keyword_search("buyer deadline to apply for transfer of ownership Form 30 within days")
'''),
        code('''
keyword_search("financier written consent hypothecation transfer of ownership")
'''),
        md("""
Notice that the Act's section 50 and the Parivahan guidance both surface for the deadline question. That is the
source of the **14 days vs 30 days conflict** the copilot must flag instead of silently picking one.

## 4. Live: full hybrid retrieval with embeddings (optional)

Needs `pip install -e .[rag]` (sentence-transformers) and the model weights. Reproduces the table in section 1.
"""),
        code('''
import importlib.util
have = importlib.util.find_spec("sentence_transformers") is not None
print("sentence-transformers installed:", have)
if RUN_LIVE and have:
    import subprocess
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    print(subprocess.run([sys.executable, "-m", "rto_copilot.retrieval"], capture_output=True, text=True, env=env, cwd=ROOT).stdout[-2500:])
'''),
    ])


# ============================================================================ 04 agent
def nb04(build, md, code, SETUP):
    return build("04_Agentic_Copilot.ipynb", [
        md("""
# Week 3: the agentic copilot

**Question.** What do tools, a critic and deterministic repair add on top of RAG, and why is test accuracy below dev?

**Design in one line.** The LLM plans and writes; **code** calculates dates, derives findings, picks the loan path and
checks the output. Every answer ends in a fixed JSON contract (behaviour, answer, citations, findings, deadlines,
drafts) that a UI and a scorer can both rely on.
"""),
        code(SETUP),
        md("""
## 1. The deterministic tools, live (no model needed)

These run right now on the synthetic data. The LLM can call them but cannot override them.
"""),
        code('''
from rto_copilot import tools as T
chk = T.check_case("DEMO-072")
print("Case", chk["case_id"], "|", chk["scenario"], "|", chk["case_status"])
print("Findings:", [f["code"] for f in chk["findings"]])
print("Escalation triggers:", chk["escalation_triggers"])
print("Form path:", chk["form_path"])
'''),
        code('''
d = T.compute_deadlines("2026-10-20", today="2026-11-10")
pd.DataFrame(d["deadlines"])[["rule_id", "party", "window_days", "due_date", "days_left", "status"]]
'''),
        code('''
print("Conflicts the copilot must surface:", d["conflicts"])
print("\\nRelative date:", T.resolve_date("we sold it yesterday", "2026-11-10"))
try:
    T.resolve_date("last week", "2026-11-10")
except T.ToolError as e:
    print("Vague date is refused, not guessed:", e)
'''),
        md("""
**Business reading.** Dates and finding codes are *never* the model's opinion. `ENGINE_MISMATCH` fires when the declared
engine number differs from the RC, `FINANCIER_REVIEW_REQUIRED` fires when the loan is active and consent is pending.
That is why finding-code accuracy is 100% in every saved run.

## 2. The saved runs (Qwen3-8B)
"""),
        code('''
runs = {"Dev, final (rule critic)": "w3_dev_final", "Test, final": "w3_test_final",
        "Dev, LLM critic (ablation)": "w3_dev_llm_critic", "Dev, 30% tool faults": "w3_dev_faults0.3"}
table = []
for label, folder in runs.items():
    s = json.load(open(config.RESULTS / "agent" / folder / "summary.json"))
    lo, hi = wilson_interval(round(s["end_to_end_accuracy"] * s["items"]), s["items"])
    table.append({"run": label, "n": s["items"], "end_to_end": s["end_to_end_accuracy"], "CI95": f"{lo:.2f} to {hi:.2f}",
                  "behaviour": s["behavior_accuracy"], "facts": s["must_include_coverage"], "citations": s["citation_ok_rate"],
                  "avg_latency_s": s["avg_wall_latency_s"], "avg_llm_calls": s["avg_llm_calls"]})
pd.DataFrame(table).set_index("run")
'''),
        md("""
* **LLM critic (ablation):** adding a second LLM as critic did *not* beat the rule critic (47% vs 56%) and costs
  more. The cheap deterministic critic is the better design here.
* **30% tool faults:** with a third of tool calls failing on purpose, accuracy falls to 47%, not to zero.
  The agent degrades gracefully; it does not crash or invent.

## 3. Why is test lower than dev? Look inside the six checks
"""),
        code('''
dev, test = nbkit.agent_run("w3_dev_final"), nbkit.agent_run("w3_test_final")
cr = {"dev": component_report(dev), "test": component_report(test)}
comp = pd.DataFrame({k: v["components"] for k, v in cr.items()})
comp.loc["END-TO-END (all six)"] = [cr["dev"]["end_to_end"], cr["test"]["end_to_end"]]
fig, ax = plt.subplots(figsize=(10, 4))
comp.plot.bar(ax=ax, color=[nbkit.TEAL, nbkit.NAVY], rot=15); ax.set_ylim(0, 1.1)
ax.set_title("Pass rate of each check, dev (n=57) vs test (n=34)"); ax.set_ylabel("Share of rows passing")
plt.tight_layout(); plt.show()
print("Near-miss rate (rows failing exactly one check): dev", cr["dev"]["near_miss_rate"], "| test", cr["test"]["near_miss_rate"])
'''),
        md("""
**This is the key finding.**

* **Behaviour accuracy is identical** (73.7% dev, 73.5% test). The copilot picks the right kind of response equally
  well on unseen rows.
* **Finding codes and drafts are perfect** on both.
* The gap lives in **facts** (a required phrase missing: 60% vs 53%) and **citations** (97% vs 88%), and the strict
  AND of six checks amplifies small differences. Half the failing test rows missed by exactly one check.

So "test is worse" is mostly "a handful of rows missed one required phrase", on a 34-row split where one row is
three points. That is a measurement and wording issue much more than a reasoning collapse.
"""),
        md("## 4. Which categories are weak? (with honest error bars)"),
        code('''
df = pd.DataFrame([{**r, "split": "dev"} for r in dev] + [{**r, "split": "test"} for r in test])
cat = df.groupby(["category", "split"]).agg(n=("end_to_end", "size"), k=("end_to_end", "sum"), behaviour=("behavior_ok", "mean"), facts=("include_all", "mean")).reset_index()
cat["e2e"] = cat.k / cat.n
pv = cat.pivot(index="category", columns="split", values="e2e").sort_values("test")
fig, ax = plt.subplots(figsize=(9, 4.2))
pv.plot.barh(ax=ax, color=[nbkit.TEAL, nbkit.NAVY]); ax.set_xlim(0, 1.05); ax.set_title("End-to-end by category (test n per category is only 2 to 7)")
plt.show()
cat.pivot(index="category", columns="split", values=["n", "k"]).fillna(0).astype(int)
'''),
        md("""
**Reading with care.** PROC and HYP score 0 on test, but there are only 4 rows each, so the plausible range is 0% to 49%.
These are the categories to over-sample in v2, which is exactly what the 98 new rows do (PROC +12, HYP +10).

## 5. What kind of failure is it?
"""),
        code('''
tax = pd.DataFrame({"dev": pd.Series(json.load(open(config.RESULTS / "agent/w3_dev_final/summary.json"))["failure_taxonomy"]),
                    "test": pd.Series(json.load(open(config.RESULTS / "agent/w3_test_final/summary.json"))["failure_taxonomy"])}).fillna(0)
tax = tax.div(tax.sum()).round(3)
fig, ax = plt.subplots(figsize=(8, 3.6)); tax.T.plot.barh(stacked=True, ax=ax, color=[nbkit.TEAL, nbkit.SKY, nbkit.AMBER, nbkit.NAVY])
ax.set_title("Failure taxonomy (share of rows)"); ax.set_xlim(0, 1); plt.tight_layout(); plt.show()
tax
'''),
        md("""
No hallucination rows in either split, and the dominant non-pass type is *reasoning or wording* (the answer is
safe but misses a required phrase). Retrieval misses are 5% on dev but 15% on test (5 of 34 rows), which is the other thing to watch. For a compliance-adjacent use case, "safe but
incomplete" is the failure mode you want.

## 6. One trace, end to end
"""),
        code('''
traces = nbkit.load_jsonl(config.RESULTS / "agent/w3_dev_final/traces.jsonl")
first = traces[0]["trace_id"]
spans = pd.DataFrame([{"span": s["name"], "ms": s["duration_ms"]} for s in traces if s["trace_id"] == first])
spans.sort_values("ms", ascending=False)
'''),
        md("""
## 7. Compare any new run automatically

After a live run on v2, this cell finds runs that are not part of the saved set and reports them with intervals.
"""),
        code('''
known = set(runs.values())
new_runs = [p.name for p in sorted((config.RESULTS / "agent").iterdir()) if p.is_dir() and p.name not in known and (p / "answers.jsonl").exists()]
if not new_runs:
    print("No new runs yet. Run (needs the endpoint):\\n  RTO_EVAL_SET=v2 python -m rto_copilot.agent_eval --split dev --cohort v2")
for name in new_runs:
    r = nbkit.agent_run(name); cr2 = component_report(r)
    print(name, "n =", cr2["n"], "e2e =", cr2["end_to_end"], "CI95 =", cr2["ci95"], "components =", cr2["components"])
'''),
        md("""
## 8. Live: ask the agent a question (optional)
"""),
        code('''
if RUN_LIVE:
    from rto_copilot.serving import Copilot
    cp = Copilot.from_env()
    out = cp.ask("notebook", "Which forms do we need if there is no loan?", "2026-11-20")
    print(out["final"]["behavior"], "|", out["final"]["answer"][:600])
else:
    print("Skipped (RUN_LIVE is False). Case checks above ran without a model.")
'''),
    ])


# ============================================================================ 05 evaluation / observability / RAI
def nb05(build, md, code, SETUP):
    return build("05_Evaluation_Observability_RAI.ipynb", [
        md("""
# Week 4: evaluation, observability and responsible AI

**Questions.** How fast and how expensive is it? Where does time go? Is it safe: refusals, injection, PII, human control?
"""),
        code(SETUP),
        md("## 1. Speed: where does a turn spend its time?"),
        code('''
rep = config.RESULTS / "report/w4_final"
stage = pd.read_csv(rep / "stage_latency.csv")
pv = stage.pivot(index="run", columns="stage", values="avg_ms_per_query") / 1000
order = [c for c in ["planner", "executor.tools", "executor.compose", "critic", "other"] if c in pv.columns]
fig, ax = plt.subplots(figsize=(10, 3.8))
pv[order].plot.barh(stacked=True, ax=ax, color=[nbkit.NAVY, nbkit.SKY, nbkit.TEAL, nbkit.AMBER, nbkit.GREY][: len(order)])
ax.set_xlabel("Average seconds per query"); ax.set_ylabel(""); ax.set_title("Latency by stage")
plt.tight_layout(); plt.show()
stage[stage.run.str.startswith("TEST")][["stage", "avg_ms_per_query", "p95_ms", "share_of_turn"]]
'''),
        md("""
**So what.** About half of the wait is the model *writing* the answer (compose); tools and the critic are cheap.
To make it feel faster, cut tokens in the prompt (average 9.8K prompt tokens per query) or stream the answer; making
the tools faster will not move the needle.

## 2. Cost
"""),
        code('''
m = pd.read_csv(rep / "metrics.csv").set_index("run")
cols = ["items", "end_to_end_accuracy", "e2e_ci95_low", "e2e_ci95_high", "avg_prompt_tokens", "avg_completion_tokens", "cost_usd_per_1000_queries", "cost_usd_per_correct_answer", "avg_wall_latency_s", "p95_wall_latency_s"]
m[cols].round(4).T
'''),
        md("""
At the pricing assumptions in `data/pricing.json`, roughly **USD 1.3 per 1,000 queries**, and USD 0.002 to 0.004 per
*correct* answer. Cost per correct answer is the number that links accuracy and money: a cheaper but less accurate
setup is not automatically cheaper.

## 3. Failure taxonomy by category (test)
"""),
        code('''
tx = pd.read_csv(rep / "taxonomy_by_category.csv")
t = tx[tx.run.str.startswith("TEST")].set_index("category")[["PASS", "TOOL_CALL_ERROR", "RETRIEVAL_MISS", "HALLUCINATION", "REASONING_ERROR"]]
fig, ax = plt.subplots(figsize=(9, 4)); t.plot.barh(stacked=True, ax=ax, color=[nbkit.TEAL, nbkit.AMBER, nbkit.SKY, nbkit.RED, nbkit.NAVY])
ax.set_title("Test rows by failure type and category"); ax.set_ylabel(""); plt.tight_layout(); plt.show()
'''),
        md("## 4. Per-category accuracy with Wilson intervals (test)"),
        code('''
bc = pd.read_csv(rep / "by_category.csv"); bc = bc[bc.run.str.startswith("TEST")].set_index("category")
fig, ax = plt.subplots(figsize=(9, 4))
k = (bc.end_to_end * bc.n).round().astype(int)
nbkit.ci_bars(ax, bc.index, k, bc.n, horizontal=True, colors=[nbkit.TEAL] * len(bc))
ax.set_title("Test end-to-end by category (wide whiskers = small n)"); plt.tight_layout(); plt.show()
'''),
        md("""
## 5. Responsible AI checks

All numbers below come from the saved Week 4 run `rai/w4_final`.
"""),
        code('''
rai = json.load(open(config.RESULTS / "rai/w4_final/summary.json"))
ref = pd.DataFrame(rai["refusal"])
ref[["split", "expected", "n", "correct_behavior", "rate", "complied_with_unsafe", "over_refused"]]
'''),
        code('''
fig, ax = plt.subplots(figsize=(9, 3.8))
r = ref.copy(); r["label"] = r.split + ": " + r.expected.str.replace("_", " ").str.lower()
nbkit.ci_bars(ax, r.label, r.correct_behavior, r.n, horizontal=True, colors=[nbkit.NAVY if s == "test" else nbkit.TEAL for s in r.split])
ax.set_title("Refusal and escalation behaviour (unsafe requests complied with: " + str(int(pd.to_numeric(ref.complied_with_unsafe, errors="coerce").fillna(0).sum())) + ")")
plt.tight_layout(); plt.show()
'''),
        code('''
inj_off, inj_on, pii, hitl = rai["injection_guard_off"], rai["injection_guard_on"], rai["pii"], rai["hitl"]
pd.DataFrame({
    "Prompt injection, guard OFF": {"attacks": inj_off["attacks"], "reached the model": inj_off["exposed_to_llm"], "attacks that succeeded": inj_off["succeeded"]},
    "Prompt injection, guard ON": {"attacks": inj_on["attacks"], "reached the model": inj_on["exposed_to_llm"], "attacks that succeeded": inj_on["succeeded"]},
})
'''),
        code('''
print("PII probes detected:", pii["detected"], "of", pii["probes"], "| raw number reached the model:", pii["raw_in_llm_prompt"], "| in answer:", pii["raw_in_answer"], "| in session file:", pii["raw_in_session_file"], "| in traces:", pii["raw_in_traces"])
print("\\nHuman-in-the-loop:")
for k in ["drafts", "drafts_sent_true", "ready_status_answers", "ready_without_not_approval", "escalations", "escalations_without_human_reviewer"]:
    print(f"  {k:<38}{hitl[k]}")
print("  tools that send, file, pay or approve:", hitl["tools_that_send_file_pay_or_approve"] or "none")
'''),
        md("""
**Reading for the business.**

* The guard stops 8 of 10 planted injection attacks before they reach the model. Even with the guard off, none
  succeeded on this model, so this is defence in depth, not proof of immunity. Treat 10 attacks as a smoke test.
* A real Aadhaar, PAN or phone number is redacted before the model, the session file or the traces see it.
* Zero drafts were marked sent, and no tool can send, file, pay or approve. Human control is by construction.
* Test refusal behaviour is weaker on **ASK_CLARIFICATION** (1 of 3): the copilot sometimes answers with a caveat
  instead of asking. That is where the 19 clarification rows in v2 give a better read.
"""),
    ])


# ============================================================================ 06 UI
def nb06(build, md, code, SETUP):
    return build("06_UI_Demo.ipynb", [
        md("""
# The minimal web app

One FastAPI service and one static HTML page. No build step, no framework.

```bash
pip install -e .
cp .env.example .env      # fill in LLM_BASE_URL, LLM_API_KEY, LLM_MODEL
rto-serve                 # http://127.0.0.1:8000
```

Without the model keys the page still works for **case lookup** (deterministic checks and deadlines). Asking the
copilot needs the model.
"""),
        code(SETUP),
        code('''
from IPython.display import Image
Image(filename=str(ROOT / "docs/ui_screenshot.png"), width=820)
'''),
        md("""
**Design choices.** Everything a user needs on one screen: a case picker with status chips (left), a conversation
(right), a collapsible *Details* block for sources, findings and deadlines, and a clearly dashed **DRAFT, not sent**
box with a reviewer form. The review form records a decision only; nothing is sent, filed or approved.

## The API, exercised offline

This uses FastAPI's test client with a scripted stand-in for the model, so you can show the full request flow without
an endpoint.
"""),
        code('''
sys.path.insert(0, str(ROOT / "tests"))
import tempfile
from pathlib import Path
from fastapi.testclient import TestClient
from rto_copilot import serving
from rto_copilot.api import create_app
from test_agent import ScriptedClient, answer, make, plan

draft = {"type": "message", "content": "Dear Kabir Das, to complete the transfer of vehicle DEMO-REG-00013 we still need your signed Form 30. Thank you.", "sent": False}
scripted = answer(answer="Draft for your review (not sent): the buyer's Form 30 is missing. You decide whether to send it.",
                  findings=[{"code": "MISSING_FORM30", "evidence": "has_form_30=0"}], drafts=[draft])
state_dir = tempfile.mkdtemp()
cp = serving.Copilot(make(ScriptedClient([plan()], [scripted])), serving.Settings(rate_per_min=60, rate_burst=10), Path(state_dir))
client = TestClient(create_app(cp))
print(client.get("/api/health").json())
'''),
        code('''
view = client.get("/api/case/DEMO-013?today=2026-11-20").json()
print("Status:", view["check"]["case_status"], "| findings:", [f["code"] for f in view["check"]["findings"]])
pd.DataFrame(view["deadlines"]["deadlines"])[["rule_id", "due_date", "status"]]
'''),
        code('''
res = client.post("/api/ask", json={"session_id": "demo", "query": "Draft a message to the buyer asking for the missing item.", "case_id": "DEMO-013", "today": "2026-11-20"}).json()
print(res["final"]["behavior"], "|", res["final"]["answer"])
print("draft sent flag:", res["final"]["drafts"][0]["sent"])
'''),
        code('''
rv = client.post("/api/review", json={"session_id": "demo", "reviewer": "Asha", "decision": "approved_for_manual_use", "draft": draft}).json()
print(rv)
print("Review log line written to:", Path(state_dir, "review_log.jsonl").name, "| sent =", rv["sent"])
'''),
        md("""
Notice the review record carries `sent: false` and only a hash of the draft. The tool never sends anything: the
human decides, and the system records that they did.

## Safety behaviour you can show live
"""),
        code('''
r = client.post("/api/ask", json={"session_id": "demo2", "query": "My Aadhaar number is 2345 6789 0123, please save it."}).json()
print("PII redacted:", r["pii_redacted"]); print("Notice shown to user:", r["notices"])
bad = client.post("/api/review", json={"session_id": "demo", "reviewer": "Asha", "decision": "send_it", "draft": draft})
print("Invalid decision is rejected with HTTP", bad.status_code)
'''),
        md("""
## Docker (optional)

```bash
docker compose up --build        # serves on http://localhost:8000
```
"""),
    ])
