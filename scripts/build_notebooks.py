"""Generate the demo notebooks (notebooks/*.ipynb) and execute them offline so outputs are saved in the files.

    python scripts/build_notebooks.py            # build + execute
    python scripts/build_notebooks.py --no-run   # build only

Cells that need the LLM endpoint are guarded by RUN_LIVE (on by default, auto-off without a key) so the notebooks run end to end offline,
using the saved Week 1 to 4 results. Set RUN_LIVE = True once LLM_BASE_URL, LLM_API_KEY and LLM_MODEL are set.
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
sys.path.insert(0, str(Path(__file__).parent))

SETUP = '''import os, sys, pathlib, warnings
warnings.filterwarnings("ignore")
ROOT = pathlib.Path.cwd()
if not (ROOT / "src").exists():
    ROOT = ROOT.parent
os.environ.setdefault("RTO_HOME", str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import json, csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rto_copilot import config, nbkit
from rto_copilot.scoring import wilson_interval, margin_of_error, component_report
nbkit.style()
pd.set_option("display.max_colwidth", 90, "display.width", 140)
config.load_dotenv()
RUN_LIVE = True    # live cells call the model; they switch themselves off if the key is not in .env yet
if RUN_LIVE and config.missing_llm_env():
    print("RUN_LIVE switched off: set", ", ".join(config.missing_llm_env()), "in .env, then restart the kernel.")
    RUN_LIVE = False
print("Project root:", ROOT)'''


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text: str):
    return nbf.v4.new_code_cell(text.strip("\n"))


def build(name: str, cells: list):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    NB_DIR.mkdir(exist_ok=True)
    path = NB_DIR / name
    nbf.write(nb, path)
    return path


# ============================================================================ 00 overview
def nb00():
    return build("00_Overview.ipynb", [
        md("""
# RTO SmartTransfer Copilot: project overview

**What this is.** An agentic RAG copilot that helps buyers, sellers and financiers with the paperwork for a used-vehicle
ownership transfer inside one West Bengal RTO. It checks documents, explains rules with citations, calculates deadlines
and prepares drafts. It never sends, files, pays, signs or approves anything.

**Who this notebook is for.** Anyone you walk through the project: the business view first (does it work, how sure are
we), then the data science view (how it is built and measured). The other notebooks go week by week.

| Notebook | Question it answers |
|---|---|
| 01 Eval set v1 to v2 | How good is our yardstick, and what does a bigger set really buy? |
| 02 Prompt baseline | What does the model do on its own? |
| 03 RAG and retrieval | Does giving it the official sources help? |
| 04 Agentic copilot | What do tools, a critic and repair add? Why did test drop below dev? |
| 05 Evaluation, observability, responsible AI | Cost, speed, safety, failure modes |
| 06 UI demo | The minimal web app |
"""),
        code(SETUP),
        md("""
## The system on one page

The model never calculates a date or decides a finding on its own. Deterministic tools do that; the model plans,
explains and drafts. A rule critic checks every answer before the user sees it.
"""),
        code('''
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(11, 3.6)); ax.axis("off"); ax.set_xlim(0, 11); ax.set_ylim(0, 3.6)
def box(x, y, w, h, text, fc, tc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.1", fc=fc, ec="none"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tc, fontsize=9.5, fontweight="bold")
def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color=nbkit.GREY, lw=1.5))
box(0.1, 1.4, 1.3, 0.8, "User\\nquestion", nbkit.MINT, nbkit.NAVY)
box(1.9, 1.4, 1.4, 0.8, "Planner\\n(LLM, JSON plan)", nbkit.NAVY)
box(3.8, 1.4, 1.5, 0.8, "Executor\\ncalls tools, writes\\nanswer (LLM)", nbkit.TEAL)
box(5.8, 1.4, 1.4, 0.8, "Rule critic\\n(up to 2 revisions)", nbkit.NAVY)
box(7.6, 1.4, 1.4, 0.8, "Repair\\n(deterministic)", nbkit.TEAL)
box(9.4, 1.4, 1.5, 0.8, "Answer JSON\\n+ citations", nbkit.MINT, nbkit.NAVY)
for a, b in [(1.4, 1.9), (3.3, 3.8), (5.3, 5.8), (7.2, 7.6), (9.0, 9.4)]:
    arrow(a, 1.8, b, 1.8)
box(3.4, 2.7, 2.3, 0.7, "7 tools: lookup_case, check_case, loan_procedure,\\nresolve_date, compute_deadlines, search_corpus, get_source", nbkit.MINT, nbkit.NAVY)
arrow(4.5, 2.7, 4.5, 2.2)
box(3.4, 0.2, 2.3, 0.7, "Hybrid retrieval: BM25 + dense (bge-small)\\n+ cross-encoder rerank over official sources", nbkit.MINT, nbkit.NAVY)
arrow(4.5, 0.9, 4.5, 1.4)
box(7.0, 0.2, 3.9, 0.7, "Serving layer: PII redaction, rate limits, answer cache,\\ninjection screening, tracing, human review log", "#EEF2F5", nbkit.NAVY)
ax.set_title("Planner, executor, critic, repair", loc="left")
plt.show()
'''),
        md("""
## Headline: how far each stage got us

End-to-end accuracy is deliberately strict. A row only counts if **all six** checks pass: right behaviour, every required
fact present, no forbidden phrase, exact finding codes, valid citations, drafts marked not sent. The whiskers are 95%
Wilson intervals. They matter: with 34 test rows the interval is about 30 points wide.
"""),
        code('''
def k_n(path_summary, key="end_to_end_accuracy", n_key="items"):
    s = json.load(open(path_summary)); return round(s[key] * s[n_key]), s[n_key]

base = pd.read_csv(config.RESULTS / "baseline/week1_qwen3_8b/summary.csv").set_index("strategy")
rag = pd.read_csv(config.RESULTS / "rag/week2_qwen3_8b/summary.csv").set_index("strategy")
rows = [
    ("Week 1\\nzero-shot", round(base.loc["S0_ZERO_SHOT", "end_to_end_accuracy"] * 57), 57),
    ("Week 1\\nfew-shot", round(base.loc["S1_FEW_SHOT", "end_to_end_accuracy"] * 57), 57),
    ("Week 2\\nRAG", round(rag.loc["RAG_RETRIEVED", "end_to_end_accuracy"] * 57), 57),
    ("Week 3\\nagent (dev)", *k_n(config.RESULTS / "agent/w3_dev_final/summary.json")),
    ("Week 3\\nagent (test)", *k_n(config.RESULTS / "agent/w3_test_final/summary.json")),
]
fig, ax = plt.subplots(figsize=(9, 4))
nbkit.ci_bars(ax, [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows],
              colors=[nbkit.GREY, nbkit.GREY, nbkit.SKY, nbkit.TEAL, nbkit.NAVY])
ax.set_ylabel("End-to-end accuracy"); ax.set_title("Strict end-to-end accuracy by stage (Qwen3-8B)")
plt.show()
'''),
        md("""
**Read-out for the business.** The agent more than doubled the baseline on the dev split (about 18% to 56%). The test
number (32%) is lower, and two things are true at once: the test split is small (34 rows, so the plausible range is
19% to 49%), and the dev split was used for tuning, so it is a little optimistic. Notebook 01 explains how the
v2 evaluation set addresses both, and notebook 04 shows where the test rows lose their point.

**Plain-language reading of "56%".** More than half of all rows pass every one of six strict checks. Behaviour (picking
the right kind of response) is right in about 74% of rows, finding codes are right in 100%, and the gap is mostly
"a required fact was missing from the wording", not "the answer was wrong".
"""),
        md("""
## How to run this project

* **Offline (no model needed).** Every notebook runs top to bottom from saved results, the deterministic tools, the
  eval set and the corpus index. This is what you use for a demo on a laptop.
* **Live.** Paste your team key into `.env` (`LLM_API_KEY=`) and restart the kernel. `RUN_LIVE = True` is already set in
  the setup cell and switches itself off while the key is missing. Live cells are clearly marked.
"""),
        code('''
missing = config.missing_llm_env()
print("LLM endpoint configured:", not missing, "" if not missing else f"(missing: {', '.join(missing)})")
print("Eval set in use:", config.EVAL_VERSION)
for p in ["src/rto_copilot", "notebooks", "data/eval", "data/cases", "results", "docs", "tests"]:
    print(f"{p:<20}", len(list((ROOT / p).rglob('*'))), "files")
'''),
    ])


# ============================================================================ 01 eval set
def nb01():
    return build("01_Eval_Set_v1_to_v2.ipynb", [
        md("""
# Evaluation set: v1 (frozen) to v2 (196 rows)

**The question you asked:** dev accuracy was higher than test accuracy. Can a bigger eval set improve test accuracy?

**Short answer.** A bigger set does not raise accuracy by itself; it makes the number *trustworthy*. It tightens the
confidence interval, gives a fresh held-out test split nobody has tuned against, and exposes which categories are weak.
Improving the accuracy itself is the job of the agent changes in notebook 04. This notebook shows the evidence.
"""),
        code(SETUP),
        code('''
from rto_copilot import evalset
v1 = pd.read_csv(config.eval_path("v1")); v2 = pd.read_csv(config.eval_path("v2"))
cases = pd.read_csv(config.CASES, encoding="utf-8-sig")
print(f"v1: {len(v1)} rows | v2: {len(v2)} rows ({(v2.cohort=='v2').sum()} new) | cases: {len(cases)}")
pd.crosstab(v2.cohort, v2.split, margins=True)
'''),
        md("## 1. What changed in composition"),
        code('''
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
cat = pd.crosstab(v2.category, v2.cohort)[["v1", "v2"]]
cat.plot.barh(stacked=True, ax=axes[0], color=[nbkit.NAVY, nbkit.TEAL]); axes[0].set_title("Rows by category"); axes[0].set_ylabel("")
beh = pd.crosstab(v2.expected_behavior, v2.cohort)[["v1", "v2"]].sort_values("v1")
beh.plot.barh(stacked=True, ax=axes[1], color=[nbkit.NAVY, nbkit.TEAL]); axes[1].set_title("Rows by expected behaviour"); axes[1].set_ylabel("")
plt.tight_layout(); plt.show()
'''),
        md("""
Two deliberate additions: more **ABSTAIN_NO_SOURCE** rows (v1 had one; the specification lists several classes of
question the sources cannot answer) and more **escalate and refuse** rows, because refusal behaviour is where a
copilot is most likely to fail quietly.
"""),
        md("""
## 2. Why size matters: the margin of error

A pass rate measured on *n* rows is only known to within a band. At 50% the 95% half-width is about
`1.96 * sqrt(0.25 / n)`.
"""),
        code('''
n = np.arange(20, 260)
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(n, [margin_of_error(x) * 100 for x in n], color=nbkit.NAVY, lw=2)
pts = {"v1 test (34)": 34, "v1 dev (57)": 57, "v2 dev or test (~95)": 95, "v2 all (196)": 196}
for label, x in pts.items():
    y = margin_of_error(x) * 100
    ax.scatter([x], [y], color=nbkit.TEAL, zorder=3); ax.annotate(f"{label}\\n+/-{y:.1f} pts", (x, y), textcoords="offset points", xytext=(8, 8), fontsize=8.5)
ax.set_xlabel("Rows in the split"); ax.set_ylabel("95% margin of error (percentage points)"); ax.set_title("Precision of an accuracy number vs size of the eval set")
plt.show()
'''),
        code('''
rows = []
for label, n1, n2 in [("v1: dev 57 vs test 34", 57, 34), ("v2: dev 96 vs test 93", 96, 93)]:
    rows.append({"comparison": label, "smallest dev-vs-test gap we can reliably detect (pts)": round(nbkit.two_prop_min_detectable(n1, n2) * 100, 1)})
pd.DataFrame(rows)
'''),
        md("""
**So what.** With v1, a gap between dev and test had to exceed roughly 30 points before it could be told apart from
luck. With v2 it is roughly 20 points. That is better, not perfect: to detect a 10-point change reliably you need
several hundred rows per split. That is the honest limit of a small project and worth saying out loud.

## 3. Is the observed dev to test drop real?
"""),
        code('''
from scipy.stats import fisher_exact
dev, test = nbkit.agent_run("w3_dev_final"), nbkit.agent_run("w3_test_final")
a, b = sum(r["end_to_end"] for r in dev), sum(r["end_to_end"] for r in test)
odds, p = fisher_exact([[a, len(dev) - a], [b, len(test) - b]])
print(f"dev {a}/{len(dev)} = {a/len(dev):.1%}   test {b}/{len(test)} = {b/len(test):.1%}   Fisher exact p = {p:.3f}")
for name, r in (("dev", dev), ("test", test)):
    cr = component_report(r); print(name, "CI95", cr["ci95"], "| near-miss rate (failed exactly one check):", cr["near_miss_rate"])
'''),
        md("""
The drop is *borderline* significant (p about 0.03), the intervals overlap, and half the failing test rows missed by
exactly one check. Reasonable reading: part sampling noise, part tuning on dev, part a few hard test rows
(for example HYP and PROC questions that need exact phrasing). The 98 new rows let us separate those explanations.

## 4. The v2 test split is genuinely held out
"""),
        code('''
sources = {r["source_id"]: r for r in evalset.read_csv(config.SOURCES)}
rules = {r["rule_id"]: r for r in evalset.read_csv(config.RULES)}
case_map = {r["case_id"]: r for r in evalset.read_csv(config.CASES)}
rows_v2 = evalset.read_csv(config.eval_path("v2"))
errors = evalset.validate(rows_v2, case_map, sources, rules, "v2") + evalset.v1_immutability_errors(evalset.read_csv(config.eval_path("v1")), rows_v2)
print("Validator errors:", errors or "none")
leak = v2.merge(cases[["case_id", "split"]].rename(columns={"split": "case_split"}), on="case_id", how="left")
leak = leak.dropna(subset=["case_split"])   # rows without a case id, or an unknown id used on purpose (DEMO-201)
print(pd.crosstab(leak.split, leak.case_split))
print("\\nEvery v2 test row uses only test-split cases:", set(leak[(leak.cohort=="v2") & (leak.split=="test")].case_split) == {"test"})
print("Every v2 dev row uses only train/val cases:", set(leak[(leak.cohort=="v2") & (leak.split=="dev")].case_split) <= {"train", "val"})
'''),
        md("""
**How v1 is protected.** `eval_set_v1.csv` is hashed (SHA-256) and the validator refuses v2 if any v1 row changed.
The new rows are `DRAFT` (reviewer blank) until the team checks them; only then does `rto-evalset --freeze` write the
v2 hash and unlock the test split.

## 5. The 48 new synthetic cases
12 per loan scenario, split 5 train, 2 val, 5 test, with findings derived from the case fields by rule.
"""),
        code('''
new = cases[cases.case_id >= "DEMO-101"]
pd.crosstab(new.scenario, new.split)
'''),
        code('''
fnd = new.expected_findings.replace("", "NONE").str.split(";").explode().value_counts()
fig, ax = plt.subplots(figsize=(8, 3)); fnd.plot.barh(ax=ax, color=nbkit.TEAL); ax.set_title("Finding codes across the 48 new cases"); ax.set_ylabel(""); plt.show()
'''),
        md("## 6. A few new rows, as the scorer sees them"),
        code('''
show = v2[(v2.cohort == "v2")].groupby("category").head(1)[["query_id", "split", "expected_behavior", "query", "gold_answer"]]
show.reset_index(drop=True)
'''),
        md("""
## 7. Sanity check: the scorer accepts the gold

Running the *oracle* (a mock that returns each row's own gold answer) must score 100%. If it does not, the row or the
scorer is broken.
"""),
        code('''
from rto_copilot import scoring, llm
known = {r["source_id"] for r in evalset.read_csv(config.SOURCES)}
res = [scoring.score(r, scoring.gold_response(r), known) for r in rows_v2]
print(f"Oracle end-to-end on all {len(res)} v2 rows: {sum(x['end_to_end'] for x in res)}/{len(res)}")
'''),
        md("""
## Review checklist for the team (before freezing v2)

1. Each new gold answer matches its cited source (PROC, HYP, CHK, DDL non-case rows).
2. Behaviour labels on the judgement calls listed in `docs/EVAL_V2_CHANGELOG.md`.
3. Refusal rows match spec section 3.
4. Then: `rto-evalset --version v2 --freeze`.
"""),
    ])


if __name__ == "__main__":
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor
    import notebooks_defs  # noqa: F401  (adds the remaining notebooks)
    paths = [nb00(), nb01()] + notebooks_defs.build_rest(build, md, code, SETUP)
    if "--no-run" not in sys.argv:
        for p in paths:
            nb = nbformat.read(p, as_version=4)
            ExecutePreprocessor(timeout=600, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(NB_DIR)}})
            nbformat.write(nb, p)
            print("executed", p.name)
