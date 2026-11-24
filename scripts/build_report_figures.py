"""Figures for the technical report and the failure-taxonomy PDF. Reads saved results only. Output: report/figures/*.png"""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rto_copilot import config, nbkit  # noqa: E402
from rto_copilot.scoring import wilson_interval  # noqa: E402

nbkit.style()
NAVY, TEAL, MINT, GREY, AMBER, RED, SKY = nbkit.NAVY, nbkit.TEAL, nbkit.MINT, nbkit.GREY, nbkit.AMBER, nbkit.RED, nbkit.SKY
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
A = json.loads((ROOT / "report" / "analysis.json").read_text())
REP = config.RESULTS / "report" / "w4_final"


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- 1 results ladder
def ladder():
    labels = ["Week 1\nzero-shot", "Week 1\nfew-shot", "Week 2\nRAG", "Week 3 agent\n(dev)", "Week 3 agent\n(held-out test)"]
    s = [10, 14, 10, 32, 11]; n = [57, 57, 57, 57, 34]
    # S0 17.5% = 10/57, S1 24.6% = 14/57, RAG 17.5% = 10/57
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    nbkit.ci_bars(ax, labels, s, n, colors=[GREY, GREY, GREY, TEAL, NAVY])
    ax.set_ylabel("End-to-end accuracy"); ax.set_title("What each step added (same scorer, Qwen3-8B)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    save(fig, "f01_ladder.png")


# ---------------------------------------------------------------- 2 agent dev runs
def agent_runs():
    names = ["v1\nfirst agent", "v2\nscope table", "v3\nrule sources", "v3\nrule critic", "v4\nloan tool", "v5\nsearch + loan", "v6\nfinal"]
    vals = [38.6, 35.1, 38.6, 40.4, 49.1, 54.4, 56.1]
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    ax.plot(range(len(vals)), vals, marker="o", color=TEAL, lw=2)
    for i, v in enumerate(vals):
        ax.text(i, v + 2, f"{v:.1f}", ha="center", fontsize=8)
    ax.axhline(24.6, color=GREY, ls="--", lw=1); ax.text(0, 26, "Week 1 few-shot 24.6", fontsize=8, color=GREY)
    ax.scatter([6.25], [32.4], color=NAVY, zorder=5); ax.text(6.35, 29, "test 32.4", fontsize=8, color=NAVY, ha="right")
    ax.set_xticks(range(len(vals))); ax.set_xticklabels(names, fontsize=8); ax.set_ylim(15, 66)
    ax.set_ylabel("Dev end-to-end (%)"); ax.set_title("Dev accuracy by agent version")
    save(fig, "f02_agent_runs.png")


# ---------------------------------------------------------------- 3 prompt strategies
def strategies():
    d = pd.read_csv(config.RESULTS / "baseline" / "week1_qwen3_8b" / "summary.csv")
    names = {"S0_ZERO_SHOT": "Zero-shot", "S1_FEW_SHOT": "Few-shot +\nstructured output", "S2_SELF_CONSISTENCY": "Self-consistency\n(3 samples)", "S3_DECOMPOSITION": "Decomposition\n(2 steps)"}
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8), gridspec_kw={"width_ratios": [1.3, 1]})
    x = np.arange(len(d)); w = 0.38
    axes[0].bar(x - w / 2, d.end_to_end_accuracy, w, color=TEAL, label="End-to-end")
    axes[0].bar(x + w / 2, d.behavior_accuracy, w, color=NAVY, label="Behaviour")
    for i, (a, b) in enumerate(zip(d.end_to_end_accuracy, d.behavior_accuracy)):
        axes[0].text(i - w / 2, a + 0.01, f"{a:.0%}", ha="center", fontsize=7); axes[0].text(i + w / 2, b + 0.01, f"{b:.0%}", ha="center", fontsize=7)
    axes[0].set_xticks(x); axes[0].set_xticklabels([names[s] for s in d.strategy], fontsize=7); axes[0].set_ylim(0, 0.85); axes[0].legend(frameon=False, fontsize=7); axes[0].set_title("Accuracy")
    axes[1].bar(x, d.avg_tokens / 1000, color=[GREY, TEAL, GREY, GREY]); axes[1].set_xticks(x); axes[1].set_xticklabels(["S0", "S1", "S2", "S3"]); axes[1].set_title("Tokens per query (k)")
    for i, v in enumerate(d.avg_tokens / 1000): axes[1].text(i, v + 0.15, f"{v:.1f}", ha="center", fontsize=8)
    fig.tight_layout(); save(fig, "f03_strategies.png")


# ---------------------------------------------------------------- 4 chunking x method
def chunking():
    chunks = ["Whole section\n(721)", "w128\n(2,308)", "w256\n(1,317)", "w384\n(1,029)"]
    data = {"BM25": [.400, .418, .389, .400], "Dense": [.292, .513, .408, .393], "Hybrid (RRF)": [.498, .571, .505, .509], "Hybrid + rerank": [.487, .509, .487, .462]}
    fig, ax = plt.subplots(figsize=(6.4, 2.9)); w = 0.2; x = np.arange(4)
    for i, (k, v) in enumerate(data.items()):
        ax.bar(x + (i - 1.5) * w, v, w, label=k, color=[GREY, SKY, TEAL, NAVY][i])
    ax.set_xticks(x); ax.set_xticklabels(chunks, fontsize=8); ax.set_ylabel("Recall@5"); ax.set_ylim(0, 0.7)
    ax.set_title("Chunking and retrieval method (plain queries, dev)"); ax.legend(frameon=False, fontsize=7, ncol=4, loc="upper left")
    save(fig, "f04_chunking.png")


def retrieval():
    methods = ["BM25", "Dense", "Hybrid", "Hybrid +\nrerank", "Hybrid + rerank\n(RRF-fused)"]
    plain = [.418, .513, .571, .509, .525]; case = [.505, .563, .621, .654, .652]
    fig, ax = plt.subplots(figsize=(6.4, 2.8)); x = np.arange(5); w = 0.38
    ax.bar(x - w / 2, plain, w, color=GREY, label="plain query"); ax.bar(x + w / 2, case, w, color=TEAL, label="case-aware query")
    for i in range(5): ax.text(i + w / 2, case[i] + 0.01, f"{case[i]:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=8); ax.set_ylabel("Recall@5"); ax.set_ylim(0, 0.8); ax.legend(frameon=False, fontsize=8)
    ax.set_title("Case-aware queries and the reranker (w128, 46 dev queries)")
    save(fig, "f05_retrieval.png")


def diagnostic():
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6))
    axes[0].bar(["Retrieved\nsources", "Gold sources\nforced in"], [.175, .175], color=[TEAL, NAVY]); axes[0].set_ylim(0, .4)
    for i in range(2): axes[0].text(i, .185, "17.5%", ha="center", fontsize=9)
    axes[0].set_title("End-to-end, same 57 rows")
    causes = ["Generation miss\n(gold in context)", "No retrieval\nneeded", "Retrieval and\ngeneration", "Retrieval miss\n(fixed by gold)"]
    axes[1].barh(causes[::-1], [26, 13, 8, 0][::-1], color=[GREY, SKY, TEAL, NAVY][::-1])
    for i, v in enumerate([0, 8, 13, 26]): axes[1].text(v + 0.5, i, str(v), va="center", fontsize=8)
    axes[1].set_title("Why 47 RAG answers failed"); axes[1].tick_params(axis="y", labelsize=7)
    fig.tight_layout(); save(fig, "f06_diagnostic.png")


# ---------------------------------------------------------------- 5 latency stages
def latency():
    d = pd.read_csv(REP / "stage_latency.csv")
    runs = ["Dev: final (rule critic)", "TEST split (final config)", "Dev ablation: LLM critic"]
    lab = ["Dev, final", "Test, final", "Dev, LLM critic"]
    stages = [("planner", SKY, "Planner"), ("executor.tools", TEAL, "Tools"), ("executor.compose", NAVY, "Compose (model)"), ("critic", AMBER, "Critic"), ("other", GREY, "Other")]
    fig, ax = plt.subplots(figsize=(6.4, 2.4)); left = np.zeros(3)
    for st, c, name in stages:
        v = np.array([d[(d.run == r) & (d.stage == st)].avg_ms_per_query.iloc[0] / 1000 for r in runs])
        ax.barh(lab, v, left=left, color=c, label=name)
        for i, (l, x) in enumerate(zip(left, v)):
            if x > 1.2: ax.text(l + x / 2, i, f"{x:.1f}", ha="center", va="center", color="white", fontsize=8)
        left += v
    ax.invert_yaxis(); ax.set_xlabel("Seconds per query"); ax.legend(frameon=False, fontsize=7, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.2))
    save(fig, "f07_latency.png")


# ---------------------------------------------------------------- 6 taxonomy
def taxonomy():
    d = pd.read_csv(REP / "taxonomy_by_category.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), sharex=False)
    for ax, run, title in zip(axes, ["Dev: final (rule critic)", "TEST split (final config)"], ["Dev (57)", "Held-out test (34)"]):
        s = d[d.run == run].set_index("category")
        order = s.assign(rate=s.PASS / s.n).sort_values("rate").index
        left = np.zeros(len(order))
        for col, c, name in [("PASS", TEAL, "Pass"), ("REASONING_ERROR", RED, "Reasoning error"), ("RETRIEVAL_MISS", AMBER, "Retrieval miss"), ("TOOL_CALL_ERROR", SKY, "Tool-call error"), ("HALLUCINATION", NAVY, "Hallucination")]:
            v = s.loc[order, col].values
            ax.barh(order, v, left=left, color=c, label=name)
            for i, (l, x) in enumerate(zip(left, v)):
                if x: ax.text(l + x / 2, i, str(x), ha="center", va="center", color="white", fontsize=7)
            left += v
        ax.set_title(title, fontsize=10)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=7, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(); save(fig, "f08_taxonomy.png")


def category_ci():
    d = pd.read_csv(REP / "by_category.csv")
    dev = d[d.run == "Dev: final (rule critic)"].set_index("category"); test = d[d.run == "TEST split (final config)"].set_index("category")
    cats = list(dev.sort_values("end_to_end").index)
    fig, ax = plt.subplots(figsize=(6.4, 2.9)); x = np.arange(len(cats)); w = 0.38
    ax.bar(x - w / 2, dev.loc[cats, "end_to_end"], w, color=TEAL, label="dev"); ax.bar(x + w / 2, [test.loc[c, "end_to_end"] if c in test.index else 0 for c in cats], w, color=NAVY, label="test")
    ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=8); ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0)); ax.legend(frameon=False, fontsize=8)
    ax.set_title("End-to-end by query category (small n per bar)"); save(fig, "f09_category.png")


def confusion():
    beh = ["ANSWER", "ANSWER_WITH_CAVEAT", "ASK_CLARIFICATION", "ESCALATE_HUMAN_REVIEW", "REFUSE_OUT_OF_SCOPE", "REFUSE_UNSAFE", "ABSTAIN_NO_SOURCE"]
    short = ["Answer", "Caveat", "Clarify", "Escalate", "Refuse\nout of scope", "Refuse\nunsafe", "Abstain"]
    m = np.zeros((len(beh), len(beh)))
    for key in ("dev", "test"):
        for e, p, n in A["runs"][key]["confusion"]:
            if e in beh and p in beh: m[beh.index(e), beh.index(p)] += n
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.imshow(m, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("t", ["#ffffff", TEAL]), vmin=0, vmax=m.max())
    for i in range(len(beh)):
        for j in range(len(beh)):
            if m[i, j]: ax.text(j, i, int(m[i, j]), ha="center", va="center", color="white" if m[i, j] > m.max() * .5 else NAVY, fontsize=9)
    ax.set_xticks(range(len(beh))); ax.set_xticklabels(short, fontsize=7); ax.set_yticks(range(len(beh))); ax.set_yticklabels(short, fontsize=7)
    ax.set_xlabel("What the copilot did"); ax.set_ylabel("What it should do"); ax.set_title("Behaviour confusion matrix (dev + test, 91 queries)", fontsize=10)
    ax.grid(False); save(fig, "f10_confusion.png")


def subcauses():
    s = A["subcauses"]; kinds = sorted({k for v in s.values() for k in v}, key=lambda k: -(s["dev"].get(k, 0) + s["test"].get(k, 0)))
    fig, ax = plt.subplots(figsize=(6.4, 2.7)); y = np.arange(len(kinds)); w = 0.38
    ax.barh(y - w / 2, [s["dev"].get(k, 0) for k in kinds], w, color=TEAL, label="dev (25 failures)"); ax.barh(y + w / 2, [s["test"].get(k, 0) for k in kinds], w, color=NAVY, label="test (23 failures)")
    ax.set_yticks(y); ax.set_yticklabels(kinds, fontsize=8); ax.invert_yaxis(); ax.legend(frameon=False, fontsize=8, loc="lower right"); ax.set_xlabel("Failed queries")
    ax.set_title("Inside the reasoning errors: what exactly went wrong")
    save(fig, "f11_subcauses.png")


# ---------------------------------------------------------------- diagrams
def box(ax, x, y, w, h, text, fc="#ffffff", ec=NAVY, fs=8, bold=False, tc=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc or NAVY, fontweight="bold" if bold else "normal", linespacing=1.25)


def arrow(ax, a, b, color=GREY, style="-|>", ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=9, color=color, lw=1.1, linestyle=ls, connectionstyle=f"arc3,rad={rad}"))


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h)); ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off"); ax.grid(False); return fig, ax


def system_diagram():
    fig, ax = canvas(7.0, 4.3)
    box(ax, 0.1, 3.3, 1.7, 0.7, "Web page\n(model picker, sources,\ntrace link)", MINT, TEAL)
    box(ax, 2.2, 3.3, 1.9, 0.7, "FastAPI\n/api/ask  /api/models\n/dashboard", MINT, TEAL)
    box(ax, 4.5, 3.3, 2.4, 0.7, "Serving layer\ninput check, PII redaction,\nrate limit, cache, one-at-a-time gate", "#fff", NAVY, 7)
    arrow(ax, (1.8, 3.65), (2.2, 3.65)); arrow(ax, (4.1, 3.65), (4.5, 3.65))
    box(ax, 0.1, 1.7, 3.3, 1.3, "AGENT\nplanner  ->  executor (tools, compose)  ->  rule critic\n(max 2 revisions, then deterministic repair)\nsession state, token and call caps", "#fff", NAVY, 8, True)
    arrow(ax, (4.9, 3.3), (3.1, 3.0))
    box(ax, 3.9, 2.0, 3.0, 0.8, "7 tools\nlookup_case  check_case  loan_procedure\nresolve_date  compute_deadlines\nsearch_corpus  get_source", "#fff", TEAL, 7)
    arrow(ax, (3.4, 2.4), (3.9, 2.4))
    box(ax, 3.9, 0.95, 3.0, 0.8, "RAG over 721 official documents\n2,308 chunks: BM25 + bge-small\nRRF fusion + bge-reranker-base", "#fff", TEAL, 7)
    arrow(ax, (5.4, 2.0), (5.4, 1.75))
    box(ax, 0.1, 0.1, 1.9, 0.8, "Model profiles\nQwen3-8B (course)\nLlama 3.1 8B (local)\nHF router", "#fff", NAVY, 7)
    arrow(ax, (1.2, 1.7), (1.2, 0.9), style="<|-|>")
    box(ax, 2.3, 0.1, 2.0, 0.8, "OpenTelemetry spans\ntraces.jsonl\n(optional Langfuse / OTLP)", "#fff", AMBER, 7)
    arrow(ax, (2.4, 1.7), (3.0, 0.9), color=AMBER, ls="--")
    box(ax, 4.6, 0.1, 2.3, 0.8, "Trace dashboard\nHTML, built from the spans", MINT, TEAL, 7)
    arrow(ax, (4.3, 0.5), (4.6, 0.5), color=AMBER, ls="--")
    save(fig, "d01_system.png")


def rag_diagram():
    fig, ax = canvas(7.0, 2.7)
    box(ax, 0.05, 1.7, 1.3, 0.8, "18 official\nsources\n(PDF, web)", MINT, TEAL, 7)
    box(ax, 1.55, 1.7, 1.3, 0.8, "721 documents\nsection / rule /\nFAQ entry", "#fff", NAVY, 7)
    box(ax, 3.05, 1.7, 1.4, 0.8, "128-word chunks\n32 overlap, title\nprepended: 2,308", "#fff", NAVY, 7)
    box(ax, 4.65, 2.05, 1.2, 0.5, "BM25 index", "#fff", TEAL, 7); box(ax, 4.65, 1.5, 1.2, 0.5, "bge-small\nembeddings", "#fff", TEAL, 7)
    for a, b in [((1.35, 2.1), (1.55, 2.1)), ((2.85, 2.1), (3.05, 2.1)), ((4.45, 2.2), (4.65, 2.3)), ((4.45, 2.0), (4.65, 1.75))]: arrow(ax, a, b)
    box(ax, 0.05, 0.1, 1.5, 0.9, "Question\n+ case facts\n(case-aware query)", MINT, TEAL, 7)
    box(ax, 1.85, 0.1, 1.5, 0.9, "Hybrid search\nBM25 + dense\nfused by RRF", "#fff", NAVY, 7)
    box(ax, 3.65, 0.1, 1.5, 0.9, "Cross-encoder\nrerank, fused\nwith RRF", "#fff", NAVY, 7)
    box(ax, 5.45, 0.1, 1.5, 0.9, "8 chunks, max 2\nper source, to the\ngenerator + cite", MINT, TEAL, 7)
    for a, b in [((1.55, 0.55), (1.85, 0.55)), ((3.35, 0.55), (3.65, 0.55)), ((5.15, 0.55), (5.45, 0.55))]: arrow(ax, a, b)
    arrow(ax, (5.0, 1.5), (2.7, 1.0), color=TEAL, ls="--")
    ax.text(5.9, 1.45, "built once\n(offline)", fontsize=7, color=GREY, ha="left")
    save(fig, "d02_rag.png")


def agent_diagram():
    fig, ax = canvas(7.0, 2.9)
    box(ax, 0.05, 1.8, 1.0, 0.7, "User turn\n+ session\nstate", MINT, TEAL, 7)
    box(ax, 1.3, 1.8, 1.5, 0.7, "PLANNER\n1 model call (JSON)\nbehaviour, role, case,\ndate, tool steps", "#fff", NAVY, 7, True)
    box(ax, 3.05, 1.8, 1.7, 0.7, "EXECUTOR: tools\n(up to 8 calls, deduped,\nerrors returned to model)", "#fff", TEAL, 7, True)
    box(ax, 5.0, 1.8, 1.9, 0.7, "EXECUTOR: compose\n1 model call, answer from\nchunks + tool results", "#fff", NAVY, 7, True)
    for a, b in [((1.05, 2.15), (1.3, 2.15)), ((2.8, 2.15), (3.05, 2.15)), ((4.75, 2.15), (5.0, 2.15))]: arrow(ax, a, b)
    box(ax, 5.0, 0.7, 1.9, 0.75, "CRITIC (rules)\nfindings, dates, citations,\nscope, drafts, identity numbers", "#fff", AMBER, 7, True)
    arrow(ax, (5.5, 1.8), (5.5, 1.45))
    arrow(ax, (6.4, 1.45), (6.4, 1.8), color=AMBER, ls="--")
    box(ax, 2.6, 0.7, 2.0, 0.75, "REPAIR (deterministic)\nafter 2 revisions: make the\nanswer match the tools", "#fff", AMBER, 7)
    arrow(ax, (5.0, 1.05), (4.6, 1.05))
    ax.text(6.47, 1.6, "revise", fontsize=6, color=AMBER, ha="left", va="center")
    ax.text(5.43, 1.6, "check", fontsize=6, color=GREY, ha="right", va="center")
    box(ax, 0.05, 0.7, 2.2, 0.75, "FINAL ANSWER\nbehaviour, answer, citations,\nfindings, deadlines, DRAFT", MINT, TEAL, 7, True)
    arrow(ax, (2.6, 1.05), (2.25, 1.05))
    ax.text(0.05, 0.3, "Session JSON updated: case id, role, transfer date, last turns", fontsize=7, color=GREY)
    ax.text(0.05, 0.05, "Every box is an OpenTelemetry span. Caps: 7 model calls, 8 tool calls, 45,000 tokens, 300 s.", fontsize=7, color=GREY)
    save(fig, "d03_agent.png")


for f in (ladder, agent_runs, strategies, chunking, retrieval, diagnostic, latency, taxonomy, category_ci, confusion, subcauses, system_diagram, rag_diagram, agent_diagram):
    f()
print(sorted(p.name for p in OUT.iterdir()))
