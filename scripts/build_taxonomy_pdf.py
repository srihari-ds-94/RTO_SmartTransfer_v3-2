"""Failure Taxonomy PDF. Reads report/analysis.json and report/figures. Output: report/Failure_Taxonomy_Synapse6.pdf"""
import json
from pathlib import Path
from PIL import Image as PILImage
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Table, TableStyle,
                                Image, PageBreak, KeepTogether)

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
A = json.loads((ROOT / "report" / "analysis.json").read_text())
F = "/usr/share/fonts/truetype/crosextra/"
pdfmetrics.registerFont(TTFont("Body", F + "Carlito-Regular.ttf"))
pdfmetrics.registerFont(TTFont("Body-B", F + "Carlito-Bold.ttf"))
pdfmetrics.registerFont(TTFont("Body-I", F + "Carlito-Italic.ttf"))
pdfmetrics.registerFont(TTFont("Head", F + "Caladea-Bold.ttf"))
pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-B", italic="Body-I", boldItalic="Body-B")

NAVY, TEAL, MINT, GREY, LIGHT = colors.HexColor("#0F2A4A"), colors.HexColor("#0E8C6F"), colors.HexColor("#D6EEE6"), colors.HexColor("#5F6B7A"), colors.HexColor("#F4F7F9")
S = {
    "body": ParagraphStyle("body", fontName="Body", fontSize=10.5, leading=14.5, textColor=colors.HexColor("#1B2733"), spaceAfter=6),
    "h1": ParagraphStyle("h1", fontName="Head", fontSize=16, leading=20, textColor=NAVY, spaceBefore=10, spaceAfter=6, keepWithNext=1),
    "h2": ParagraphStyle("h2", fontName="Head", fontSize=12.5, leading=16, textColor=TEAL, spaceBefore=8, spaceAfter=4, keepWithNext=1),
    "cap": ParagraphStyle("cap", fontName="Body-I", fontSize=9, leading=12, textColor=GREY, spaceAfter=8),
    "cell": ParagraphStyle("cell", fontName="Body", fontSize=9.2, leading=11.8),
    "cellb": ParagraphStyle("cellb", fontName="Body-B", fontSize=9.2, leading=11.8, textColor=NAVY),
    "th": ParagraphStyle("th", fontName="Body-B", fontSize=9.2, leading=11.8, textColor=colors.white),
    "title": ParagraphStyle("title", fontName="Head", fontSize=26, leading=31, textColor=NAVY, spaceAfter=4),
    "sub": ParagraphStyle("sub", fontName="Body", fontSize=13, leading=17, textColor=TEAL, spaceAfter=14),
    "bul": ParagraphStyle("bul", fontName="Body", fontSize=10.5, leading=14.5, leftIndent=14, bulletIndent=2, spaceAfter=3),
}
P = lambda t, s="body": Paragraph(t, S[s])
B = lambda t: Paragraph(t, S["bul"], bulletText="•")


def img(name, width_mm):
    w, h = PILImage.open(FIG / name).size
    wd = width_mm * mm
    return Image(str(FIG / name), width=wd, height=wd * h / w)


def table(rows, widths, head=True, zebra=True):
    data = [[Paragraph(str(c), S["th" if (head and i == 0) else "cell"]) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * mm for w in widths], repeatRows=1 if head else 0)
    st = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#C9D3DC")),
          ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5), ("TOPPADDING", (0, 0), (-1, -1), 3.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5)]
    if head:
        st.append(("BACKGROUND", (0, 0), (-1, 0), NAVY))
    if zebra:
        for i in range(1, len(rows)):
            if i % 2 == 0:
                st.append(("BACKGROUND", (0, i), (-1, i), LIGHT))
    t.setStyle(TableStyle(st))
    return t


def footer(c, d):
    c.saveState()
    c.setFont("Body", 8.5); c.setFillColor(GREY)
    c.drawString(18 * mm, 10 * mm, "RTO SmartTransfer Copilot | Failure Taxonomy | Team Synapse6")
    c.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {d.page}")
    c.setStrokeColor(MINT); c.setLineWidth(1.2); c.line(18 * mm, 14 * mm, A4[0] - 18 * mm, 14 * mm)
    c.restoreState()


R = A["runs"]
def cnt(run, ft):
    return sum(1 for f in R[run]["failures"] if f["failure_type"] == ft)
def pct(a, b):
    return f"{100 * a / b:.0f}%"

story = []
# ------------------------------------------------------------------ cover block
story += [Spacer(1, 18 * mm), P("Failure Taxonomy", "title"), P("RTO SmartTransfer Copilot, Track B", "sub"),
          P("Executive Post Graduate Programme in Generative AI and Agentic AI<br/>Capstone project, Team Synapse6"),
          P("Bhawani Jaswal, Binoj Varghese, Chandra Bhushan Kumar, Shubham Kumar, SRIHARI R M, Srikanth"), Spacer(1, 8 * mm)]
dn, dp, tn, tp = R["dev"]["n"], R["dev"]["passed"], R["test"]["n"], R["test"]["passed"]
story += [P("Summary", "h1"),
          P(f"Every query the agent fails is given exactly one primary failure class. This document defines the classes, shows how many failures fall in each, "
            f"gives real examples from the saved runs, and lists the fixes. On the development split the agent passes {dp} of {dn} queries ({pct(dp, dn)}). "
            f"On the held-out test split it passes {tp} of {tn} ({pct(tp, tn)}). The {tn - tp} test failures split as "
            f"{cnt('test', 'REASONING_ERROR')} reasoning errors, {cnt('test', 'RETRIEVAL_MISS')} retrieval misses, {cnt('test', 'TOOL_CALL_ERROR')} tool-call error and {cnt('test', 'HALLUCINATION')} hallucinations."),
          B("<b>Most failures are reasoning errors, and most of those are near misses.</b> In 14 of the 23 test failures the agent chose the right behaviour (answer, caveat, refuse) but left out a required fact or phrase."),
          B("<b>Retrieval is not the main cause.</b> Forcing the gold source into the context gave no gain in Week 2 (17.5% both ways). Gold was already in context for 26 of 47 failed RAG answers."),
          B("<b>The classes are a heuristic.</b> A class tells you where to look first. It does not prove the cause. Section 6 lists what the labels cannot show."),
          Spacer(1, 3 * mm)]
rows = [["Run", "Queries", "Passed", "Reasoning", "Retrieval miss", "Tool call", "Hallucination"]]
names = {"dev": "Dev, final config", "test": "Test, final config (held out)", "dev_llm_critic": "Dev, LLM critic ablation", "dev_faults": "Dev, 30% injected tool faults"}
for k in ["dev", "test", "dev_llm_critic", "dev_faults"]:
    rows.append([names[k], R[k]["n"], f"{R[k]['passed']} ({pct(R[k]['passed'], R[k]['n'])})", cnt(k, "REASONING_ERROR"), cnt(k, "RETRIEVAL_MISS"), cnt(k, "TOOL_CALL_ERROR"), cnt(k, "HALLUCINATION")])
story += [table(rows, [52, 18, 24, 21, 24, 18, 24]), P("Table 1. Failures by primary class. Same scorer, same Qwen3-8B model, rule-based critic unless stated. Dev n=57, test n=34.", "cap")]
story.append(PageBreak())

# ------------------------------------------------------------------ definitions
story += [P("1. Definitions and decision order", "h1"),
          P("The course asks for four classes: retrieval miss, reasoning error, tool-call error and hallucination. Each failed query is checked against them in the order below, and the first match wins. The order puts the most mechanical causes first, so a query is only called a reasoning error when nothing more specific explains it.")]
rows = [["Order", "Class", "Rule applied to a failed query", "Typical symptom"],
        ["1", "TOOL_CALL_ERROR", "A tool returned a ToolError, raised an exception, or the run ended with a malformed-JSON or loop stop. Includes injected faults.", "Invalid date such as 31/11/2026; unknown case id; tool timeout."],
        ["2", "RETRIEVAL_MISS", "A gold source id for the query is not in the context the model saw (retrieved chunks plus any get_source result).", "The Form 34 chunk was not among the 6 sources shown."],
        ["3", "HALLUCINATION", "The answer cites a source id that was never retrieved, or states a date that appears in neither the query nor any tool output.", "A citation id with no matching chunk."],
        ["4", "REASONING_ERROR", "Residual. Evidence was available and tools worked, but the answer still failed one of the six checks.", "Missing required fact; caveat added to a plain answer; wrong refusal type."]]
story += [table(rows, [14, 34, 70, 56]), P("Table 2. Decision order. Implemented in the scorer, then re-derived in scripts/analyse_failures.py from the saved answers.", "cap")]
story += [P("Six checks decide pass or fail", "h2"),
          P("A query passes only when all six hold: (1) the behaviour label matches (ANSWER, ANSWER_WITH_CAVEAT, ASK_CLARIFICATION, REFUSE_OUT_OF_SCOPE, REFUSE_UNSAFE, ESCALATE_HUMAN_REVIEW, ABSTAIN_NO_SOURCE); (2) every must-include fact or phrase is present; (3) no forbidden phrase appears; (4) the finding codes match exactly; (5) every citation id is valid and was retrieved; (6) every draft has sent=false. End-to-end accuracy is the AND of the six, which is why it sits well below behaviour accuracy (56.1% against 73.7% on dev)."),
          P("Mapping to the course classes", "h2")]
rows = [["Course class", "Our class", "Also covers"],
        ["Retrieval miss", "RETRIEVAL_MISS", "Gold source absent from context. It marks where to look, it does not prove retrieval caused the failure."],
        ["Reasoning error", "REASONING_ERROR", "Wrong behaviour, missing fact, wrong refusal type, over-caveating, wrong clarification decision."],
        ["Tool-call error", "TOOL_CALL_ERROR", "Tool errors surfaced to the model, malformed JSON that survived the re-ask, loop or cap stops."],
        ["Hallucination", "HALLUCINATION", "Invalid or unretrieved citation id, unsupported date. Unretrieved citations are also stripped before the answer is shown."]]
story += [table(rows, [34, 38, 102])]

# ------------------------------------------------------------------ results
story += [P("2. What the numbers show", "h1"),
          img("f08_taxonomy.png", 150),
          P("Figure 1. Primary failure class per run. Reasoning errors dominate in every run.", "cap")]
dev_f, test_f = dn - dp, tn - tp
story += [P(f"On dev, {cnt('dev', 'REASONING_ERROR')} of {dev_f} failures ({pct(cnt('dev', 'REASONING_ERROR'), dev_f)}) are reasoning errors and {cnt('dev', 'RETRIEVAL_MISS')} are retrieval misses. "
            f"On test, {cnt('test', 'REASONING_ERROR')} of {test_f} ({pct(cnt('test', 'REASONING_ERROR'), test_f)}) are reasoning errors, {cnt('test', 'RETRIEVAL_MISS')} ({pct(cnt('test', 'RETRIEVAL_MISS'), test_f)}) retrieval misses and one is a tool-call error. "
            "The agent completed every task in every run (task completion 100%), so none of these failures is a crash or a timeout. They are wrong or incomplete answers."),
          P("Two ablations change the mix. Turning on the LLM critic moved dev accuracy from 56.1% to 47.4% and added one hallucination flag, so the final configuration keeps the rule-based critic. "
            "Injecting a 30% tool-fault rate produced 9 tool-call errors and dropped accuracy to 47.4%, but all 57 tasks still completed. The retry and escalation paths held; the cost was answer quality, not availability.")]
story += [P("3. By question category", "h1"), img("f09_category.png", 150),
          P("Figure 2. End-to-end accuracy by category with Wilson 95% intervals. Most categories hold 2 to 9 queries, so the intervals are wide. Read the bars as direction, not as a ranking.", "cap"),
          P("Out-of-scope and case-lookup questions are the strongest. Hypothetical loan questions (HYP) and procedure questions (PROC) are the weakest on test: 0 of 4 pass in each. These are the questions where the gold answer needs several specific facts at once, so one missing phrase fails the query. "
            "The test split has 34 rows, so the 95% interval on its 32.4% is roughly 19% to 49%.")]
story += [P("4. How behaviour goes wrong", "h1"), img("f10_confusion.png", 115),
          P("Figure 3. Expected behaviour (rows) against predicted behaviour (columns), dev and test combined (91 queries).", "cap"),
          P("Refusals and escalations are reliable on test: 5 of 5 out-of-scope, 2 of 2 unsafe and 2 of 2 escalation queries are correct. On dev they are weaker (7 of 9, 2 of 3 and 2 of 4), so the test result may be partly luck on a small sample. "
            "The common error runs the other way: the agent adds a caveat to a question that should get a plain answer (5 test queries, 9 on dev), or makes the wrong clarification decision (3 on test: two answered when the agent should have asked, one asked when it should have answered). "
            "This is over-caution, and it is the largest single behaviour error.")]
story += [P("5. Why reasoning errors happen", "h1"), img("f11_subcauses.png", 150),
          P("Figure 4. Sub-causes inside the failed queries, dev and test. A sub-cause is a second label under the primary class.", "cap")]
sc = A["subcauses"]
rows = [["Sub-cause", "Dev", "Test", "What it means"]]
desc = {"right behaviour, missing facts or wording": "Behaviour label is correct but a must-include fact or phrase is absent.",
        "plain answer vs caveat mismatch": "Agent adds a caveat where gold is a plain answer, or the reverse.",
        "clarification mismatch": "Agent answers when it should ask, or asks when it should answer.",
        "wrong kind of refusal or escalation": "Refusal happens but with the wrong label (unsafe vs out of scope vs escalate).",
        "over-refusal": "Agent refuses a question that is in scope.",
        "missed refusal or escalation": "Agent answers a question it should refuse or escalate."}
for k in desc:
    rows.append([k, sc["dev"].get(k, 0), sc["test"].get(k, 0), desc[k]])
story += [KeepTogether([table(rows, [58, 14, 14, 88]), P("Table 3. Sub-causes of failure. Dev counts sum to 25 failures, test counts to 23.", "cap")]),
          P("The largest sub-cause is the near miss: the answer is right in kind but misses a required word. Examples are PROC-05 (missing address proof, tax token, fee), PROC-08 (missing \"may be required\") and PROC-07 (missing Form 60 for a buyer without PAN). "
            "Part of this is the scorer being strict about wording, and part is the 8B model compressing a long checklist. The two are hard to separate without a human read of each row, which is the next step in the backlog.")]

# ------------------------------------------------------------------ examples
def ex(run, ft, qid):
    for f in R[run]["failures"]:
        if f["failure_type"] == ft and f["query_id"] == qid:
            return f
story += [P("6. Worked examples from the saved runs", "h1"),
          P("Each row is a real failed query. The answer text is shortened. The full text is in results/agent/*/answers.jsonl and in the trace dashboard.")]
rows = [["Class", "Query (id)", "Expected, got", "What happened", "Fix"]]
x = ex("test", "TOOL_CALL_ERROR", "CLR-06")
rows.append(["TOOL_CALL_ERROR", f"\"{x['query']}\" ({x['query_id']}, test)", f"Expected: {x['expected'].replace('_',' ').lower()}<br/>Got: {x['predicted'].replace('_',' ').lower()}",
             "The date 31/11/2026 does not exist, so resolve_date returned a ToolError. The model saw the error and still answered with a caveat instead of asking for a valid date.", "Planner rule: a ToolError on a date forces ASK_CLARIFICATION."])
x = ex("test", "RETRIEVAL_MISS", "HYP-08")
rows.append(["RETRIEVAL_MISS", f"\"{x['query']}\" ({x['query_id']}, test)", f"Expected: {x['expected'].replace('_',' ').lower()}<br/>Got: {x['predicted'].replace('_',' ').lower()}",
             "Gold source is the Form 34 page (SRC-PARIVAHAN-HPA). The six sources shown held Form 35, MVA s.51 and Form 30. The answer never named Form 34.", "Query expansion for new-loan wording; add a hypothecation-addition alias."])
x = ex("test", "RETRIEVAL_MISS", "PROC-14")
rows.append(["RETRIEVAL_MISS", f"\"{x['query']}\" ({x['query_id']}, test)", f"Expected: {x['expected'].replace('_',' ').lower()}<br/>Got: {x['predicted'].replace('_',' ').lower()}",
             "Form 34 (add) and Form 35 (remove) were compared, but the Parivahan pages that state the difference were not in context. Form 34 was found, the explanation was not.", "Raise per-source cap for comparison questions."])
x = ex("dev_llm_critic", "HALLUCINATION", "HYP-10")
rows.append(["HALLUCINATION", f"\"{x['query']}\" ({x['query_id']}, dev, LLM critic)", f"Expected: {x['expected'].replace('_',' ').lower()}<br/>Got: {x['predicted'].replace('_',' ').lower()}",
             "Behaviour was right, but the heuristic found a citation id or date not backed by retrieval or tools. The same query is flagged again in the fault run. One case in 57, so treat it as a flag to review, not a rate.", "Keep the citation-strip step and the rule critic. Review the row by hand."])
x = ex("dev", "REASONING_ERROR", "PROC-10")
rows.append(["REASONING_ERROR", f"\"{x['query']}\" ({x['query_id']}, dev)", f"Expected: {x['expected'].replace('_',' ').lower()}<br/>Got: {x['predicted'].replace('_',' ').lower()}",
             "Over-refusal. Fee amounts are outside the copilot, but the gold answer points to the rule and the official fee schedule. The agent refused flatly.", "Refusal template that names the rule and the authority."])
x = ex("test", "REASONING_ERROR", "PROC-05")
rows.append(["REASONING_ERROR", f"\"{x['query']}\" ({x['query_id']}, test)", f"Expected: {x['expected'].replace('_',' ').lower()}<br/>Got: {x['predicted'].replace('_',' ').lower()}",
             "Right behaviour. The checklist listed RC, Form 29 and Form 30 but omitted address proof, tax token and fee. Gold source was in context.", "Checklist tool that returns a complete list, so the model copies it."])
x = ex("dev", "REASONING_ERROR", "PROC-07")
rows.append(["REASONING_ERROR", f"\"{x['query']}\" ({x['query_id']}, dev)", f"Expected: {x['expected'].replace('_',' ').lower()}<br/>Got: {x['predicted'].replace('_',' ').lower()}",
             "Caveat added to a plain answer, and Form 60 (the alternative to PAN) was missing. Gold sources were in context.", "Shorter system prompt; few-shot for plain answers."])
x = ex("dev", "RETRIEVAL_MISS", "OOS-02")
rows.append(["RETRIEVAL_MISS", f"\"{x['query']}\" ({x['query_id']}, dev)", f"Expected: {x['expected'].replace('_',' ').lower()}<br/>Got: {x['predicted'].replace('_',' ').lower()}",
             "Cross-RTO transfer should be refused as out of scope and point to Form 28 or NOC. The NOC page was not retrieved, so the agent answered as a normal transfer.", "A scope table was added in agent v2, yet this row still fails. Add inter-RTO wording to query expansion."])
story += [table(rows, [31, 38, 26, 52, 27]), P("Table 4. Real failures by class. The fix column is a proposal unless stated.", "cap")]

# ------------------------------------------------------------------ caveats
story += [P("7. What the labels cannot tell you", "h1"),
          B("<b>RETRIEVAL_MISS does not prove retrieval caused the failure.</b> In Week 2 we forced the gold sources into the prompt for all 57 rows. End-to-end accuracy stayed at 17.5%. Of 47 failed RAG answers, 26 had the gold source in context, 13 needed no retrieval, 8 failed in both retrieval and generation, and none were fixed by retrieval alone."),
          B("<b>Some REASONING_ERROR rows are scorer strictness.</b> Must-include checks match phrases. A correct answer worded differently fails. BLEU and ROUGE, computed next to the scorer, did not show the dev-to-test drop, so they cannot replace it."),
          B("<b>The classes are assigned by rules, not by a person.</b> A human review of the 23 test failures is still to be done, and it will move some rows between classes."),
          B("<b>The sample is small.</b> Dev has 57 rows and test has 34. A single query moves test accuracy by 2.9 points. The second evaluation set (v2, 196 rows) is a draft that the team has not yet reviewed or frozen, and it has not been run live."),
          B("<b>The test drop is real but its size is uncertain.</b> Dev 56.1% against test 32.4% reflects tuning on dev. The two intervals (43% to 68% and 19% to 49%) overlap only between 43% and 49%.")]

# ------------------------------------------------------------------ backlog
story += [P("8. Fix backlog, ranked", "h1")]
rows = [["#", "Fix", "Targets", "Expected effect", "Status"],
        ["1", "Human review of all 23 test failures and 25 dev failures; split scorer strictness from real errors", "REASONING_ERROR", "Replaces the heuristic with a ground-truth taxonomy.", "To do"],
        ["2", "Return a complete document checklist from a tool instead of letting the model compose it", "Missing facts (14 test)", "Largest sub-cause. Tool output is deterministic.", "To do"],
        ["3", "Planner rule: a ToolError on a date forces a clarification question", "TOOL_CALL_ERROR, clarification mismatch", "Fixes CLR-06 type cases.", "To do"],
        ["4", "Stop adding caveats unless the case has a financier or a conflict flag", "Plain answer vs caveat (5 test)", "Reduces over-caution.", "To do"],
        ["5", "Refusal template naming the rule and the authority", "Over-refusal", "Fixes PROC-10 type cases.", "To do"],
        ["6", "Review and freeze eval v2, then run the held-out half live", "Sample size", "Test n rises from 34 to 93; margin falls from about 17 to 10 points.", "Draft"],
        ["7", "Run the same eval on a second open model (Llama 3.1 8B local)", "Model dependence", "Shows which failures belong to the model.", "Ready, not run"]]
story += [table(rows, [8, 66, 38, 44, 18])]
story += [Spacer(1, 4 * mm), P("Reproduce: <font name='Body-B'>python scripts/analyse_failures.py</font> then <font name='Body-B'>python scripts/build_report_figures.py</font> then <font name='Body-B'>python scripts/build_taxonomy_pdf.py</font>. All three read saved results and make no model calls.", "cap")]

out = ROOT / "report" / "Failure_Taxonomy_Synapse6.pdf"
doc = BaseDocTemplate(str(out), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=20 * mm,
                      title="Failure Taxonomy, RTO SmartTransfer Copilot", author="Team Synapse6")
doc.addPageTemplates([PageTemplate(id="p", frames=[Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")], onPage=footer)])
doc.build(story)
print(out)
