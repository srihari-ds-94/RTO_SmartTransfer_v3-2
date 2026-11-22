# Eval set v2: what changed and why

## In one paragraph

v1 (98 rows, frozen on 2026-11-14, SHA-256 in `data/eval/eval_set_v1.sha256`) is untouched and is carried into v2 as
`cohort=v1`. v2 adds **98 new rows** (`cohort=v2`) and **48 new synthetic cases** (DEMO-101 to DEMO-148), giving
**196 rows** (103 dev, 93 test). The new test rows only use new test-split cases, so they are a clean held-out set that
no prompt, example or retrieval tweak has ever seen.

## Why this is not just "more rows"

A bigger set does not by itself raise accuracy. It does two honest things:

| | v1 | v2 |
|---|---|---|
| Dev rows (scored for tuning) | 64 (57 after exemplars) | 103 (96 after exemplars) |
| Test rows (scored once) | 34 | 93 |
| 95% margin of error, whole set at 50% | +/-9.9 pts | +/-7.0 pts |
| 95% margin of error, test only at 50% | +/-16.8 pts | +/-10.2 pts |

With 34 test rows, a drop from 70% (dev) to 60% (test) is inside the noise. With 93 test rows the same gap starts to
mean something. v1 also spent its test split during Week 3 and 4 tuning, so the v2 test rows are the first clean read.

Read the dev to test gap as three effects mixed together: sampling noise (shrinks with n), tuning on dev (needs a
fresh test set), and real difficulty differences between splits (check the per-category table in the notebook).

## Composition of the 98 new rows

| Category | Dev | Test | Notes |
|---|---|---|---|
| CASE | 9 | 13 | Gold computed from the case record, checked against `tools.check_case` |
| DDL | 6 | 10 | Dates computed from `sale_date` + rule windows; 3 ABSTAIN/clarify rows |
| PROC | 5 | 7 | Document lists, forms, caveats where Parivahan and the WB FAQ differ |
| HYP | 4 | 6 | Form 34/35, consent, successor bank (escalate), VAHAN continuation (abstain) |
| CHK | 3 | 4 | Role x scenario checklists, one clarification |
| DRAFT | 3 | 4 | Drafts marked not sent, refusal to sign or send |
| CLAR | 3 | 5 | Vague dates, impossible dates, unknown case, missing role |
| OOS | 3 | 5 | Cross-RTO, commercial, payments, email, off-topic, language |
| SAFE | 3 | 5 | Consent bypass, PII, guarantees, injection, forgery verdict, legal dispute |

Behaviour mix of the whole v2 set: ANSWER 77, ANSWER_WITH_CAVEAT 41, ASK_CLARIFICATION 19, ESCALATE_HUMAN_REVIEW 16,
REFUSE_OUT_OF_SCOPE 26, REFUSE_UNSAFE 13, ABSTAIN_NO_SOURCE 4 (v1 had 1; the spec lists several no-source classes).

## New synthetic cases

48 cases, 12 per loan scenario (5 train, 2 val, 5 test), same conventions as DEMO-001 to DEMO-100: engine mismatch is
declared number + 200, findings are derived by rule (never typed), sale dates run 2026-10-05 to 2026-11-14.
`synthetic_cases_100.csv` is kept as the original; `synthetic_cases.csv` is the 148-case file the code reads.
Every case was re-checked: `tools.check_case` returns exactly `expected_findings` for all 148.

## Status of the new rows: DRAFT until the team reviews

New rows carry `verification_status` DATASET_DERIVED, SOURCE_SEEN or POLICY and an empty `reviewed_by`.
`rto-evalset --version v2 --freeze` refuses to freeze until every row is VERIFIED with reviewers named. Suggested
review split (about 15 minutes per person):

1. Rule and source facts (PROC, HYP, CHK, DDL non-case rows): check each gold against the cited source.
2. Behaviour labels on the judgement calls listed below.
3. Safety and scope rows: confirm each refusal matches spec section 3.

### Judgement calls to look at first

* **DDL "exact late-filing amount" (ABSTAIN_NO_SOURCE)** and **HYP "VAHAN continuation in West Bengal" (ABSTAIN_NO_SOURCE)**:
  an agent could reasonably answer with a caveat instead. Decide which label the spec intends.
* **Clean LOAN_TERMINATED cases** are labelled ANSWER_WITH_CAVEAT (mirrors v1 CASE-17: Form 35 must be in the package).
* **Consent cases** are always REVIEW_REQUIRED, so the clean ones are ANSWER_WITH_CAVEAT (mirrors CASE-11).
* **PAN card question** (PROC): the WB FAQ and Parivahan differ on whose PAN is needed, so it is a caveat row.
* **"Insurance and PUC left out"** (PROC): the sources list the documents but do not say what an office accepts.

### Known staleness in frozen v1 (not edited)

CLR-05 says demo cases run "from DEMO-001 to DEMO-100". With v2 they run to DEMO-148. The row is frozen, so the
text stays; the must_include items (`DEMO-150`, "not found|no case") are still correct and the scorer is unaffected.

## Reproduce

```bash
python scripts/build_eval_v2.py          # rebuild cases + eval_set_v2.csv deterministically
rto-evalset --version v2                  # validate (no leakage, no near-duplicates, gold consistent)
RTO_EVAL_SET=v2 python -m rto_copilot.baseline --mock oracle   # sanity: the scorer accepts the gold (100%)
```
