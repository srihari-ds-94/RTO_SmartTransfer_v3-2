"""Validate an eval set against the synthetic dataset and the registries, and freeze it.

    rto-evalset                       # validate the default version (v2) and print a summary
    rto-evalset --version v1          # validate the frozen Week 1 set
    rto-evalset --version v2 --freeze # after team review: write the SHA-256 record that unlocks the test split

Checks beyond the Week 1 validator:
  * v2 must contain every v1 row unchanged (the frozen set is never edited in place)
  * duplicate or near-duplicate query text is rejected
  * v2 rows carry a ``cohort`` column (v1 = the frozen 98 rows, v2 = rows added later)
  * test rows may only use test-split cases, dev rows only train and val cases (no leakage)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
from collections import Counter
from datetime import date, timedelta

from . import config
from .scoring import margin_of_error

COLUMNS = [
    "query_id", "split", "category", "expected_behavior", "user_role", "scenario",
    "case_id", "reference_date", "query", "gold_answer", "must_include",
    "must_not_include", "gold_finding_codes", "gold_source_ids", "gold_rule_ids",
    "verification_status", "reviewed_by", "agent_id", "also_involves",
]
COLUMNS_V2 = COLUMNS + ["cohort"]
AGENTS = {"PLANNER", "VERIFICATION", "DEADLINE_NOTIFICATION", "CRITIC"}
CATEGORY_PREFIX = {
    "PROC": "PROC", "HYP": "HYP", "DDL": "DDL", "CASE": "CASE", "CHK": "CHK",
    "DRAFT": "DRF", "CLAR": "CLR", "OOS": "OOS", "SAFE": "SAF",
}
BEHAVIORS = {
    "ANSWER", "ANSWER_WITH_CAVEAT", "ASK_CLARIFICATION", "ABSTAIN_NO_SOURCE",
    "REFUSE_OUT_OF_SCOPE", "REFUSE_UNSAFE", "ESCALATE_HUMAN_REVIEW",
}
ROLES = {"buyer", "seller", "financier", "citizen", "rto_reviewer"}
SCENARIOS = {
    "NO_LOAN", "LOAN_TERMINATED", "LOAN_ACTIVE_PENDING", "LOAN_ACTIVE_WITH_CONSENT",
    "UNSPECIFIED", "NOT_APPLICABLE",
}
STATUSES = {"SOURCE_SEEN", "NEEDS_SOURCE_CHECK", "DATASET_DERIVED", "POLICY", "VERIFIED"}
CASE_SPLITS_FOR = {"dev": {"train", "val"}, "test": {"test"}}
SIZE_RANGE = {"v1": (60, 100), "v2": (150, 250)}


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def split_list(value, sep=";"):
    return [v.strip() for v in value.split(sep) if v.strip()]


def contains_any(text, alternatives):
    low = text.lower()
    return any(alt.lower() in low for alt in alternatives)


def normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def validate(rows, cases, sources, rules, version: str = "v2"):
    errors = []
    seen, seen_text = set(), {}

    for n, row in enumerate(rows, start=2):
        qid = row["query_id"]
        where = f"{qid} (line {n})"

        if qid in seen:
            errors.append(f"{where}: duplicate query_id")
        seen.add(qid)
        key = normalise(row["query"])
        if key in seen_text and seen_text[key] != qid:
            errors.append(f"{where}: same question text as {seen_text[key]}")
        seen_text.setdefault(key, qid)

        cat = row["category"]
        if cat not in CATEGORY_PREFIX:
            errors.append(f"{where}: unknown category {cat}")
        elif not qid.startswith(CATEGORY_PREFIX[cat] + "-"):
            errors.append(f"{where}: id prefix does not match category {cat}")
        if row["split"] not in CASE_SPLITS_FOR:
            errors.append(f"{where}: split must be dev or test")
        if "cohort" in row and row["cohort"] not in ("v1", "v2"):
            errors.append(f"{where}: cohort must be v1 or v2")
        if row["expected_behavior"] not in BEHAVIORS:
            errors.append(f"{where}: unknown expected_behavior {row['expected_behavior']}")
        if row["user_role"] not in ROLES:
            errors.append(f"{where}: unknown user_role {row['user_role']}")
        if row["scenario"] not in SCENARIOS:
            errors.append(f"{where}: unknown scenario {row['scenario']}")
        if row["verification_status"] not in STATUSES:
            errors.append(f"{where}: unknown verification_status {row['verification_status']}")
        if row["agent_id"] not in AGENTS:
            errors.append(f"{where}: unknown agent_id {row['agent_id']}")
        for agent in split_list(row["also_involves"]):
            if agent not in AGENTS or agent == row["agent_id"]:
                errors.append(f"{where}: invalid also_involves entry {agent}")
        if not row["query"].strip() or not row["gold_answer"].strip():
            errors.append(f"{where}: query and gold_answer are required")
        if not split_list(row["gold_source_ids"]):
            errors.append(f"{where}: at least one gold_source_id is required")

        for sid in split_list(row["gold_source_ids"]):
            if sid not in sources:
                errors.append(f"{where}: unknown source_id {sid}")
        for rid in split_list(row["gold_rule_ids"]):
            if rid not in rules:
                errors.append(f"{where}: unknown rule_id {rid}")

        gold = row["gold_answer"]
        for item in split_list(row["must_include"]):
            if not contains_any(gold, split_list(item, "|")):
                errors.append(f"{where}: gold_answer is missing must_include item '{item}'")
        for item in split_list(row["must_not_include"]):
            if item.lower() in gold.lower():
                errors.append(f"{where}: gold_answer contains must_not_include item '{item}'")

        case_id = row["case_id"]
        if not case_id:
            if row["gold_finding_codes"]:
                errors.append(f"{where}: gold_finding_codes set without a case_id")
            if cat == "CASE":
                errors.append(f"{where}: CASE items need a case_id")
            continue

        case = cases.get(case_id)
        if case is None:
            if row["expected_behavior"] != "ASK_CLARIFICATION":
                errors.append(f"{where}: case_id {case_id} not in synthetic dataset")
            continue
        if row["scenario"] != case["scenario"]:
            errors.append(f"{where}: scenario {row['scenario']} != dataset {case['scenario']}")
        if case["split"] not in CASE_SPLITS_FOR.get(row["split"], set()):
            errors.append(f"{where}: {row['split']} item uses {case['split']} case {case_id} (leakage)")

        gold_codes = set(split_list(row["gold_finding_codes"]))
        data_codes = set(split_list(case["expected_findings"]))
        if (cat == "CASE" or gold_codes) and gold_codes != data_codes:
            errors.append(f"{where}: finding codes {sorted(gold_codes)} != dataset {sorted(data_codes)}")

        if row["reference_date"]:
            sale = date.fromisoformat(case["sale_date"])
            ref = date.fromisoformat(row["reference_date"])
            if sale > ref:
                if row["expected_behavior"] != "ASK_CLARIFICATION":
                    errors.append(f"{where}: sale date {sale} is after reference date; expected ASK_CLARIFICATION")
                continue
            for rid in split_list(row["gold_rule_ids"]):
                window = rules.get(rid, {}).get("window_days", "")
                if window:
                    due = (sale + timedelta(days=int(window))).isoformat()
                    if due not in gold:
                        errors.append(f"{where}: gold_answer lacks due date {due} for {rid}")

    lo, hi = SIZE_RANGE[version]
    if not lo <= len(rows) <= hi:
        errors.append(f"eval set has {len(rows)} items; {version} requires {lo}-{hi}")
    return errors


def v1_immutability_errors(v1_rows: list[dict], v2_rows: list[dict]) -> list[str]:
    """Every v1 row must appear in v2 with identical content: the frozen set is only ever extended."""
    by_id = {r["query_id"]: r for r in v2_rows}
    errors = []
    for r in v1_rows:
        new = by_id.get(r["query_id"])
        if new is None:
            errors.append(f"{r['query_id']}: v1 row is missing from v2")
        elif {k: new[k] for k in COLUMNS} != {k: r[k] for k in COLUMNS}:
            diff = [k for k in COLUMNS if new[k] != r[k]]
            errors.append(f"{r['query_id']}: v1 row was edited in v2 (columns {diff})")
        elif new.get("cohort") != "v1":
            errors.append(f"{r['query_id']}: v1 row must carry cohort=v1 in v2")
    return errors


def freeze_errors(rows):
    return [
        f"{r['query_id']}: verification_status={r['verification_status']} reviewed_by='{r['reviewed_by']}'"
        for r in rows
        if r["verification_status"] != "VERIFIED" or not r["reviewed_by"].strip()
    ]


def summary(rows):
    print(f"Items: {len(rows)}  (95% margin of error at 50% pass rate: +/-{margin_of_error(len(rows)):.3f})")
    for field in ("cohort", "split", "category", "expected_behavior", "agent_id", "verification_status"):
        if field in rows[0]:
            counts = Counter(r[field] for r in rows)
            print(f"  {field}: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    if "cohort" in rows[0]:
        both = Counter((r["cohort"], r["split"]) for r in rows)
        print("  cohort x split: " + ", ".join(f"{c}/{s}={v}" for (c, s), v in sorted(both.items())))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", default=config.EVAL_VERSION, choices=["v1", "v2"])
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()

    path = config.eval_path(args.version)
    rows = read_csv(path)
    expected = COLUMNS if args.version == "v1" else COLUMNS_V2
    if list(rows[0].keys()) != expected:
        sys.exit(f"Column mismatch.\nExpected: {expected}\nFound:    {list(rows[0].keys())}")
    cases = {r["case_id"]: r for r in read_csv(config.CASES)}
    sources = {r["source_id"]: r for r in read_csv(config.SOURCES)}
    rules = {r["rule_id"]: r for r in read_csv(config.RULES)}

    summary(rows)
    errors = validate(rows, cases, sources, rules, args.version)
    if args.version == "v2":
        errors += v1_immutability_errors(read_csv(config.eval_path("v1")), rows)
    if errors:
        print(f"\n{len(errors)} consistency error(s):")
        print("\n".join(f"  - {e}" for e in errors))
        sys.exit(1)
    print("\nConsistency checks passed.")

    pending = freeze_errors(rows)
    if not args.freeze:
        print(f"{len(pending)} item(s) still need reviewer verification before the freeze.")
        return
    if pending:
        print(f"\nCannot freeze: {len(pending)} item(s) not verified:")
        print("\n".join(f"  - {p}" for p in pending[:20]) + ("\n  ..." if len(pending) > 20 else ""))
        sys.exit(1)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    config.freeze_path(args.version).write_text(
        f"{digest}  {path.name}  frozen_on={date.today().isoformat()}\n", encoding="utf-8")
    print(f"Frozen. SHA-256 {digest} written to {config.freeze_path(args.version).name}")


if __name__ == "__main__":
    main()
