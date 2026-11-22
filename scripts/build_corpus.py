"""Turn corpus/raw/ into corpus/documents.jsonl: one document per statute section, rule, FAQ entry, guidance page or form.

Each document carries the source_id used by the eval set (e.g. SRC-MVA-S50, SRC-WB-FAQ), so retrieval can be scored
against gold_source_ids. Chunking happens later, in retrieval.py, so chunk sizes can be compared on the same documents.

    python build_corpus.py
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pymupdf
from bs4 import BeautifulSoup

HERE = Path(__file__).resolve().parents[1] / "data"   # corpus lives in data/corpus
RAW = HERE / "corpus" / "raw"
OUT = HERE / "corpus" / "documents.jsonl"
MANIFEST = HERE / "corpus" / "sources_manifest.csv"

# A section or rule heading: "50. Transfer of ownership.—(1)", "1[50A. Title.—", "69. Assignment of new registration mark—"
# Headings usually start a line; in text copied from a web page they can also follow "] " or ". " mid-line.
HEADING = re.compile(r"(?:^|(?<=\n)|(?<=\] )|(?<=\. ))[ \t]*(?:\d{1,2}\s*\[)?(\d{1,3}[A-Z]{0,3})\.\s*([A-Z][^\n—–]{2,160}?)\.?\s*[—–]")
NOT_A_TITLE = re.compile(r"^(Ins|Subs|Omitted|The words|Added|Rep|Renumbered|Inserted|Substituted|Earlier|Vide|Amended|Clause|Cl|Sub|Proviso|Words|Figures|Now)\b")
FOOTNOTE_LINE = re.compile(r"^\s*\d{1,3}\s*\.?\s*(Ins|Subs|Omitted|The words|Added|Rep|Renumbered|Inserted|Substituted|Earlier|Vide|Amended|Clause|Cl\.|Sub-|Proviso|Words|Figures|Now)\b.*$")
PAGE_NUMBER_LINE = re.compile(r"^\s*\d{1,3}\s*$")

PARIVAHAN = {
    "parivahan_ownership.html": ("SRC-PARIVAHAN-TO", "Parivahan - Transfer of Ownership"),
    "parivahan_hp_termination.html": ("SRC-PARIVAHAN-HPT", "Parivahan - Hypothecation Termination"),
    "parivahan_hp_endorsement.html": ("SRC-PARIVAHAN-HPA", "Parivahan - Hypothecation Addition"),
    "parivahan_no_objection.html": ("SRC-PARIVAHAN-NOC", "Parivahan - No Objection Certificate"),
    "parivahan_duplicate_rc.html": ("SRC-PARIVAHAN-DUP", "Parivahan - Duplicate RC"),
    "parivahan_address_change.html": ("SRC-PARIVAHAN-ADDR", "Parivahan - Change of Address"),
    "parivahan_hp_continuation.html": ("SRC-PARIVAHAN-HPC", "Parivahan - Hypothecation Continuation user manual"),
}
FORMS = {
    "form29_text.pdf": ("SRC-FORM29", "Form 29 - Notice of transfer of ownership of a motor vehicle"),
    "form30_vahan.html": ("SRC-FORM30", "Form 30 - Application for intimation and transfer of ownership"),
    "form34.pdf": ("SRC-FORM34", "Form 34 - Entry of hypothecation agreement after registration"),
    "form35.pdf": ("SRC-FORM35", "Form 35 - Notice of termination of hypothecation agreement"),
}
STATUTES = {
    # file -> (source_id prefix, title, unit word, text marking the start of the body after the table of contents)
    "mva_1988.pdf": ("SRC-MVA-S", "Motor Vehicles Act, 1988", "section", "ACT NO. 59 OF 1988"),
    "cmvr_1989.pdf": ("SRC-CMVR-R", "Central Motor Vehicles Rules, 1989", "rule", ""),
    "wb_mvr_1989.pdf": ("SRC-WBMVR-R", "West Bengal Motor Vehicles Rules, 1989", "rule", ""),
    "gca_1897.txt": ("SRC-GCA-S", "General Clauses Act, 1897", "section", "ACT NO. 10 OF 1897"),
}


def squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def pdf_text(path: Path) -> str:
    with pymupdf.open(path) as doc:
        return "\n".join(page.get_text() for page in doc)


def clean_statute(text: str) -> str:
    lines = [ln for ln in text.splitlines() if not FOOTNOTE_LINE.match(ln) and not PAGE_NUMBER_LINE.match(ln)]
    return "\n".join(lines)


def number_key(num: str) -> int:
    return int(re.match(r"\d+", num).group())


def split_statute(text: str) -> list[tuple[str, str, str]]:
    """Return (number, title, body) for each section/rule, keeping headings whose numbers move forward in order."""
    found, last = [], 0
    for m in HEADING.finditer(text):
        num, title = m.group(1), squash(m.group(2))
        n = number_key(num)
        if NOT_A_TITLE.match(title) or n < last or n > last + 25:
            continue
        found.append((m.start(), num, title))
        last = n
    units = []
    for i, (start, num, title) in enumerate(found):
        end = found[i + 1][0] if i + 1 < len(found) else len(text)
        body = squash(text[start:end])
        if units and units[-1][0] == num:
            units[-1] = (num, units[-1][1], units[-1][2] + " " + body)
        else:
            units.append((num, title, body))
    return units


def html_text(path: Path) -> str:
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "noscript"]):
        tag.decompose()
    main = soup.find("main") or soup.body or soup
    return squash(main.get_text(" ", strip=True))


def faq_entries(path: Path) -> list[tuple[str, str, str]]:
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    main = soup.find("main") or soup.body
    entries, section = [], "General"
    for h in main.find_all("h4"):
        prev = h.find_previous(string=re.compile(r"^[A-Z][A-Z &/]{3,}$"))
        if prev:
            section = prev.strip().title()
        parts = []
        for sib in h.next_siblings:
            if getattr(sib, "name", None) == "h4":
                break
            parts.append(sib.get_text(" ", strip=True) if hasattr(sib, "get_text") else str(sib))
        entries.append((section, squash(h.get_text(" ", strip=True)), squash(" ".join(parts))))
    return entries


def main() -> int:
    with MANIFEST.open(newline="", encoding="utf-8") as f:
        manifest = {r["file"]: r for r in csv.DictReader(f)}
    docs = []

    def add(doc_id, source_id, title, locator, text, file):
        m = manifest[file]
        docs.append({"doc_id": doc_id, "source_id": source_id, "title": title, "locator": locator, "url": m["url"],
                     "publisher": m["publisher"], "status": m["status"], "file": file, "text": text, "n_words": len(text.split())})

    for file, (prefix, act, unit, body_marker) in STATUTES.items():
        path = RAW / file
        text = path.read_text(encoding="utf-8") if path.suffix == ".txt" else pdf_text(path)
        if body_marker and body_marker in text:
            text = text[text.index(body_marker):]
        for num, title, body in split_statute(clean_statute(text)):
            short = prefix.replace("SRC-", "").rsplit("-", 1)[0]
            add(f"{short}-{unit[0].upper()}{num}", prefix + num, f"{act}, {unit} {num}: {title}", f"{unit} {num}", body, file)

    for file, (source_id, title) in PARIVAHAN.items():
        add(source_id.replace("SRC-", ""), source_id, title, "full page", html_text(RAW / file), file)

    counter = Counter()
    for section, question, answer in faq_entries(RAW / "wb_transport_faq.html"):
        counter[section] += 1
        add(f"WB-FAQ-{re.sub(r'[^A-Z]', '', section.upper())[:12]}-{counter[section]:02d}", "SRC-WB-FAQ",
            f"West Bengal Transport FAQ ({section}): {question}", f"{section} Q{counter[section]}", f"{question} {answer}",
            "wb_transport_faq.html")

    for file, (source_id, title) in FORMS.items():
        path = RAW / file
        text = html_text(path) if path.suffix == ".html" else squash(pdf_text(path))
        add(source_id.replace("SRC-", ""), source_id, title, "full form", text, file)

    docs = [d for d in docs if d["n_words"] >= 5]
    ids = Counter(d["doc_id"] for d in docs)
    dupes = [k for k, v in ids.items() if v > 1]
    if dupes:
        print("duplicate doc_ids:", dupes[:10])
        return 1
    with OUT.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    by_file = Counter(d["file"] for d in docs)
    for file, n in by_file.items():
        print(f"{n:5d}  {file}")
    words = sorted(d["n_words"] for d in docs)
    print(f"\n{len(docs)} documents, {sum(words):,} words; median {words[len(words) // 2]} words, max {words[-1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
