"""Download the Week 2 source documents into corpus/raw/ and record them in corpus/sources_manifest.csv.

Every file is an official publication (or a clearly marked secondary copy) so the corpus is real text, not padding.
Already-downloaded files are skipped; pass --refresh to fetch again.

    python fetch_sources.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import ssl
import sys
import time
from datetime import date
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parents[1] / "data"   # corpus lives in data/corpus
ROOT = HERE.parent
RAW = HERE / "corpus" / "raw"
MANIFEST = HERE / "corpus" / "sources_manifest.csv"
UA = {"User-Agent": "Mozilla/5.0 (RTO SmartTransfer capstone; research use)"}

# Text extracted from the official PDF by hand because indiacode.nic.in timed out on direct download.
ORIGIN = {"gca_1897.txt": "https://www.indiacode.nic.in/bitstream/123456789/15374/1/the_general_clauses_act%2C_1897.pdf"}

# file name -> (url or local path, publisher, status). Status OFFICIAL = government site; SECONDARY = copy hosted elsewhere.
SOURCES = {
    "mva_1988.pdf": ("https://www.indiacode.nic.in/bitstream/123456789/9460/1/a1988-59.pdf", "India Code (Legislative Department)", "OFFICIAL"),
    "cmvr_1989.pdf": ("https://transport.wb.gov.in/wp-content/download/Rules/The%20C%20M%20V%20Rules-1989.pdf", "West Bengal Transport Department", "OFFICIAL"),
    "wb_mvr_1989.pdf": ("https://wbmvtoa.in/downloads/WB-MOTOR-VEHICLE-RULES-1989.pdf", "WB Motor Vehicles Tax Officers' Association (copy of the WB Gazette rules)", "SECONDARY"),
    "gca_1897.txt": ("corpus/manual/gca_1897.txt", "India Code", "OFFICIAL"),
    "parivahan_ownership.html": ("https://mparivahan.parivahan.gov.in/mstatic/english/rc-info-ownership.html", "Parivahan (MoRTH)", "OFFICIAL"),
    "parivahan_hp_termination.html": ("https://mparivahan.parivahan.gov.in/mstatic/english/rc-info-hp-termination.html", "Parivahan (MoRTH)", "OFFICIAL"),
    "parivahan_hp_endorsement.html": ("https://mparivahan.parivahan.gov.in/mstatic/english/rc-info-hp-endorsement.html", "Parivahan (MoRTH)", "OFFICIAL"),
    "parivahan_no_objection.html": ("https://mparivahan.parivahan.gov.in/mstatic/english/rc-info-no-objection.html", "Parivahan (MoRTH)", "OFFICIAL"),
    "parivahan_duplicate_rc.html": ("https://mparivahan.parivahan.gov.in/mstatic/english/rc-info-duplicate-rc.html", "Parivahan (MoRTH)", "OFFICIAL"),
    "parivahan_address_change.html": ("https://mparivahan.parivahan.gov.in/mstatic/english/rc-info-address-change.html", "Parivahan (MoRTH)", "OFFICIAL"),
    "parivahan_hp_continuation.html": ("https://staging.parivahan.gov.in/vahaneservice/usermanual/selectedservice/hpc/", "Parivahan VAHAN user manual", "OFFICIAL"),
    "wb_transport_faq.html": ("https://transport.wb.gov.in/references/faq/", "West Bengal Transport Department", "OFFICIAL"),
    "form29.pdf": ("../02_Official_RTO_Sources/downloaded/Form29.pdf", "Parivahan (MoRTH)", "OFFICIAL"),
    "form30.pdf": ("../02_Official_RTO_Sources/downloaded/Form30.pdf", "Parivahan (MoRTH)", "OFFICIAL"),
    "form34.pdf": ("../02_Official_RTO_Sources/downloaded/Form34.pdf", "Parivahan (MoRTH)", "OFFICIAL"),
    "form35.pdf": ("../02_Official_RTO_Sources/downloaded/Form35.pdf", "Parivahan (MoRTH)", "OFFICIAL"),
    # Form29.pdf and Form30.pdf above are scanned images with no text layer; these official copies carry the same form as text.
    "form29_text.pdf": ("https://transport.rajasthan.gov.in/content/dam/transport/pdf/forms/FORM-29.pdf", "Rajasthan Transport Department (copy of central Form 29)", "OFFICIAL"),
    "form30_vahan.html": ("https://vahan.parivahan.gov.in/vahanservice/vahan/ui/FeeRecieptPrintReport/Form_30_TO.xhtml", "VAHAN (MoRTH)", "OFFICIAL"),
}


class LegacyTLSAdapter(requests.adapters.HTTPAdapter):
    """transport.wb.gov.in still needs TLS legacy renegotiation; certificates are verified as usual."""

    def init_poolmanager(self, *args, **kwargs):
        ctx = ssl.create_default_context()
        ctx.options |= getattr(ssl, "OP_LEGACY_SERVER_CONNECT", 0x4)
        kwargs["ssl_context"] = ctx
        return super().init_poolmanager(*args, **kwargs)


SESSION = requests.Session()
SESSION.headers.update(UA)
SESSION.mount("https://transport.wb.gov.in", LegacyTLSAdapter())


def fetch(name: str, location: str) -> bytes:
    if not location.startswith("http"):
        return (HERE / location).read_bytes()
    for attempt in range(3):
        try:
            r = SESSION.get(location, timeout=120)
            r.raise_for_status()
            return r.content
        except requests.exceptions.Timeout:
            if attempt == 2:
                raise
            time.sleep(5)
    raise RuntimeError("unreachable")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    old = {}
    if MANIFEST.exists():
        with MANIFEST.open(newline="", encoding="utf-8") as f:
            old = {r["file"]: r for r in csv.DictReader(f)}
    rows, failed = [], []
    for name, (location, publisher, status) in SOURCES.items():
        path = RAW / name
        if path.exists() and not args.refresh and name in old:
            rows.append(old[name])
            print(f"cached   {name}")
            continue
        try:
            data = fetch(name, location)
        except Exception as exc:  # network errors are reported, not fatal, so one dead site does not block the rest
            failed.append(name)
            print(f"FAILED   {name}: {exc}")
            continue
        path.write_bytes(data)
        rows.append({"file": name, "url": ORIGIN.get(name, location), "publisher": publisher, "status": status,
                     "accessed_on": date.today().isoformat(), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        print(f"fetched  {name} ({len(data):,} bytes)")
    with MANIFEST.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "url", "publisher", "status", "accessed_on", "bytes", "sha256"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} sources in {MANIFEST.name}; {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
