"""Fetch exact OpenDART receipt versions. The local attempt ledger counts failed calls too."""

import argparse
import csv
import hashlib
import io
import json
import os
import re
import time
import zipfile
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree

import requests
from dotenv import dotenv_values


ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "reports/20261109_PM_m9_v1/manual_check_sample.csv"
STORE = ROOT / "data/interim/pm10-manual-check-20261117"
API = "https://opendart.fss.or.kr/api/document.xml"
LEDGER = STORE / "document_calls.jsonl"
MAX_ATTEMPTS = 200


def log(event, receipt, **fields):
    with LEDGER.open("a", encoding="utf-8") as ledger:
        entry = dict(event=event, receipt=receipt, at=datetime.now().astimezone().isoformat(), **fields)
        ledger.write(json.dumps(entry, ensure_ascii=False) + "\n")
        ledger.flush()
        os.fsync(ledger.fileno())


def count_attempts():
    if not LEDGER.exists():
        return 0
    return sum(json.loads(line)["event"] == "attempt" for line in LEDGER.read_text(encoding="utf-8").splitlines())


def extract_exact_receipt(zip_path, receipt):
    expected = f"{receipt}.xml"
    with zipfile.ZipFile(zip_path) as archive:
        members = [name for name in archive.namelist() if name in {expected, "/" + expected}]
        if len(members) != 1:
            return archive.namelist()
        zip_path.with_name(expected).write_bytes(archive.read(members[0]))
    return None


def download(session, key, receipt, folder):
    log("attempt", receipt)
    try:
        response = session.get(API, params={"crtfc_key": key, "rcept_no": receipt},
                               timeout=(8, 30), allow_redirects=False)
    except requests.RequestException as error:
        result = type(error).__name__
    else:
        result = f"HTTP {response.status_code}"
        if response.ok and zipfile.is_zipfile(io.BytesIO(response.content)):
            zip_path = folder / f"{receipt}.zip"
            zip_path.write_bytes(response.content)
            digest = hashlib.sha256(response.content).hexdigest()
            members = extract_exact_receipt(zip_path, receipt)
            if members is None:
                log("result", receipt, status="saved", sha256=digest)
                return "saved"
            result = "ZIP without exact receipt XML"
            log("result", receipt, status=result, sha256=digest, members=members)
            return result
        elif response.ok:
            try:
                status = ElementTree.fromstring(response.content).findtext("status")
            except ElementTree.ParseError:
                status = None
            if status and re.fullmatch(r"\d{3}", status):
                result = f"DART {status}"
    log("result", receipt, status=result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipts", nargs="*", help="Optional exact receipt numbers. Default: all sample rows.")
    args = parser.parse_args()
    if args.receipts:
        receipts = args.receipts
    else:
        with SAMPLE.open(encoding="utf-8-sig", newline="") as source:
            receipts = [re.search(r"rcpNo=(\d{14})$", row["url"])[1] for row in csv.DictReader(source)]
    if any(not re.fullmatch(r"\d{14}", receipt) for receipt in receipts):
        parser.error("Expected 14-digit receipt numbers")

    folder = STORE / "documents"
    folder.mkdir(parents=True, exist_ok=True)
    key = (dotenv_values(ROOT / ".env").get("OPENDART_API_KEY") or "").strip()
    if not key:
        raise SystemExit("OPENDART_API_KEY missing in .env")
    with requests.Session() as session:
        consecutive_transport_failures = 0
        for receipt in dict.fromkeys(receipts):
            if (folder / f"{receipt}.xml").exists():
                continue
            zip_path = folder / f"{receipt}.zip"
            if zip_path.exists() and extract_exact_receipt(zip_path, receipt) is None:
                print(receipt, "extracted cached ZIP", flush=True)
                continue
            if count_attempts() >= MAX_ATTEMPTS:
                raise SystemExit("OpenDART call limit reached")
            result = download(session, key, receipt, folder)
            print(receipt, result, flush=True)
            consecutive_transport_failures = (
                consecutive_transport_failures + 1 if result.endswith(("Error", "Timeout")) else 0
            )
            if result in {"DART 010", "DART 011", "DART 012", "DART 020", "DART 901"}:
                break
            if consecutive_transport_failures == 2:
                break
            time.sleep(0.3)
    print(f"Total OpenDART document attempts: {count_attempts()}", flush=True)


if __name__ == "__main__":
    main()
