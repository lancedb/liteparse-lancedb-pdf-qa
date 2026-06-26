#!/usr/bin/env python3
"""Download the selected Climate Finance Bench subset.

This script intentionally uses only the Python standard library so it can run
before project dependencies are installed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen


REPO = "Pladifes/climate_finance_bench"
DEFAULT_REF = "main"
DEFAULT_COMPANIES = [
    "Samsung",
    "NVIDIA",
    "Google",
    "Ali Baba Group",
    "Nestle",
    "Total Energies S.A",
]

BENCHMARK_REL_PATH = "data/Benchmark/Climate Finance Bench - Dataset.json"
DATA_CONTENTS_API = f"https://api.github.com/repos/{REPO}/contents/data"


def fetch_bytes(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "liteparse-lancedb-pdf-qa"})
    with urlopen(request) as response:
        return response.read()


def fetch_json(url: str) -> Any:
    return json.loads(fetch_bytes(url).decode("utf-8"))


def write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_pages(page_ref: str) -> list[int]:
    return [int(value) for value in re.findall(r"page\s*(\d+)", page_ref)]


def required_modality(row: dict[str, Any]) -> str:
    extract_type = row.get("Extract type") or ""
    if "figure" in extract_type:
        return "figure"
    if "table" in extract_type:
        return "table"
    return "text"


def difficulty(row: dict[str, Any]) -> str:
    pages = set(parse_pages(row.get("Pages") or ""))
    question_type = row.get("Type of question")
    modality = required_modality(row)
    if question_type == "LR" or len(pages) >= 3:
        return "hard"
    if question_type == "NR" or modality in {"table", "figure"} or len(pages) == 2:
        return "medium"
    return "easy"


def normalize_questions(rows: list[dict[str, Any]], companies: set[str]) -> list[dict[str, Any]]:
    normalized = []
    for row in rows:
        company = row["Company's name"]
        pages = parse_pages(row.get("Pages") or "")
        if company not in companies or not pages:
            continue

        question_id = (
            f"{company.replace(' ', '_').replace('.', '').replace('&', 'and')}_{row['Question ID']}"
        )
        normalized.append(
            {
                "question_id": question_id,
                "company": company,
                "fiscal_year": row["Fiscal year"],
                "source_pdf": row["Documents"],
                "question": row["Question"],
                "expected_answer": row["Answer"],
                "expected_pages": sorted(set(pages)),
                "raw_pages": row["Pages"],
                "document_extracts": row["Document extracts"],
                "question_type": row["Type of question"],
                "extract_type": row.get("Extract type"),
                "required_modality": required_modality(row),
                "difficulty": difficulty(row),
            }
        )
    return normalized


def download(url: str, destination: Path, *, force: bool, dry_run: bool) -> bool:
    if destination.exists() and destination.stat().st_size > 0 and not force:
        print(f"skip existing {destination}")
        return False
    if dry_run:
        print(f"would download {url} -> {destination}")
        return False
    print(f"download {url} -> {destination}")
    write_bytes(destination, fetch_bytes(url))
    return True


def get_reports_tree(ref: str) -> list[dict[str, Any]]:
    contents_url = f"{DATA_CONTENTS_API}?ref={quote(ref)}"
    contents = fetch_json(contents_url)
    reports_entry = next(item for item in contents if item["name"] == "Company Reports")
    git_url = reports_entry["_links"]["git"] + "?recursive=1"
    tree = fetch_json(git_url)
    return tree["tree"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/raw/climate_finance_bench"))
    parser.add_argument("--ref", default=DEFAULT_REF)
    parser.add_argument("--companies", nargs="+", default=DEFAULT_COMPANIES)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    companies = set(args.companies)
    raw_base = f"https://raw.githubusercontent.com/{REPO}/{quote(args.ref)}/"
    benchmark_url = raw_base + quote(BENCHMARK_REL_PATH, safe="/")
    benchmark_path = args.out / "Benchmark" / "Climate Finance Bench - Dataset.json"

    if args.dry_run:
        print(f"selected companies: {', '.join(args.companies)}")

    download(benchmark_url, benchmark_path, force=args.force, dry_run=args.dry_run)
    if args.dry_run:
        benchmark_rows = json.loads(fetch_bytes(benchmark_url).decode("utf-8"))
    else:
        benchmark_rows = json.loads(benchmark_path.read_text())

    selected_questions = normalize_questions(benchmark_rows, companies)
    eval_dir = args.out.parent.parent / "eval"
    selected_questions_path = eval_dir / "selected_questions.jsonl"
    if args.dry_run:
        print(
            f"would write {len(selected_questions)} selected questions -> {selected_questions_path}"
        )
    else:
        eval_dir.mkdir(parents=True, exist_ok=True)
        selected_questions_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected_questions)
        )

    tree = get_reports_tree(args.ref)
    selected_reports = []
    for item in tree:
        path = item.get("path", "")
        if item.get("type") != "blob" or not path.endswith(".pdf"):
            continue
        parts = path.split("/")
        if len(parts) < 3:
            continue
        company = parts[1]
        if company not in companies:
            continue
        selected_reports.append(item)

    manifest_entries = []
    reports_base_url = raw_base + "data/Company%20Reports/"
    for item in sorted(selected_reports, key=lambda record: record["path"]):
        rel_path = item["path"]
        company = rel_path.split("/")[1]
        url = reports_base_url + quote(rel_path, safe="/")
        destination = args.out / "Company Reports" / Path(rel_path)
        download(url, destination, force=args.force, dry_run=args.dry_run)
        manifest_entries.append(
            {
                "company": company,
                "upstream_path": f"data/Company Reports/{rel_path}",
                "url": url,
                "local_path": str(destination),
                "upstream_size": item.get("size"),
                "sha256": None
                if args.dry_run or not destination.exists()
                else sha256_file(destination),
            }
        )

    manifest = {
        "source_repo": f"https://github.com/{REPO}",
        "ref": args.ref,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "companies": args.companies,
        "benchmark": {
            "url": benchmark_url,
            "local_path": str(benchmark_path),
            "sha256": None
            if args.dry_run or not benchmark_path.exists()
            else sha256_file(benchmark_path),
            "rows": len(benchmark_rows),
            "selected_questions": len(selected_questions),
        },
        "reports": manifest_entries,
    }

    manifest_path = args.out / "manifest.json"
    if args.dry_run:
        print(f"would write manifest -> {manifest_path}")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        print(f"wrote {manifest_path}")
        print(f"wrote {selected_questions_path}")
        print(f"selected reports: {len(selected_reports)}")
        print(f"selected questions: {len(selected_questions)}")

    missing = companies - {entry["company"] for entry in manifest_entries}
    if missing:
        print(f"missing selected companies: {', '.join(sorted(missing))}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
