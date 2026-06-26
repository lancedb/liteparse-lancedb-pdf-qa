"""Report manifest loading and document metadata helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import fitz

from config import doc_id


def load_manifest(raw_dir: Path) -> dict[str, Any]:
    return json.loads((raw_dir / "manifest.json").read_text(encoding="utf-8"))


def report_entries(raw_dir: Path) -> list[dict[str, Any]]:
    manifest = load_manifest(raw_dir)
    reports = []
    for entry in manifest.get("reports", []):
        local_path = Path(entry["local_path"])
        if not local_path.is_absolute():
            local_path = raw_dir.parents[2] / local_path
        reports.append({**entry, "local_path": str(local_path)})
    return reports


def fiscal_year_by_pdf(questions: list[dict[str, Any]]) -> dict[tuple[str, str], int]:
    years: dict[tuple[str, str], int] = {}
    for question in questions:
        years[(question["company"], question["source_pdf"])] = int(question["fiscal_year"])
    return years


def enrich_report(entry: dict[str, Any], questions: list[dict[str, Any]]) -> dict[str, Any]:
    pdf_path = Path(entry["local_path"])
    years = fiscal_year_by_pdf(questions)
    fiscal_year = years.get((entry["company"], pdf_path.name), "unknown")
    with fitz.open(pdf_path) as document:
        page_count = document.page_count
    return {
        **entry,
        "source_pdf": pdf_path.name,
        "fiscal_year": fiscal_year,
        "doc_id": doc_id(entry["company"], fiscal_year, pdf_path.name),
        "page_count": page_count,
        "file_size": pdf_path.stat().st_size,
    }
