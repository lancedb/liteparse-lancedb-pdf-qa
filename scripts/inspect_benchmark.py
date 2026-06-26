#!/usr/bin/env python3
"""Inspect the selected Climate Finance Bench subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from benchmark import load_questions, summarize_questions
from config import EVAL_DIR, RAW_DIR
from reports import enrich_report, report_entries
from rich.console import Console
from rich.table import Table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--questions", type=Path, default=EVAL_DIR / "selected_questions.jsonl")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON summary.")
    args = parser.parse_args()

    questions = load_questions(args.questions)
    reports = [enrich_report(entry, questions) for entry in report_entries(args.raw_dir)]
    summary = summarize_questions(questions)
    summary["reports"] = [
        {
            "company": report["company"],
            "source_pdf": report["source_pdf"],
            "page_count": report["page_count"],
            "file_size": report["file_size"],
            "doc_id": report["doc_id"],
        }
        for report in reports
    ]

    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    console = Console()
    console.print(f"[bold]Selected questions:[/bold] {summary['question_count']}")

    company_table = Table(title="Questions by Company")
    company_table.add_column("Company")
    company_table.add_column("Questions", justify="right")
    for company, count in summary["company_counts"].items():
        company_table.add_row(company, str(count))
    console.print(company_table)

    report_table = Table(title="Selected Reports")
    report_table.add_column("Company")
    report_table.add_column("Pages", justify="right")
    report_table.add_column("Size MB", justify="right")
    report_table.add_column("PDF")
    for report in reports:
        report_table.add_row(
            report["company"],
            str(report["page_count"]),
            f"{report['file_size'] / (1024 * 1024):.1f}",
            report["source_pdf"],
        )
    console.print(report_table)

    modality_table = Table(title="Question Labels")
    modality_table.add_column("Group")
    modality_table.add_column("Counts")
    for group in ["extract_type_counts", "modality_counts", "difficulty_counts"]:
        modality_table.add_row(group, json.dumps(summary[group], ensure_ascii=False))
    console.print(modality_table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
