#!/usr/bin/env python3
"""Parse selected ESG PDFs with LiteParse and write normalized JSONL outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from benchmark import load_questions
from config import EVAL_DIR, PARSED_DIR, RAW_DIR
from parse import parse_reports
from rich.console import Console


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--questions", type=Path, default=EVAL_DIR / "selected_questions.jsonl")
    parser.add_argument("--pages", choices=["labeled", "all"], default="labeled")
    parser.add_argument("--out", type=Path, default=PARSED_DIR)
    parser.add_argument("--companies", nargs="*", help="Optional company subset.")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--ocr", action="store_true", help="Enable OCR. Default disables OCR.")
    parser.add_argument("--max-docs", type=int, help="Development shortcut.")
    parser.add_argument("--metrics", type=Path, default=Path("results/extraction_performance.json"))
    args = parser.parse_args()

    questions = load_questions(args.questions)
    summary = parse_reports(
        raw_dir=args.raw_dir,
        questions=questions,
        out_dir=args.out,
        pages_mode=args.pages,
        companies=set(args.companies) if args.companies else None,
        no_ocr=not args.ocr,
        dpi=args.dpi,
        max_docs=args.max_docs,
    )
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(
        json.dumps(summary["extraction"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    Console().print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
