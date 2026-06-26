#!/usr/bin/env python3
"""Summarize retrieval JSONL results by benchmark metadata fields."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
from config import EVAL_DIR
from rich.console import Console


METRICS = ["any_page_hit", "page_coverage", "all_pages_hit", "modality_hit"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=EVAL_DIR / "retrieval_results.jsonl")
    parser.add_argument("--out", type=Path, default=Path("results/retrieval_breakdowns.json"))
    parser.add_argument("--label", default="retrieval")
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.results.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = {
        "label": args.label,
        "by_company": summarize(rows, ["retrieval_mode", "company"]),
        "by_question_type": summarize(rows, ["retrieval_mode", "question_type"]),
        "by_modality": summarize(rows, ["retrieval_mode", "expected_modality"]),
        "by_difficulty": summarize(rows, ["retrieval_mode", "difficulty"]),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    Console().print({"wrote": str(args.out)})
    return 0


def summarize(rows: list[dict[str, Any]], group_fields: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(row.get(field) for field in group_fields)
        groups[key].append(row)

    output = []
    for key, items in sorted(groups.items()):
        record = {field: value for field, value in zip(group_fields, key, strict=True)}
        record["questions"] = len(items)
        for metric in METRICS:
            name = f"{metric}_rate" if metric != "page_coverage" else metric
            record[name] = sum(float(item[metric]) for item in items) / len(items)
        latencies = sorted(float(item["latency_ms"]) for item in items)
        record["latency_ms_p50"] = latencies[len(latencies) // 2]
        output.append(record)
    return output


if __name__ == "__main__":
    raise SystemExit(main())
