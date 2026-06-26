"""Climate Finance Bench loading, normalization checks, and summaries."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from config import doc_id


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_questions(path: Path) -> list[dict[str, Any]]:
    questions = read_jsonl(path)
    for row in questions:
        row["doc_id"] = doc_id(row["company"], row["fiscal_year"], row["source_pdf"])
        row["expected_page_ids"] = [
            f"{row['doc_id']}:p{page_num}" for page_num in row.get("expected_pages", [])
        ]
    return questions


def labeled_pages_by_pdf(questions: Iterable[dict[str, Any]]) -> dict[tuple[str, str], set[int]]:
    pages: dict[tuple[str, str], set[int]] = defaultdict(set)
    for row in questions:
        key = (row["company"], row["source_pdf"])
        pages[key].update(int(page) for page in row.get("expected_pages", []))
    return dict(pages)


def summarize_questions(questions: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(questions)
    by_company = Counter(row["company"] for row in rows)
    by_extract_type = Counter(row.get("extract_type") or "unknown" for row in rows)
    by_modality = Counter(row.get("required_modality") or "unknown" for row in rows)
    by_difficulty = Counter(row.get("difficulty") or "unknown" for row in rows)
    return {
        "question_count": len(rows),
        "company_counts": dict(sorted(by_company.items())),
        "extract_type_counts": dict(sorted(by_extract_type.items())),
        "modality_counts": dict(sorted(by_modality.items())),
        "difficulty_counts": dict(sorted(by_difficulty.items())),
    }


def expected_pages_for_question(question: dict[str, Any]) -> set[int]:
    return {int(page) for page in question.get("expected_pages", [])}
