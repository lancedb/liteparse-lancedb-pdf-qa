"""Retrieval scoring helpers."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from benchmark import expected_pages_for_question


def score_result(question: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    expected_pages = expected_pages_for_question(question)
    hit_pages = {int(hit["page_num"]) for hit in result["top_k"]}
    expected_hit_pages = expected_pages & hit_pages
    expected_modality = question.get("required_modality") or "text"
    modality_hit = True
    if expected_modality in {"table", "figure"}:
        modality_hit = any(
            int(hit["page_num"]) in expected_pages
            and (
                hit.get("asset_type") in {expected_modality, "extracted_image", "page_screenshot"}
                or hit.get("source_table") == "assets"
            )
            for hit in result["top_k"]
        )
    return {
        **result,
        "company": question["company"],
        "question_type": question.get("question_type"),
        "difficulty": question.get("difficulty"),
        "expected_pages": sorted(expected_pages),
        "expected_modality": expected_modality,
        "any_page_hit": bool(expected_hit_pages),
        "page_coverage": len(expected_hit_pages) / len(expected_pages) if expected_pages else 0.0,
        "all_pages_hit": expected_pages <= hit_pages if expected_pages else False,
        "modality_hit": modality_hit,
    }


def write_results(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_mode[row["retrieval_mode"]].append(row)

    summary = {}
    for mode, mode_rows in sorted(by_mode.items()):
        latencies = sorted(float(row["latency_ms"]) for row in mode_rows)
        summary[mode] = {
            "questions": len(mode_rows),
            "any_page_hit_rate": _mean(row["any_page_hit"] for row in mode_rows),
            "page_coverage": _mean(row["page_coverage"] for row in mode_rows),
            "all_pages_hit_rate": _mean(row["all_pages_hit"] for row in mode_rows),
            "modality_hit_rate": _mean(row["modality_hit"] for row in mode_rows),
            "latency_ms_p50": statistics.median(latencies) if latencies else 0.0,
            "latency_ms_p95": _percentile(latencies, 0.95),
        }
    return summary


def _mean(values: Iterable[float | bool]) -> float:
    items = [float(value) for value in values]
    return sum(items) / len(items) if items else 0.0


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    index = min(len(values) - 1, round((len(values) - 1) * percentile))
    return values[index]
