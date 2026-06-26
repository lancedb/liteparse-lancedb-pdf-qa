#!/usr/bin/env python3
"""Evaluate answer correctness with a PydanticAI answer agent and LLM judge."""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from answer_eval import run_answer_eval
from config import EVAL_DIR
from rich.console import Console


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--questions", type=Path, default=EVAL_DIR / "selected_questions.jsonl")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, help="Development shortcut.")
    parser.add_argument("--out", type=Path, default=EVAL_DIR / "answer_eval_results.jsonl")
    parser.add_argument("--metrics", type=Path, default=Path("results/answer_eval_metrics.json"))
    parser.add_argument("--answer-model", default="gpt-5.4")
    parser.add_argument("--judge-model", default="gpt-5.4-mini")
    parser.add_argument(
        "--embedding-backend", choices=["hash", "sentence-transformers", "openai"], default="openai"
    )
    parser.add_argument("--text-model", default="text-embedding-3-small")
    parser.add_argument("--text-dimensions", type=int)
    parser.add_argument(
        "--max-evidence-images",
        type=int,
        default=3,
        help="Attach up to this many retrieved screenshot/asset images to the answer agent.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=6,
        help="Number of questions to evaluate concurrently.",
    )
    args = parser.parse_args()

    metrics = run_answer_eval(
        db_path=args.db,
        questions_path=args.questions,
        out_path=args.out,
        metrics_path=args.metrics,
        answer_model=args.answer_model,
        judge_model=args.judge_model,
        embedding_backend=args.embedding_backend,
        text_model=args.text_model,
        text_dimensions=args.text_dimensions,
        top_k=args.top_k,
        limit=args.limit,
        max_evidence_images=args.max_evidence_images,
        concurrency=args.concurrency,
    )
    Console().print(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
