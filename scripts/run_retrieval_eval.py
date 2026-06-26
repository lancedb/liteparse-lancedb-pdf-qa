#!/usr/bin/env python3
"""Run retrieval modes and score benchmark page-label hits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from benchmark import load_questions
from config import EVAL_DIR, LANCEDB_DIR
from embeddings import ImageEmbedder, TextEmbedder
from eval import score_result, summarize_results, write_results
from retrieval import Retriever
from rich.console import Console


def default_text_model(backend: str) -> str:
    if backend == "openai":
        return "text-embedding-3-small"
    if backend == "sentence-transformers":
        return "sentence-transformers/all-MiniLM-L6-v2"
    return "hash"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=LANCEDB_DIR)
    parser.add_argument("--questions", type=Path, default=EVAL_DIR / "selected_questions.jsonl")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out", type=Path, default=EVAL_DIR / "retrieval_results.jsonl")
    parser.add_argument("--metrics", type=Path, default=Path("results/metrics.json"))
    parser.add_argument(
        "--modes", nargs="+", default=["chunks", "pages", "assets", "images", "hybrid_bundle"]
    )
    parser.add_argument("--limit", type=int, help="Development shortcut.")
    parser.add_argument(
        "--embedding-backend", choices=["hash", "sentence-transformers", "openai"], default="hash"
    )
    parser.add_argument("--text-model", help="Text embedding model.")
    parser.add_argument(
        "--text-dimensions",
        type=int,
        help="Optional text vector dimensions. Must match the built LanceDB table.",
    )
    parser.add_argument("--image-embedding-backend", choices=["hash", "open-clip"], default="hash")
    parser.add_argument("--clip-model", default="ViT-B-32")
    parser.add_argument("--clip-pretrained", default="laion2b_s34b_b79k")
    args = parser.parse_args()
    text_model = args.text_model or default_text_model(args.embedding_backend)

    questions = load_questions(args.questions)
    if args.limit:
        questions = questions[: args.limit]

    retriever = Retriever(
        args.db,
        TextEmbedder(args.embedding_backend, text_model, args.text_dimensions),
        ImageEmbedder(
            backend=args.image_embedding_backend,
            model_name=args.clip_model,
            pretrained=args.clip_pretrained,
        ),
    )
    rows = []
    for question in questions:
        query_vector = retriever.text_embedder.embed(question["question"])
        for mode in args.modes:
            rows.append(
                score_result(
                    question,
                    retriever.retrieve(
                        question, mode=mode, top_k=args.top_k, query_vector=query_vector
                    ),
                )
            )

    write_results(args.out, rows)
    summary = summarize_results(rows)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    Console().print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
