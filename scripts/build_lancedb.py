#!/usr/bin/env python3
"""Build LanceDB tables from parsed LiteParse JSONL outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from config import EVAL_DIR, LANCEDB_DIR, PARSED_DIR
from embeddings import ImageEmbedder, TextEmbedder
from index import build_lancedb
from rich.console import Console


def default_text_model(backend: str) -> str:
    if backend == "openai":
        return "text-embedding-3-small"
    if backend == "sentence-transformers":
        return "sentence-transformers/all-MiniLM-L6-v2"
    return "hash"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parsed-dir", type=Path, default=PARSED_DIR)
    parser.add_argument("--questions", type=Path, default=EVAL_DIR / "selected_questions.jsonl")
    parser.add_argument("--db", type=Path, default=LANCEDB_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--embedding-backend", choices=["hash", "sentence-transformers", "openai"], default="hash"
    )
    parser.add_argument(
        "--text-model",
        help="Text embedding model. Defaults to text-embedding-3-small for OpenAI.",
    )
    parser.add_argument(
        "--text-dimensions",
        type=int,
        help="Optional text vector dimensions. Supported by OpenAI text-embedding-3 models.",
    )
    parser.add_argument("--image-embedding-backend", choices=["hash", "open-clip"], default="hash")
    parser.add_argument("--clip-model", default="ViT-B-32")
    parser.add_argument("--clip-pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--metrics", type=Path, default=Path("results/storage_metrics.json"))
    args = parser.parse_args()
    text_model = args.text_model or default_text_model(args.embedding_backend)

    summary = build_lancedb(
        parsed_dir=args.parsed_dir,
        questions_path=args.questions,
        db_path=args.db,
        overwrite=args.overwrite,
        text_embedder=TextEmbedder(args.embedding_backend, text_model, args.text_dimensions),
        image_embedder=ImageEmbedder(
            backend=args.image_embedding_backend,
            model_name=args.clip_model,
            pretrained=args.clip_pretrained,
        ),
    )
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    Console().print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
