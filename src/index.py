"""Build LanceDB tables from normalized LiteParse outputs."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import lancedb
import pyarrow as pa

from benchmark import load_questions, read_jsonl
from config import directory_size_bytes
from embeddings import ImageEmbedder, TextEmbedder
from schema import (
    assets_schema,
    chunks_schema,
    documents_schema,
    eval_questions_schema,
    pages_schema,
)


def _table(rows: list[dict[str, Any]], schema: pa.Schema) -> pa.Table:
    return pa.Table.from_pylist(rows, schema=schema)


def build_lancedb(
    *,
    parsed_dir: Path,
    questions_path: Path,
    db_path: Path,
    overwrite: bool,
    text_embedder: TextEmbedder,
    image_embedder: ImageEmbedder,
) -> dict[str, Any]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(db_path)
    mode = "overwrite" if overwrite else "create"

    manifest_rows = read_jsonl(parsed_dir / "manifest.jsonl")
    documents = [_document_row(row) for row in manifest_rows]

    pages = []
    chunks = []
    assets = []
    for document in manifest_rows:
        document_dir = Path(document["document_dir"])
        pages.extend(read_jsonl(document_dir / "pages.jsonl"))
        chunks.extend(read_jsonl(document_dir / "chunks.jsonl"))
        assets.extend(read_jsonl(document_dir / "assets.jsonl"))

    _add_page_embeddings(pages, text_embedder, image_embedder)
    _add_chunk_embeddings(chunks, text_embedder)
    _add_asset_embeddings(assets, text_embedder, image_embedder)

    questions = load_questions(questions_path)
    _add_question_embeddings(questions, text_embedder)

    started = time.perf_counter()
    text_vector_dim = text_embedder.dimensions
    image_vector_dim = image_embedder.dimensions
    if text_vector_dim is None:
        raise ValueError("text embedder did not expose a vector dimension")

    tables = {
        "documents": db.create_table(
            "documents", data=_table(documents, documents_schema()), mode=mode
        ),
        "pages": db.create_table(
            "pages", data=_table(pages, pages_schema(text_vector_dim, image_vector_dim)), mode=mode
        ),
        "chunks": db.create_table(
            "chunks", data=_table(chunks, chunks_schema(text_vector_dim)), mode=mode
        ),
        "assets": db.create_table(
            "assets",
            data=_table(assets, assets_schema(text_vector_dim, image_vector_dim)),
            mode=mode,
        ),
        "eval_questions": db.create_table(
            "eval_questions",
            data=_table(questions, eval_questions_schema(text_vector_dim)),
            mode=mode,
        ),
    }
    index_seconds = _create_indexes(tables)

    return {
        "db_path": str(db_path),
        "row_counts": {
            "documents": len(documents),
            "pages": len(pages),
            "chunks": len(chunks),
            "assets": len(assets),
            "eval_questions": len(questions),
        },
        "embedding": {
            "text_backend": text_embedder.backend,
            "text_model": text_embedder.model_name,
            "text_dimensions": text_vector_dim,
            "image_backend": image_embedder.backend,
            "image_model": image_embedder.model_name,
            "image_pretrained": image_embedder.pretrained,
            "image_dimensions": image_vector_dim,
        },
        "write_seconds": time.perf_counter() - started,
        "index_seconds": index_seconds,
        "db_size_bytes": directory_size_bytes(db_path),
    }


def _document_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "doc_id": row["doc_id"],
        "company": row["company"],
        "fiscal_year": str(row["fiscal_year"]),
        "source_pdf": row["source_pdf"],
        "local_path": row["local_path"],
        "source_url": row.get("url"),
        "sha256": row.get("sha256"),
        "page_count": int(row["page_count"]),
        "file_size": int(row["file_size"]),
        "parse_config": json.dumps(row.get("parse_config", {}), sort_keys=True),
        "parsed_pages": int(row["parsed_pages"]),
        "parse_seconds": float(row["parse_seconds"]),
        "screenshot_seconds": float(row["screenshot_seconds"]),
        "timings": json.dumps(row.get("timings", {}), sort_keys=True),
    }


def _add_page_embeddings(
    rows: list[dict[str, Any]], text_embedder: TextEmbedder, image_embedder: ImageEmbedder
) -> None:
    vectors = text_embedder.embed_many(row.get("text") or "" for row in rows)
    for row, vector in zip(rows, vectors, strict=True):
        screenshot_path = row.get("screenshot_path")
        blob = Path(screenshot_path).read_bytes() if screenshot_path else None
        row["screenshot_blob"] = blob
        row["text_vector"] = vector
        row["image_vector"] = image_embedder.embed_blob(blob)


def _add_chunk_embeddings(rows: list[dict[str, Any]], text_embedder: TextEmbedder) -> None:
    vectors = text_embedder.embed_many(row.get("text") or "" for row in rows)
    for row, vector in zip(rows, vectors, strict=True):
        row["text_vector"] = vector


def _add_asset_embeddings(
    rows: list[dict[str, Any]], text_embedder: TextEmbedder, image_embedder: ImageEmbedder
) -> None:
    vectors = text_embedder.embed_many(row.get("text") or "" for row in rows)
    for row, vector in zip(rows, vectors, strict=True):
        path = Path(row["path"])
        blob = path.read_bytes() if path.exists() else None
        row["blob"] = blob
        row["text_vector"] = vector
        row["image_vector"] = image_embedder.embed_blob(blob)


def _add_question_embeddings(rows: list[dict[str, Any]], text_embedder: TextEmbedder) -> None:
    vectors = text_embedder.embed_many(row.get("question") or "" for row in rows)
    for row, vector in zip(rows, vectors, strict=True):
        row["fiscal_year"] = str(row["fiscal_year"])
        row["question_vector"] = vector


def _create_indexes(tables: dict[str, Any]) -> float:
    started = time.perf_counter()
    searchable = ("pages", "chunks", "assets")

    # Scalar (BTREE) indexes on the columns every query prefilters on. They keep
    # the company/source_pdf prefilter fast and stay portable across local and
    # remote tables.
    for table_name in searchable:
        table = tables[table_name]
        table.create_scalar_index("company", replace=True)
        table.create_scalar_index("source_pdf", replace=True)

    # Full-text indexes back keyword search over the text columns.
    for table_name in searchable:
        tables[table_name].create_fts_index("text", replace=True)

    # ANN vector indexes only pay off once a table is large enough to train one.
    # Below ~256 rows LanceDB runs an exact search, which is both faster and
    # exact at this scale, so we build indexes only where they actually help.
    vector_specs = [
        ("pages", "text_vector"),
        ("pages", "image_vector"),
        ("chunks", "text_vector"),
        ("assets", "text_vector"),
        ("assets", "image_vector"),
    ]
    for table_name, vector_column in vector_specs:
        table = tables[table_name]
        if table.count_rows() >= 256:
            table.create_index(vector_column_name=vector_column, metric="cosine", replace=True)

    return time.perf_counter() - started
