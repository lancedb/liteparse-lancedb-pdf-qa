"""Retrieval modes over LanceDB tables."""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lancedb

from embeddings import ImageEmbedder, TextEmbedder


# Columns selected per table. Page identity travels on every hit so we can merge.
_COLUMNS: dict[str, list[str]] = {
    "chunks": ["chunk_id", "page_id", "doc_id", "company", "source_pdf", "page_num", "text"],
    "pages": ["page_id", "doc_id", "company", "source_pdf", "page_num", "text", "screenshot_path"],
    "assets": [
        "asset_id",
        "page_id",
        "doc_id",
        "company",
        "source_pdf",
        "page_num",
        "asset_type",
        "path",
        "text",
    ],
}


@dataclass
class Retriever:
    db_path: Path
    text_embedder: TextEmbedder
    image_embedder: ImageEmbedder | None = None

    def __post_init__(self) -> None:
        self.db = lancedb.connect(self.db_path)

    def retrieve(
        self,
        question: dict[str, Any],
        *,
        mode: str,
        top_k: int,
        query_vector: list[float] | None = None,
    ) -> dict[str, Any]:
        embedding_started = time.perf_counter()
        query_vector = query_vector or self.text_embedder.embed(question["question"])
        embedding_ms = (time.perf_counter() - embedding_started) * 1000

        started = time.perf_counter()
        if mode in ("chunks", "pages", "assets"):
            hits = self._search_table(mode, query_vector, top_k, question, _COLUMNS[mode])
        elif mode == "hybrid_bundle":
            hits = self._hybrid_bundle(question, query_vector, top_k)
        elif mode == "images":
            hits = self._image_bundle(question, top_k)
        else:
            raise ValueError(f"unknown retrieval mode: {mode}")
        return {
            "question_id": question["question_id"],
            "retrieval_mode": mode,
            "top_k": hits[:top_k],
            "latency_ms": (time.perf_counter() - started) * 1000,
            "query_embedding_ms": embedding_ms,
        }

    def _search_table(
        self,
        table_name: str,
        query_vector: list[float],
        top_k: int,
        question: dict[str, Any],
        columns: list[str],
        *,
        vector_column: str = "text_vector",
    ) -> list[dict[str, Any]]:
        table = self.db.open_table(table_name)
        rows = (
            table.search(query_vector, vector_column_name=vector_column)
            .where(_question_filter(question), prefilter=True)
            .select([*columns, "_distance"])
            .limit(top_k)
            .to_list()
        )
        return [_normalize_hit(table_name, row) for row in rows]

    def _hybrid_bundle(
        self, question: dict[str, Any], query_vector: list[float], top_k: int
    ) -> list[dict[str, Any]]:
        candidates = []
        for table_name in ("chunks", "pages", "assets"):
            candidates.extend(
                self._search_table(
                    table_name, query_vector, top_k * 2, question, _COLUMNS[table_name]
                )
            )
        return _merge_by_page(candidates, top_k)

    def _image_bundle(self, question: dict[str, Any], top_k: int) -> list[dict[str, Any]]:
        if self.image_embedder is None:
            raise ValueError("images mode requires an image embedder")
        # Embed the text question into CLIP's image space and search the image vectors
        # on whole pages and on extracted figures, then merge by page.
        image_query = self.image_embedder.embed_text_query(question["question"])
        candidates = []
        for table_name in ("pages", "assets"):
            candidates.extend(
                self._search_table(
                    table_name,
                    image_query,
                    top_k * 2,
                    question,
                    _COLUMNS[table_name],
                    vector_column="image_vector",
                )
            )
        return _merge_by_page(candidates, top_k)

    def page_screenshots(self, page_ids: list[str], max_images: int) -> list[tuple[str, bytes]]:
        """Read page-screenshot bytes from the LanceDB blob column for the given pages.

        Uses Lance's blob API so the bytes are only materialized for these few pages,
        never scanned during search. Local/OSS path (relies on ``to_lance()``).
        """
        ids = list(dict.fromkeys(pid for pid in page_ids if pid))[:max_images]
        if not ids:
            return []
        dataset = self.db.open_table("pages").to_lance()
        quoted = ", ".join("'" + pid.replace("'", "''") + "'" for pid in ids)
        located = dataset.to_table(
            columns=["page_id"], filter=f"page_id IN ({quoted})", with_row_address=True
        )
        found_ids = located.column("page_id").to_pylist()
        addresses = located.column("_rowaddr").to_pylist()
        if not addresses:
            return []
        blobs = dataset.take_blobs("screenshot_blob", addresses=addresses)
        by_id = {pid: blob.read() for pid, blob in zip(found_ids, blobs)}
        return [(pid, by_id[pid]) for pid in ids if pid in by_id]


def _merge_by_page(candidates: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    grouped: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for hit in sorted(candidates, key=lambda item: item["distance"]):
        key = hit["page_id"]
        if key not in grouped:
            grouped[key] = {**hit, "source_tables": [hit["source_table"]], "bundle_hits": [hit]}
        else:
            grouped[key]["source_tables"].append(hit["source_table"])
            grouped[key]["bundle_hits"].append(hit)
            grouped[key]["score"] = max(grouped[key]["score"], hit["score"])
            grouped[key]["distance"] = min(grouped[key]["distance"], hit["distance"])
    return list(grouped.values())[:top_k]


def _question_filter(question: dict[str, Any]) -> str:
    company = str(question["company"]).replace("'", "''")
    source_pdf = str(question["source_pdf"]).replace("'", "''")
    return f"company = '{company}' AND source_pdf = '{source_pdf}'"


def _normalize_hit(table_name: str, row: dict[str, Any]) -> dict[str, Any]:
    distance = float(row.pop("_distance", 0.0))
    text = row.get("text") or ""
    return {
        **row,
        "source_table": table_name,
        "asset_type": row.get("asset_type")
        or ("page_screenshot" if table_name == "pages" else "text"),
        "distance": distance,
        "score": 1.0 / (1.0 + max(distance, 0.0)),
        "text_preview": text[:500],
    }
