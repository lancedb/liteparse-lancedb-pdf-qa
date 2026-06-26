"""Arrow schemas for LanceDB tables."""

from __future__ import annotations

import pyarrow as pa


DEFAULT_TEXT_VECTOR_DIM = 384
IMAGE_VECTOR_DIM = 512


def vector_field(name: str, dim: int) -> pa.Field:
    return pa.field(name, pa.list_(pa.float32(), dim))


def blob_field(name: str = "blob") -> pa.Field:
    return pa.field(name, pa.large_binary(), metadata={b"lance-encoding:blob": b"true"})


def documents_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("doc_id", pa.string()),
            pa.field("company", pa.string()),
            pa.field("fiscal_year", pa.string()),
            pa.field("source_pdf", pa.string()),
            pa.field("local_path", pa.string()),
            pa.field("source_url", pa.string()),
            pa.field("sha256", pa.string()),
            pa.field("page_count", pa.int32()),
            pa.field("file_size", pa.int64()),
            pa.field("parse_config", pa.string()),
            pa.field("parsed_pages", pa.int32()),
            pa.field("parse_seconds", pa.float64()),
            pa.field("screenshot_seconds", pa.float64()),
            pa.field("timings", pa.string()),
        ]
    )


def pages_schema(text_vector_dim: int, image_vector_dim: int = IMAGE_VECTOR_DIM) -> pa.Schema:
    return pa.schema(
        [
            pa.field("page_id", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("company", pa.string()),
            pa.field("source_pdf", pa.string()),
            pa.field("page_num", pa.int32()),
            pa.field("width", pa.float64()),
            pa.field("height", pa.float64()),
            pa.field("text", pa.large_string()),
            pa.field("text_chars", pa.int32()),
            pa.field("screenshot_path", pa.string()),
            blob_field("screenshot_blob"),
            vector_field("text_vector", text_vector_dim),
            vector_field("image_vector", image_vector_dim),
        ]
    )


def chunks_schema(text_vector_dim: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("chunk_id", pa.string()),
            pa.field("page_id", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("company", pa.string()),
            pa.field("source_pdf", pa.string()),
            pa.field("page_num", pa.int32()),
            pa.field("chunk_index", pa.int32()),
            pa.field("text", pa.large_string()),
            vector_field("text_vector", text_vector_dim),
        ]
    )


def assets_schema(text_vector_dim: int, image_vector_dim: int = IMAGE_VECTOR_DIM) -> pa.Schema:
    return pa.schema(
        [
            pa.field("asset_id", pa.string()),
            pa.field("page_id", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("company", pa.string()),
            pa.field("source_pdf", pa.string()),
            pa.field("page_num", pa.int32()),
            pa.field("asset_type", pa.string()),
            pa.field("mime_type", pa.string()),
            pa.field("path", pa.string()),
            pa.field("text", pa.large_string()),
            blob_field("blob"),
            vector_field("text_vector", text_vector_dim),
            vector_field("image_vector", image_vector_dim),
        ]
    )


def eval_questions_schema(text_vector_dim: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("question_id", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("company", pa.string()),
            pa.field("fiscal_year", pa.string()),
            pa.field("source_pdf", pa.string()),
            pa.field("question", pa.large_string()),
            pa.field("expected_answer", pa.large_string()),
            pa.field("expected_pages", pa.list_(pa.int32())),
            pa.field("expected_page_ids", pa.list_(pa.string())),
            pa.field("extract_type", pa.string()),
            pa.field("question_type", pa.string()),
            pa.field("required_modality", pa.string()),
            pa.field("difficulty", pa.string()),
            vector_field("question_vector", text_vector_dim),
        ]
    )
