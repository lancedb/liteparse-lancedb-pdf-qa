"""Shared paths and small filesystem/normalization helpers for the demo pipeline."""

from __future__ import annotations

import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "climate_finance_bench"
EVAL_DIR = DATA_DIR / "eval"
PARSED_DIR = DATA_DIR / "parsed" / "liteparse"
LANCEDB_DIR = DATA_DIR / "lancedb" / "esg_pdf_qa.lancedb"


def slugify(value: str) -> str:
    """Return a filesystem/database friendly slug."""
    value = value.lower().replace("&", "and")
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return re.sub(r"_+", "_", value).strip("_")


def doc_id(company: str, fiscal_year: int | str, source_pdf: str) -> str:
    """Build a stable document id from benchmark/report metadata."""
    year = str(fiscal_year)
    pdf_stem = Path(source_pdf).stem
    return f"{slugify(company)}_{year}_{slugify(pdf_stem)[:48]}"


def directory_size_bytes(path: Path) -> int:
    """Total on-disk size of a directory tree, used to report the LanceDB store size."""
    if not path.exists():
        return 0
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())
