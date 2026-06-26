"""LiteParse extraction and normalized page/chunk/asset output."""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from liteparse import LiteParse

from benchmark import labeled_pages_by_pdf, write_jsonl
from reports import enrich_report, report_entries


def compress_pages(pages: Iterable[int]) -> str:
    ordered = sorted(set(int(page) for page in pages))
    if not ordered:
        return ""
    ranges: list[str] = []
    start = prev = ordered[0]
    for page in ordered[1:]:
        if page == prev + 1:
            prev = page
            continue
        ranges.append(f"{start}-{prev}" if start != prev else str(start))
        start = prev = page
    ranges.append(f"{start}-{prev}" if start != prev else str(start))
    return ",".join(ranges)


def chunk_page_text(text: str, *, max_chars: int = 1200, overlap: int = 150) -> list[str]:
    text = " ".join((text or "").split())
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            split = text.rfind(" ", start, end)
            if split > start + max_chars // 2:
                end = split
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(0, end - overlap)
    return [chunk for chunk in chunks if chunk]


@dataclass
class TimingCollector:
    stages: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def stage(self, name: str):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.stages[name] = self.stages.get(name, 0.0) + (time.perf_counter() - started)


def _dump_liteparse_json(result: Any, path: Path) -> None:
    """Persist the structured parse result for offline inspection."""
    raw = {
        "pages": [
            {
                "page": page.page_num,
                "width": page.width,
                "height": page.height,
                "text": page.text,
                "text_items": [asdict(item) for item in page.text_items],
            }
            for page in result.pages
        ],
        "images": [
            {"id": image.id, "page": image.page, "format": image.format} for image in result.images
        ],
    }
    path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")


def parse_reports(
    *,
    raw_dir: Path,
    questions: list[dict[str, Any]],
    out_dir: Path,
    pages_mode: str,
    companies: set[str] | None = None,
    no_ocr: bool = True,
    dpi: int = 150,
    max_docs: int | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    labeled = labeled_pages_by_pdf(questions)
    selected_reports = []
    for entry in report_entries(raw_dir):
        if companies and entry["company"] not in companies:
            continue
        selected_reports.append(enrich_report(entry, questions))
    if max_docs is not None:
        selected_reports = selected_reports[:max_docs]

    manifest_rows = []
    for report in selected_reports:
        manifest_rows.append(
            parse_one_report(
                report=report,
                out_dir=out_dir,
                target_pages=None
                if pages_mode == "all"
                else labeled.get((report["company"], report["source_pdf"]), set()),
                no_ocr=no_ocr,
                dpi=dpi,
            )
        )
    write_jsonl(out_dir / "manifest.jsonl", manifest_rows)
    summary = summarize_extraction(manifest_rows)
    (out_dir / "extraction_metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return {"documents": len(manifest_rows), "out_dir": str(out_dir), "extraction": summary}


def parse_one_report(
    *,
    report: dict[str, Any],
    out_dir: Path,
    target_pages: Iterable[int] | None,
    no_ocr: bool,
    dpi: int,
) -> dict[str, Any]:
    total_started = time.perf_counter()
    timings = TimingCollector()
    pdf_path = Path(report["local_path"])
    document_dir = out_dir / "documents" / report["doc_id"]
    screenshots_dir = document_dir / "screenshots"
    assets_dir = document_dir / "extracted_assets"
    document_dir.mkdir(parents=True, exist_ok=True)
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    target = compress_pages(target_pages or [])
    # Configure the parser once, then run it in-process via the native Python SDK.
    parser = LiteParse(
        ocr_enabled=not no_ocr,
        dpi=dpi,
        image_mode="embed",
        target_pages=target or None,
        output_format="json",
        quiet=True,
    )

    with timings.stage("liteparse_parse_seconds"):
        result = parser.parse(pdf_path)
    _dump_liteparse_json(result, document_dir / "liteparse.json")
    parsed_pages = [page.page_num for page in result.pages]

    with timings.stage("liteparse_screenshot_seconds"):
        screenshots = parser.screenshot(pdf_path, page_numbers=parsed_pages or None)

    with timings.stage("write_normalized_outputs_seconds"):
        # Persist screenshot bytes; map page_num -> on-disk path for later rows.
        screenshot_paths: dict[int, Path] = {}
        for shot in screenshots:
            shot_path = screenshots_dir / f"page_{shot.page_num}.png"
            shot_path.write_bytes(shot.image_bytes)
            screenshot_paths[shot.page_num] = shot_path

        page_rows = []
        chunk_rows = []
        for page in result.pages:
            page_num = page.page_num
            page_id = f"{report['doc_id']}:p{page_num}"
            screenshot_path = screenshot_paths.get(page_num)
            text = page.text or ""
            page_rows.append(
                {
                    "page_id": page_id,
                    "doc_id": report["doc_id"],
                    "company": report["company"],
                    "source_pdf": report["source_pdf"],
                    "page_num": page_num,
                    "width": page.width,
                    "height": page.height,
                    "text": text,
                    "text_chars": len(text),
                    "screenshot_path": str(screenshot_path) if screenshot_path else None,
                }
            )
            for index, chunk in enumerate(chunk_page_text(text)):
                chunk_rows.append(
                    {
                        "chunk_id": f"{page_id}:c{index}",
                        "page_id": page_id,
                        "doc_id": report["doc_id"],
                        "company": report["company"],
                        "source_pdf": report["source_pdf"],
                        "page_num": page_num,
                        "chunk_index": index,
                        "text": chunk,
                    }
                )

        # The `assets` table holds extracted figures only. Whole-page screenshots
        # live on the `pages` table (as a blob + page image vector), so we don't
        # duplicate their bytes here.
        asset_rows = []
        for image in result.images:
            page_id = f"{report['doc_id']}:p{image.page}"
            stem = f"image_{image.id}"
            image_path = assets_dir / f"{stem}.{image.format}"
            image_path.write_bytes(image.bytes)
            asset_rows.append(
                {
                    "asset_id": f"{page_id}:asset:{stem}",
                    "page_id": page_id,
                    "doc_id": report["doc_id"],
                    "company": report["company"],
                    "source_pdf": report["source_pdf"],
                    "page_num": image.page,
                    "asset_type": "extracted_image",
                    "mime_type": f"image/{image.format}",
                    "path": str(image_path),
                    "text": stem.replace("_", " "),
                }
            )

        write_jsonl(document_dir / "pages.jsonl", page_rows)
        write_jsonl(document_dir / "chunks.jsonl", chunk_rows)
        write_jsonl(document_dir / "assets.jsonl", asset_rows)

    parse_config = {
        "engine": "liteparse-python-sdk",
        "format": "json",
        "no_ocr": no_ocr,
        "dpi": dpi,
        "image_mode": "embed",
        "target_pages": target or "all",
    }
    return {
        **report,
        "parse_config": parse_config,
        "parsed_pages": len(page_rows),
        "chunk_rows": len(chunk_rows),
        "asset_rows": len(asset_rows),
        "parse_seconds": timings.stages.get("liteparse_parse_seconds", 0.0),
        "screenshot_seconds": timings.stages.get("liteparse_screenshot_seconds", 0.0),
        "total_extraction_seconds": time.perf_counter() - total_started,
        "timings": timings.stages,
        "pages_per_second": len(page_rows) / timings.stages.get("liteparse_parse_seconds", 1.0),
        "document_dir": str(document_dir),
    }


def summarize_extraction(rows: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {
        "documents": len(rows),
        "pages": sum(int(row.get("parsed_pages", 0)) for row in rows),
        "chunks": sum(int(row.get("chunk_rows", 0)) for row in rows),
        "assets": sum(int(row.get("asset_rows", 0)) for row in rows),
        "liteparse_parse_seconds": sum(float(row.get("parse_seconds", 0.0)) for row in rows),
        "liteparse_screenshot_seconds": sum(
            float(row.get("screenshot_seconds", 0.0)) for row in rows
        ),
        "total_extraction_seconds": sum(
            float(row.get("total_extraction_seconds", 0.0)) for row in rows
        ),
    }
    totals["parse_pages_per_second"] = (
        totals["pages"] / totals["liteparse_parse_seconds"]
        if totals["liteparse_parse_seconds"]
        else 0.0
    )
    totals["screenshot_pages_per_second"] = (
        totals["pages"] / totals["liteparse_screenshot_seconds"]
        if totals["liteparse_screenshot_seconds"]
        else 0.0
    )
    totals["end_to_end_pages_per_second"] = (
        totals["pages"] / totals["total_extraction_seconds"]
        if totals["total_extraction_seconds"]
        else 0.0
    )
    totals["by_document"] = [
        {
            "doc_id": row["doc_id"],
            "company": row["company"],
            "source_pdf": row["source_pdf"],
            "pages": row.get("parsed_pages", 0),
            "chunks": row.get("chunk_rows", 0),
            "assets": row.get("asset_rows", 0),
            "liteparse_parse_seconds": row.get("parse_seconds", 0.0),
            "liteparse_screenshot_seconds": row.get("screenshot_seconds", 0.0),
            "total_extraction_seconds": row.get("total_extraction_seconds", 0.0),
            "parse_pages_per_second": row.get("pages_per_second", 0.0),
        }
        for row in rows
    ]
    return totals
