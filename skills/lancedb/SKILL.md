---
name: lancedb
description: Use when writing, reviewing, debugging, or documenting LanceDB pipelines in Python or TypeScript, especially code that should work across local LanceDB OSS tables and remote LanceDB Enterprise/Cloud tables. Helps avoid non-portable full-table materialization, choose idiomatic query/search patterns, and apply LanceDB performance defaults for ingestion, indexing, filtering, and diagnostics.
---

# Building LanceDB Pipelines

Use this skill to produce LanceDB pipelines that are portable between local and remote tables (for LanceDB Enterprise/Cloud) and idiomatic for the selected SDK.

## LanceDB Table Modes

LanceDB has two common execution modes:

- **Local table**: embedded, open source, in-process LanceDB. The client opens data from a local path or object storage URI and executes queries in the application process.
- **Remote table**: LanceDB Enterprise/Cloud table opened through a `db://...` URI. The data may be very large, commonly backed by object storage, and queried through a remote service.

Do NOT assume local-only table helpers exist on remote tables. If the user asks for LanceDB Enterprise, Cloud, `db://...`, production remote access, or a remote table, focus on the remote table path: use `search()` / `query()`, keep reads bounded with `select()` and `limit()`, and avoid table-level full materialization APIs.

## Workflow

1. Identify the SDK: Python, TypeScript, or both.
2. Identify the table mode: local/embedded OSS, remote Enterprise/Cloud, or portable across both. If the user says "LanceDB Enterprise", choose the remote table path.
3. Read the matching language branch before writing or changing code:
   - Python patterns: `references/python/patterns.md`
   - Python API quick reference: `references/python/api_reference.md`
   - Python performance guidance: `references/python/performance.md`
   - TypeScript patterns: `references/typescript/patterns.md`
   - TypeScript API quick reference: `references/typescript/api_reference.md`
   - TypeScript performance guidance: `references/typescript/performance.md`
4. Start with `patterns.md` for the selected SDK. Read `api_reference.md` when choosing method names or return collectors. Read `performance.md` when the task involves ingestion, indexing, filtering, query tuning, diagnostics, or large datasets.
5. Prefer `search()` or `query()` builders with explicit `select()` and `limit()` for reads.
6. Avoid table-level full materialization in remote or portable code. This is the main local-vs-remote pitfall.
7. If reviewing an existing file or repo, run `scripts/check_materialization.py` on the relevant paths and inspect each finding before editing.
8. Cross-check unfamiliar or non-trivial API claims against the source tree instead of relying on memory.

## Core Portability Rule

Do not write code that assumes a local table API will exist on a remote table. Remote tables can be very large, so whole-table materialization helpers are intentionally unavailable or unsafe.

This does **not** mean result conversion is forbidden. Bounded query/search result collection is normal:

- Python: `table.search(...).select([...]).limit(10).to_pandas()`
- TypeScript: `await table.search(...).select([...]).limit(10).toArray()`

The unsafe pattern is table-level or unbounded collection, plus local-only dataset escape hatches in remote code:

- Python: `table.to_pandas()`, `table.to_arrow()`, `table.to_polars()`; `table.to_lance()` is local/OSS-only dataset access, not materialization
- TypeScript: `await table.toArrow()`, `await table.query().toArray()` without `limit()`

## Script

Run the scanner when reviewing or modifying an existing codebase:

```bash
python skills/lancedb/scripts/check_materialization.py path/to/file_or_dir
```

The script reports likely unsafe full-table materialization in Python and TypeScript. Treat results as review prompts, not automatic proof of a bug.
