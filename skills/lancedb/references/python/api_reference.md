# Python API Reference

Quick method reference for Python LanceDB code. Cross-check source for non-trivial claims.

## Connect

```python
import lancedb

db = lancedb.connect("./data/lancedb")      # local/OSS
db = lancedb.connect("db://my-db", api_key=api_key, region=region)  # remote
```

Async:

```python
db = await lancedb.connect_async("./data/lancedb")
```

## Table Reads

| Task | Preferred API |
| --- | --- |
| Vector search | `table.search(query_vector).limit(k)` |
| Full scan with filters/projection | `table.query().where(...).select(...).limit(...)` |
| Filter | `.where("col > 10")` |
| Projection | `.select(["id", "text"])` |
| Bound result count | `.limit(20)` |
| Collect bounded result as pandas | `.to_pandas()` on query/search result |
| Collect bounded result as Arrow | `.to_arrow()` on query/search result |
| Collect bounded result as Polars | `.to_polars()` on query/search result |

## Local vs Remote Table Methods

| API | Local table | Remote table | Agent guidance |
| --- | --- | --- | --- |
| `table.search(...)` | Yes | Yes | Preferred read path |
| `table.query()` | Yes | Yes | Preferred scan/filter path |
| `table.to_pandas()` | Yes | No / unsafe for portability | Avoid in portable code |
| `table.to_arrow()` | Yes | No / unsafe for portability | Avoid in portable code |
| `table.to_polars()` | Yes | No / unsafe for portability | Avoid in portable code |
| `table.to_lance()` | Yes | No | Local/OSS escape hatch only |

## Indexes

Use `create_index(...)` for vector indexes and modern index configs. Use scalar indexes for filtered or merge keys.

Common calls:

```python
table.create_index("vector")
table.create_scalar_index("status")
table.create_fts_index("text")
```

Check source docs before specifying advanced index config names or parameters.

## Filtering And Recall Knobs

```python
table.search(query_vector).where("status = 'ready'")  # pre-filter by default
table.search(query_vector).where("status = 'ready'", prefilter=False)
table.search(query_vector).limit(10).refine_factor(20)
table.search(query_vector).limit(10).nprobes(50)
```

Use post-filtering only when fewer than `limit` results are acceptable.

## Diagnostics

```python
print(table.search(query_vector).where("year > 2000").limit(10).analyze_plan())
print(table.index_stats("vector_idx"))
```

Use these before changing indexes or search tuning.

## Maintenance

```python
table.optimize()
```

Use this only for local/OSS tables after large writes, many update/delete operations, or on a schedule. It handles compaction, cleanup of old versions according to retention, and index optimization. Do not add this for LanceDB Enterprise/Cloud remote tables; Enterprise handles compaction and cleanup automatically from cluster configuration.
