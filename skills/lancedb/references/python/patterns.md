# Python Patterns

Use these patterns when writing Python code with `lancedb`.

## Before Writing Code

Check the existing environment and project dependencies before choosing an output type. If `pandas` or `polars` is already installed and used by the project, returning query results as a DataFrame is idiomatic. If not, prefer Arrow results or Python object/list output depending on the surrounding code.

Common Python conventions:

- Pandas projects: bounded query/search results as `DataFrame` via `.to_pandas()`.
- Polars projects: bounded query/search results as `DataFrame` via `.to_polars()`.
- Arrow-native projects: bounded query/search results as `pyarrow.Table` via `.to_arrow()`, or iterate Arrow batches when streaming is needed.

## Recommended Patterns

### Bounded search or query

Use this for application reads, examples, notebooks, and agent-generated scripts:

```python
results = (
    table.search(query_vector)
    .where("status = 'ready'")
    .select(["id", "text"])
    .limit(20)
    .to_pandas()
    # or .to_polars()
)
```

Why: `search()` / `query()` works across local and remote tables. `select()` avoids fetching unused columns. `limit()` prevents accidental full-table reads.

### Bounded query result conversion

It is fine to collect bounded query/search results:

```python
arrow_table = table.query().select(["id"]).limit(100).to_arrow()
df = table.search(query_vector).limit(10).to_pandas()
polars_df = table.search(query_vector).limit(10).to_polars()
```

### Local-only Lance dataset API

`table.to_lance()` does not itself materialize the full dataset. It returns the underlying `lance.LanceDataset`, making the table accessible through the PyLance dataset API. Use it when the task is explicitly local/OSS and needs Lance dataset methods not exposed by LanceDB:

```python
# Local/OSS only: RemoteTable does not expose table.to_lance().
ds = table.to_lance()
for batch in ds.to_batches(columns=["id", "text"], batch_size=10_000):
    process(batch)
```

### Async Python

Keep the same shape and bound the result before collecting:

```python
results = await (
    async_table.query()
    .where("status = 'ready'")
    .select(["id", "text"])
    .limit(20)
    .to_pandas()
)
```

## Anti-Patterns

### Table-level full materialization

Avoid these in portable code:

```python
df = table.to_pandas()
arrow_table = table.to_arrow()
polars_df = table.to_polars()
```

Why: local tables expose these convenience methods, but remote tables intentionally do not expose the same full-table materialization surface. Remote tables can be much larger than local development tables.

`table.to_lance()` is different: it is not a full materialization call, but it is still local/OSS-only and should not appear in code meant to run against remote Enterprise tables.

### Unbounded result collection

Avoid query/search collection without a meaningful limit:

```python
df = table.query().to_pandas()
rows = table.search(query_vector).to_list()
```

Prefer `select(...).limit(...)` first.

### Per-row writes

Avoid loops like:

```python
for row in rows:
    table.add([row])
```

Each `add()` creates a new version and fragment. Use bulk inputs or batched iterators instead.

### Guessing performance fixes

Avoid changing `nprobes`, `refine_factor`, or index types before checking the query plan and index stats. Diagnose first, then tune one knob at a time.
