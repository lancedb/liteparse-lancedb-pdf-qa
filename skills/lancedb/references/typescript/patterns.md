# TypeScript Patterns

Use these patterns when writing TypeScript code with `@lancedb/lancedb`.

## Recommended Patterns

### Bounded query

Use this for application reads, scripts, and examples:

```typescript
const rows = await table
  .query()
  .where("status = 'ready'")
  .select(["id", "text"])
  .limit(20)
  .toArray();
```

### Bounded vector search

```typescript
const rows = await table
  .search(queryVector)
  .select(["id", "text"])
  .limit(20)
  .toArray();
```

### Batch streaming for larger reads

When the task needs many rows, avoid collecting everything at once:

```typescript
for await (const batch of table
  .query()
  .where("status = 'ready'")
  .select(["id", "text"])
  .limit(10_000)) {
  process(batch);
}
```

## Anti-Patterns

### Table-level full materialization

Avoid this in portable or large-table code:

```typescript
const tableArrow = await table.toArrow();
```

Why: `table.toArrow()` is a whole-table operation. It is easy for agents to copy from small local examples and accidentally load a remote production table into memory.

### Unbounded query collection

Avoid:

```typescript
const rows = await table.query().toArray();
const arrow = await table.query().toArrow();
```

Prefer `select(...).limit(...)` before collecting, or use async iteration for large reads.

### Per-row writes

Avoid loops that write one row per call. Batch rows before adding them so LanceDB does not create many small commits/fragments.

### Guessing performance fixes

Avoid changing `nprobes`, `refineFactor`, `ef`, or index settings before checking `analyzePlan()` and `indexStats(...)`.
