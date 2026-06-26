# LiteParse + LanceDB: a local multimodal PDF evidence store

A small, fully reproducible demo that turns messy, visual ESG reports into a local,
inspectable evidence store and then benchmarks retrieval over it. It is the companion code
for the blog post *Building a Local Multimodal PDF Evidence Store with LiteParse and LanceDB*.

The pipeline runs entirely on your machine:

```text
PDF reports
  -> LiteParse (Python SDK)   text + extracted figures + page screenshots
  -> normalized records       documents / pages / chunks / assets, keyed by page
  -> LanceDB                  text, blobs, vectors, and metadata in one store
  -> retrieval modes          chunks / pages / assets / images / hybrid_bundle
  -> optional answer agent    PydanticAI answer + LLM judge, reading image bytes from LanceDB
```

[LiteParse](https://github.com/run-llama/liteparse) (by LlamaIndex) parses each PDF through its
native Python SDK — a Rust core with no cloud calls, no LLMs, and no API keys. [LanceDB](https://github.com/lancedb/lancedb)
stores the text, the page screenshots and extracted figures (as blobs), the embeddings, and the
provenance in a single multimodal store, with indexes and versioning alongside the data.

## The dataset

A six-report subset of [Climate Finance Bench](https://github.com/Pladifes/climate_finance_bench),
with 50 page-labeled questions:

| Company | Report | Pages | Questions |
|---|---|---:|---:|
| Ali Baba Group | 2024 ESG Report | 200 | 10 |
| Google | 2024 Environmental Report | 86 | 9 |
| NVIDIA | FY2024 Corporate Sustainability Report | 41 | 8 |
| Nestle | 2023 Creating Shared Value & Sustainability | 89 | 8 |
| Samsung | 2024 Sustainability Report | 83 | 7 |
| Total Energies | 2024 Sustainability & Climate Progress | 112 | 8 |

For fast iteration the experiments parse only the benchmark-labeled pages (70 of 611). To run on
full documents, drop the `--pages labeled` filter (use `--pages all`); the pipeline is otherwise
identical.

## What lands in LanceDB

Five tables, all keyed by a stable `page_id` (`{doc_id}:p{page_num}`):

| Table | Rows | Holds |
|---|---:|---|
| `documents` | 6 | report metadata, checksum, parse config, timings |
| `pages` | 70 | full page text, page screenshot (blob), text + CLIP image vectors |
| `chunks` | 252 | page-bounded text spans and their text vectors |
| `assets` | 77 | extracted figures, their bytes (blob), text + CLIP image vectors |
| `eval_questions` | 50 | normalized questions and their expected pages |

Indexes are scalar `BTREE` on the prefilter columns (`company`, `source_pdf`) and an `FTS` index
on `text`. At this scale LanceDB runs an exact vector search; an ANN index is only built once a
table grows past a few hundred rows.

Retrieval exposes five modes: `chunks`, `pages`, and `assets` (text-vector search), `images`
(CLIP text-to-image search over the page and figure image vectors), and `hybrid_bundle` (pools the
three text searches and merges candidates by `page_id`).

## Setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/). The OpenAI embeddings and answer/judge
models need an API key:

```bash
cp .env.example .env          # then add your OPENAI_API_KEY
set -a && source .env && set +a
```

Dependencies (installed automatically by `uv run`) include `liteparse`, `lancedb`, `pylance`,
`open-clip-torch`, `openai`, and `pydantic-ai`. No system packages are needed for these
born-digital PDFs (LiteParse bundles PDFium; OCR is off).

## Reproduce the results

Each step writes its outputs and metrics to disk. Generated data under `data/` is gitignored.

**1. Download the dataset** (reproducible; verified by SHA256 in `manifest.json`):

```bash
uv run python scripts/download_climate_finance_bench.py \
  --companies Samsung NVIDIA Google "Ali Baba Group" Nestle "Total Energies S.A" \
  --out data/raw/climate_finance_bench
```

**2. Parse with LiteParse** → normalized `documents`/`pages`/`chunks`/`assets` records, page
screenshots, and extracted figures:

```bash
uv run python scripts/parse_reports.py \
  --pages labeled \
  --out data/parsed/liteparse \
  --metrics results/extraction_performance.json
```

**3. Build the LanceDB store** with OpenAI text embeddings + OpenCLIP image embeddings:

```bash
uv run python scripts/build_lancedb.py \
  --parsed-dir data/parsed/liteparse \
  --db data/lancedb/esg_pdf_qa_openai_small_clip.lancedb \
  --overwrite \
  --embedding-backend openai --text-model text-embedding-3-small \
  --image-embedding-backend open-clip \
  --metrics results/storage_metrics_openai_small_clip.json
```

**4. Run the retrieval benchmark** across all five modes:

```bash
uv run python scripts/run_retrieval_eval.py \
  --db data/lancedb/esg_pdf_qa_openai_small_clip.lancedb \
  --embedding-backend openai --text-model text-embedding-3-small \
  --image-embedding-backend open-clip \
  --top-k 5 \
  --out data/eval/retrieval_results_openai_small_clip.jsonl \
  --metrics results/metrics_openai_small_clip.json
```

**5. (Optional) Answer-correctness eval** — a capstone that retrieves the `hybrid_bundle`, reads
the page screenshots straight from LanceDB blobs, sends them to an answer agent, and grades the
output with an LLM judge:

```bash
uv run python scripts/run_answer_eval.py \
  --db data/lancedb/esg_pdf_qa_openai_small_clip.lancedb \
  --answer-model gpt-5.4 --judge-model gpt-5.4-mini \
  --embedding-backend openai --text-model text-embedding-3-small \
  --out data/eval/answer_eval_results.jsonl \
  --metrics results/answer_eval_metrics.json
```

For an offline smoke test (no API key), omit the OpenAI flags: the build and retrieval scripts fall
back to a deterministic local hash embedder. Use `--limit N` on the eval scripts to run a subset.

## Results

All numbers below are from the commands above on the 70 labeled pages.

**Extraction** (LiteParse, labeled-page run):

| Stage | Time | Throughput |
|---|---:|---:|
| `parse()` | 0.51 s | ~136 pages/s |
| `screenshot()` | 1.77 s | ~40 pages/s |
| End-to-end | 2.35 s | ~30 pages/s |

**Storage** (OpenAI `text-embedding-3-small` + OpenCLIP `ViT-B-32`): write 0.20 s, index 0.10 s,
total **114 MB**. Almost all of it is page screenshots in `pages` (101 MB); extracted figures in
`assets` add 10 MB, and everything else is a couple of MB. For comparison, the six source PDFs are
64 MB — the store is larger because it keeps full-resolution page renders on hand.

**Retrieval** (top-5, page-label hits):

| Mode | Any@5 | Cov@5 | All@5 | Modality@5 | P50 |
|---|---:|---:|---:|---:|---:|
| `hybrid_bundle` | 0.82 | 0.733 | 0.66 | 0.68 | 4.7 ms |
| `pages` | 0.76 | 0.672 | 0.60 | 0.94 | 1.7 ms |
| `images` | 0.76 | 0.609 | 0.48 | 0.90 | 17.9 ms |
| `chunks` | 0.72 | 0.588 | 0.48 | 0.58 | 1.7 ms |
| `assets` | 0.38 | 0.277 | 0.20 | 0.66 | 1.7 ms |

The `hybrid_bundle` leads on page recall because the page-keyed layout lets it pool chunk, page,
and figure hits and let each page win on its best signal. The single visual modes (`pages`,
`images`) lead on `modality_hit`, since they reliably return page-level or image evidence for the
table and figure questions.

## Repository layout

```text
src/            parse, schema, index (build), retrieval, embeddings, eval, answer_eval
scripts/        download / inspect / parse / build / run_retrieval_eval / run_answer_eval
data/           raw PDFs, parsed records, LanceDB store, eval sets (gitignored)
results/        extraction, storage, and retrieval metrics (JSON)
```
