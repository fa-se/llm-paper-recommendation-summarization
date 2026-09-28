# Paper Recommendation & Summarization

---

Developed during my master's thesis at TU Berlin, this library provides an end-to-end RAG pipeline for paper
recommendation and summarization.
Its purpose is to assist researchers in staying up-to-date with the latest research in their field. As such, it is
focused on the discovery of new research, using [OpenAlex](https://openalex.org/) as a data source.

From a free-form text description (FFTD) of a research interest, the system retrieves a set of candidate papers from
OpenAlex, ranks them with regard to the query, and generates summaries tailored to the user's interest.

## Architecture

### Overview

![Architecture](media/system_architecture.svg)

### Retrieval

![Retrieval](media/retrieval.svg)

### Summarization

![Summarization](media/summarization.svg)

## Usage

```python
from datetime import datetime

from core import retrieval, summarization

query = "My research investigates the impact of climate change on coral reef degradation ..."
start_date = datetime(2025, 1, 1)

topics, added = retrieval.ingest(query, start_date, limit=2000, num_topics=10)  # build the corpus
works = retrieval.search(query, n=5, start_date=start_date)  # hybrid search + setwise LLM reranking
summaries = summarization.summarize(query, works[:3])  # summaries tailored to the query
```

The docker image provides an execution environment for this library. For more examples,
see [usage_example.ipynb](usage_example.ipynb), [scripts/smoke_test.py](scripts/smoke_test.py) and
[scripts/trace_run.py](scripts/trace_run.py), which records every pipeline stage (timings, tokens, cost, intermediate
results) as a replayable event stream (`core/instrumentation.py`).

### Code structure

```
core/
  __init__.py                   retrieval and summarization, created on first use (create_services)
  config.py                     settings from environment variables: database, API keys, models
  services/retrieval_service.py ingest (topics -> fetch -> filter -> embed -> index), search (semantic + BM25 ->
                                hybrid -> rerank)
  services/setwise_reranker.py  setwise heapsort LLM reranking, with concurrent comparisons
  services/summarization_service.py  tailored summaries
  services/deduplication.py     skips duplicate and junk works at ingest
  llm_interfaces/               OpenAI embeddings and chat completions (tokens and cost per call), prompts (tasks.py)
  repositories/                 SQL: vector and BM25 search, storage
  openalex.py                   OpenAlex API access
  works.py, sqlalchemy_models.py  the pipeline's work objects, the database tables
  instrumentation.py            Trace: per-stage timings, tokens, cost and events
demo/                           web app that runs the pipeline live and visualizes each stage (see "Demo app")
scripts/                        smoke test, traced demo runs, pre-ingest of the demo queries, schema export, topic map
setup/                          database image, schema, OpenAlex topic embeddings
notebooks/                      thesis experiments and evaluation, written against earlier versions of core (thesis-era
                                code: commit 796e20c)
```

### Demo app

`demo/` is a small web app (FastAPI, server-sent events, plain JavaScript) that runs the pipeline for a query and shows
each stage while it runs, from the trace events of `core/instrumentation.py`:

- **Topic routing:** all 4,516 OpenAlex topics on a 2D map (UMAP of their embeddings), with the query's 10 most similar
  topics.
- **Hybrid retrieval:** BM25, dense and hybrid rankings side by side, with the LLM's final top 5 traced through them. The
  blend can be changed in the browser (another weight, or reciprocal rank fusion) to see which candidates it would keep.
- **LLM reranking:** the setwise heapsort as it happens, with the comparisons in flight, and the latest decision with
  its prompt.
- **Tailored summaries:** abstract, the filled-out reasoning structure, and the summary.
- Time, tokens and cost per stage, and a timeline of all API calls.

```sh
uv run --env-file .env python -m demo        # http://localhost:8000
```

Every successful live run is saved to `demo/recordings/`; a replay plays such a recording in the browser with its
original timing, without database or API key (`coral_reefs.json` and `rag_hallucinations.json` are committed). Keys
1-4 switch between the stages; with "stage by stage", a replay pauses before each stage (space continues).

A live run with "fetch new papers first" adds papers to the corpus (and so changes the BM25 corpus statistics, and
every paper's score). "Reset corpus" in the page, or `uv run --env-file .env python -m demo.corpus reset`, deletes them
again and rebuilds the BM25 index, which restores the baseline in `demo/corpus_baseline.json` exactly (checked by a
checksum over all papers' BM25 token vectors); `python -m demo.corpus snapshot` makes the current corpus the baseline.

## Setup

### Prerequisites
- PostgreSQL 18 (tested with 18.6)
    - with [pgvector](https://github.com/pgvector/pgvector) (tested with 0.8.6)
    - with [VectorChord-bm25](https://github.com/tensorchord/VectorChord-bm25) (BM25, tested with 0.3.0) and
      [pg_tokenizer](https://github.com/tensorchord/pg_tokenizer.rs) (tested with 0.1.1), both in
      `shared_preload_libraries`
    - `setup/Dockerfile` builds such an image (amd64 and arm64, from release packages); `docker compose up -d db`
      runs it
- Docker with Docker Compose, or [uv](https://docs.astral.sh/uv/) to run it locally
- OpenAI API key

### Preparing the Database
1. Setup the database schema via `setup/ddl.sql`.\
   E.g. `psql -U [DB_USER] -d [DB_NAME] -f setup/ddl.sql`
2. Load OpenAlex embeddings for topic matching via `setup/openalex_embeddings.sql`.\
   E.g. `psql -U [DB_USER] -d [DB_NAME] -f setup/openalex_embeddings.sql`
3. Optional: restart the database, so that it loads the BM25 tokenizer at startup (`setup/ddl.sql` sets this up)
   instead of in each new connection.

`pg_dump` leaves out the BM25 tokenizer (it lives in pg_tokenizer's extension tables). To restore a dump, create the
schema with `setup/ddl.sql` first, then restore the data only (`pg_restore --data-only`).

### BM25 since September 2026: VectorChord-bm25 instead of pg_bestmatch.rs
The thesis computed BM25 with [pg_bestmatch.rs](https://github.com/tensorchord/pg_bestmatch.rs) on PostgreSQL 16.
pg_bestmatch.rs is no longer maintained (no commits since November 2024, PostgreSQL 17 at most), so BM25 now comes from
its successor by the same team, VectorChord-bm25, on PostgreSQL 18. **BM25 rankings differ from the thesis's**, so its
comparison of BM25, dense and hybrid retrieval (Tables 6-9) doesn't reproduce exactly:
- The same: the tokenizer (BERT's lowercased WordPiece vocabulary, `bert_base_uncased`), the IDF
  `ln((N + 1) / (df + 0.5))`, k1 = 1.2 and b = 0.75.
- Different: a token that occurs several times in the query counts that many times (pg_bestmatch counted each distinct
  query token once); document lengths are stored quantized, as in Lucene; and scores are k1 + 1 times larger, which
  doesn't change the ranking.
- Measured on the demo corpus (3,509 papers), against pg_bestmatch: for 30 abstracts as queries, the BM25 top 10 shares
  61 % of its papers on average, the top 100 71 %. For the two demo descriptions, 5 and 8 of the top 10. Counting each
  query token once instead gives 97 % and 98 %, so nearly all of the difference is the query weighting. The hybrid
  top 50 that goes to the LLM hardly changes, as BM25 weighs 0.2 in the blend: 49 of 50 papers stay, for both demo
  descriptions.
- For pg_bestmatch's query weighting: in `PublicationRepository.search_by_bm25`, count each query token once, i.e.
  replace `tokenize(...)` by `(SELECT array_agg(DISTINCT t) FROM unnest(tokenize(...)) t)`.

New papers only need tokenizing: the index adds them to its corpus statistics itself. Before, each ingest recomputed
the BM25 vector of every paper.

### Upgrading a database from before September 2026 (PostgreSQL 16 + pg_bestmatch.rs)
The compose file now uses a new volume (`postgres18_data`); the old one (`postgres_data`) stays as it is. A plain dump
doesn't restore into the new schema (the `bm25` column changed type): copy the four `openalex_*` tables with
`pg_dump --data-only`, and the publications without their `bm25` column (`\copy (SELECT <all other columns> FROM
publication) TO STDOUT`, then `\copy publication(<the same columns>) FROM STDIN`), into a database created from
`setup/ddl.sql`; then `PublicationRepository.rebuild_bm25()` tokenizes the abstracts.

### Setup Instructions
1. Copy `.env.example` to `.env` and fill in the required values (database connection parameters, OpenAI API key, etc).
2. Create the shared Docker network via `docker network create postgres_network`. The compose file uses it as an
   external network, so that other containers can reach the database as well.
3. Run `docker compose build` to obtain an image with the required dependencies.

Without Docker: `uv sync` creates `.venv` with the locked dependencies (`uv.lock`) and installs this project in editable
mode. `uv run --env-file .env <script>` then runs a script with the environment from `.env`, e.g.
`uv run --env-file .env scripts/trace_run.py rag_hallucinations --summaries 3`, which prints time, tokens and cost per
pipeline stage.

### Testing the Setup
You can test the setup by running\
`docker compose run --rm app python scripts/smoke_test.py 'llm rerankers'`\
or, without Docker, `uv run --env-file .env scripts/smoke_test.py 'llm rerankers'`

This script tests all components of the system, including the database connection, database extensions, OpenAI and
OpenAlex APIs, and the reranking model.

This will use `llm rerankers` as the research interest description (FFTD), perform topic matching and retrieve a small
number (100) of candidate papers from OpenAlex.
These papers will be stored in the database and then ranked w.r.t the FFTD using a hybrid ranking model (embedding +
BM25). After reranking via *[setwise.heapsort](https://arxiv.org/abs/2310.09497v2)*, the top 5 results are printed. The top 3 are then summarized, and the
summaries are printed.

The unit tests need neither API keys nor a database: `uv run python -m unittest discover -s tests -t .`\
Lint and format: `uv run ruff check` and `uv run ruff format`.
