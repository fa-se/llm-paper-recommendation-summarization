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

The docker image provides an execution environment for this library. For usage examples,
see [usage_example.ipynb](usage_example.ipynb) or
[setup/test.py](setup/test.py).

## Setup

### Prerequisites
- PostgreSQL instance
    - with [pgvector](https://github.com/pgvector/pgvector) (tested with 0.7.2 and 0.8.0)
    - with [pg_bestmatch_rs](https://github.com/tensorchord/pg_bestmatch.rs) (for BM25, tested with 0.0.1)
- Docker with Docker Compose
- OpenAI API key

### Preparing the Database
1. Setup the database schema via `setup/ddl.sql`.\
   E.g. `psql -U [DB_USER] -d [DB_NAME] -f setup/ddl.sql`
2. Load OpenAlex embeddings for topic matching via `setup/openalex_embeddings.sql`.\
   E.g. `psql -U [DB_USER] -d [DB_NAME] -f setup/openalex_embeddings.sql`

### Setup Instructions
1. Copy `.env.example` to `.env` and fill in the required values (database connection parameters, OpenAI API key, etc).
2. Run `docker compose build` to obtain an image with the required dependencies.

### Testing the Setup
You can test the setup by running\
`docker compose run --rm app bash -c "python3 setup/test.py 'llm rerankers'"`

This script tests all components of the system, including the database connection, database extensions, OpenAI and
OpenAlex APIs, and the reranking model.

This will use `llm rerankers` as the research interest description (FFTD), perform topic matching and retrieve a small
number (100) of candidate papers from OpenAlex.
These papers will be stored in the database and then ranked w.r.t the FFTD using a hybrid ranking model (embedding +
BM25). After reranking via *[setwise.heapsort](https://arxiv.org/abs/2310.09497v2)*, the top 5 results are printed. The top 3 are then summarized, and the
summaries are printed.
