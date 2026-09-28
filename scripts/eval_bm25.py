"""Which BM25 variant finds the cited papers? The thesis' citation proxy, for keyword search alone.

The move from pg_bestmatch.rs to VectorChord-bm25 (2026-09-28) changed how a query token is weighted: by its count in
the query, where pg_bestmatch counted each distinct token once. Both tokenize with BERT's WordPiece vocabulary, which
splits rare words into pieces; search engines usually use whole words, stemmed, without stopwords. This compares:
  bert_tf     BERT WordPiece, query tokens weighted by count (the pipeline since 2026-09-28)
  bert_set    BERT WordPiece, each query token once (pg_bestmatch's weighting; its rankings agree ~97 %)
  words_tf    words: Unicode segmentation, lowercase, NFKD, NLTK English stopwords, Porter2 stemmer; by count
  words_set   the same words, each query token once

Query papers: the thesis' 100, in the fixed order of the reranking eval (scripts/eval_reranking.py, its OUT gives
queries.json), those with an abstract and >= 20 references. For the ones that eval ran, its OUT/q/ files give the paper
and its 10 topics; for the others, the paper comes from OpenAlex and its topics from the pipeline's topic matching (one
query embedding, ~$0.00002). Per query paper, the corpus is built as that eval did: the 2,000 newest works since 2020
on the 10 topics, filtered like ingest, plus the cited papers that have an abstract, without the query paper itself.
Its abstract is the query; its cited papers are the relevance labels. Per variant: cited papers in the BM25 top 10 and
in the top 100 (the list the hybrid blend takes). No other API calls; ~12 OpenAlex requests per query. Everything
fetched is cached in OUT/q/ and OUT/corpus/, so a rerun costs nothing.

An experiment, not part of the pipeline. It runs in its own scratch database (never the one configured in .env):
    uv run --env-file .env scripts/eval_bm25.py run --eval-dir <eval_reranking OUT> --out <dir> [--n 100] [--db ...]
    uv run --with scipy scripts/eval_bm25.py analyze --out <dir>
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import psycopg
import pyalex
import requests

import core
from core.config import Settings
from core.instrumentation import Trace
from core.openalex import fetch_works_by_topics
from core.services.deduplication import DuplicateFilter, has_usable_abstract
from scripts.eval_reranking import CORPUS_LIMIT, CORPUS_START, MIN_REFERENCES, NUM_TOPICS, openalex_by_ids

VARIANTS = ["bert_tf", "bert_set", "words_tf", "words_set"]
TOP = 100
TEXT_ANALYZER = """
pre_tokenizer = "unicode_segmentation"
[[character_filters]]
to_lowercase = {}
[[character_filters]]
unicode_normalization = "nfkd"
[[token_filters]]
skip_non_alphanumeric = {}
[[token_filters]]
stopwords = "nltk_english"
[[token_filters]]
stemmer = "english_porter2"
"""


def fetch_query_paper(qid: str) -> dict:
    try:
        work = pyalex.Works()[qid]
    except requests.HTTPError as error:
        if error.response is not None and error.response.status_code == 404:  # gone from OpenAlex since 2024
            return {"id": qid, "abstract": None, "referenced_works": [], "missing": True}
        raise
    return {
        "id": qid,
        "title": work["title"],
        "abstract": work["abstract"] if work["abstract_inverted_index"] else None,
        "referenced_works": [ref.rsplit("/", 1)[-1] for ref in work["referenced_works"]],
    }


def query_paper(eval_dir: Path, out: Path, qid: str) -> dict:
    """The query paper and its topics: from the reranking eval if it ran this query, else fetched (cached)."""
    path = out / "q" / f"{qid}.json"
    if path.exists():
        return json.loads(path.read_text())
    source, pool = eval_dir / "q" / f"{qid}.query.json", eval_dir / "q" / f"{qid}.pool.json"
    query = json.loads(source.read_text()) if source.exists() else fetch_query_paper(qid)
    query["eligible"] = bool(query["abstract"]) and len(query["referenced_works"]) >= MIN_REFERENCES
    if pool.exists():
        query["topics"] = [topic["id"] for topic in json.loads(pool.read_text())["topics"]]
    elif query["eligible"]:
        with Trace() as trace:
            query["topics"] = [topic.id for topic in core.retrieval.match_topics(query["abstract"], NUM_TOPICS)]
        query["topics_cost_usd"] = trace.usage["cost_usd"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(query))
    return query


def fetch_corpus(out: Path, query: dict) -> dict:
    """The query paper's corpus, as the reranking eval built it (cached)."""
    path = out / "corpus" / f"{query['id']}.json"
    if path.exists():
        return json.loads(path.read_text())
    works = fetch_works_by_topics(query["topics"], CORPUS_START, limit=CORPUS_LIMIT)
    duplicate_filter = DuplicateFilter()
    corpus = {
        work.id: work.abstract
        for work in works
        if has_usable_abstract(work.abstract) and duplicate_filter.is_new(work.title, work.abstract)
    }
    # only id and abstract (Work.from_openalex would need a publication date, which a few cited works lack)
    cited = {
        int(work["ids"]["openalex"].split("W")[-1]): work["abstract"]
        for work in openalex_by_ids(query["referenced_works"], has_abstract=True)
        if work["abstract_inverted_index"]
    }
    for work_id, abstract in cited.items():
        corpus.setdefault(work_id, abstract)
    corpus.pop(int(query["id"][1:]), None)  # the query paper would find itself
    info = {
        "query": query["abstract"],
        "cited": sorted(work_id for work_id in cited if work_id in corpus),
        "corpus": corpus,
        "fetched": datetime.now().isoformat(timespec="seconds"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info))
    return info


def setup_db(args) -> psycopg.Connection:
    if args.db == os.environ.get("DB_NAME"):
        sys.exit(f"Refusing to run in {args.db}, the database in .env; use a scratch database.")
    params = dict(host=os.environ["DB_HOST"], user=os.environ["DB_USER"], password=os.environ["DB_PASSWORD"])
    with psycopg.connect(dbname="postgres", autocommit=True, **params) as admin:
        if not admin.execute("SELECT 1 FROM pg_database WHERE datname = %s", (args.db,)).fetchone():
            admin.execute(f'CREATE DATABASE "{args.db}"')
    conn = psycopg.connect(
        dbname=args.db, autocommit=True, options="-csearch_path=public,bm25_catalog,tokenizer_catalog", **params
    )
    conn.execute("CREATE EXTENSION IF NOT EXISTS pg_tokenizer CASCADE")
    conn.execute("CREATE EXTENSION IF NOT EXISTS vchord_bm25 CASCADE")
    if not conn.execute("SELECT 1 FROM tokenizer_catalog.tokenizer WHERE name = 'bert'").fetchone():
        conn.execute("""SELECT create_tokenizer('bert', 'model = "bert_base_uncased"')""")
    if not conn.execute("SELECT 1 FROM tokenizer_catalog.text_analyzer WHERE name = 'words'").fetchone():
        conn.execute("SELECT create_text_analyzer('words', %s)", (TEXT_ANALYZER,))
    conn.execute("SET bm25_catalog.bm25_limit = -1")  # exact
    return conn


def rank(conn: psycopg.Connection, qid: str, info: dict) -> dict:
    """Loads the corpus into its own table (BM25 statistics are per corpus) and ranks it with each variant."""
    table, model = f"docs_{qid.lower()}", f"words_{qid.lower()}"
    conn.execute(f"DROP TABLE IF EXISTS {table}")
    if conn.execute("SELECT 1 FROM tokenizer_catalog.tokenizer WHERE name = %s", (model,)).fetchone():
        conn.execute("SELECT drop_tokenizer(%s)", (model,))
        conn.execute("SELECT drop_custom_model(%s)", (model,))
    conn.execute(f"CREATE TABLE {table} (id bigint PRIMARY KEY, abstract text, bert bm25vector, words bm25vector)")
    with conn.cursor().copy(f"COPY {table} (id, abstract) FROM STDIN") as copy:
        for work_id, abstract in info["corpus"].items():
            copy.write_row((int(work_id), abstract))
    conn.execute(f"UPDATE {table} SET bert = tokenize(abstract, 'bert')")
    # a vocabulary of the corpus' stemmed words; the query's other words match nothing anyway
    conn.execute(
        "SELECT create_custom_model_tokenizer_and_trigger(%s, %s, 'words', %s, 'abstract', 'words')",
        (model, model, table),
    )
    for column in ("bert", "words"):
        conn.execute(f"CREATE INDEX {table}_{column} ON {table} USING bm25 ({column} bm25_ops)")
    rankings = {}
    for variant in VARIANTS:
        column, weighting = variant.split("_")
        tokenizer = "bert" if column == "bert" else model
        tokens = "tokenize(%(q)s, %(tok)s)"
        if weighting == "set":
            tokens = f"(SELECT array_agg(DISTINCT t) FROM unnest({tokens}) t)"
        sql = f"""
            WITH query AS MATERIALIZED (SELECT to_bm25query('{table}_{column}', {tokens}) AS q)
            SELECT id FROM {table} ORDER BY {column} <&> (SELECT q FROM query) LIMIT {TOP}"""
        rankings[variant] = [row[0] for row in conn.execute(sql, {"q": info["query"], "tok": tokenizer})]
    conn.execute(f"DROP TABLE {table}")
    conn.execute("SELECT drop_tokenizer(%s)", (model,))
    conn.execute("SELECT drop_custom_model(%s)", (model,))
    return rankings


def run(args):
    out = Path(args.out)
    eval_dir = Path(args.eval_dir)
    pyalex.config.email = Settings.from_env().openalex_contact_email  # the polite pool
    pyalex.config.max_retries = 5
    pyalex.config.retry_backoff_factor = 1.0
    conn = setup_db(args)
    results = {}
    for qid in json.loads((eval_dir / "queries.json").read_text()):
        query = query_paper(eval_dir, out, qid)
        if not query["eligible"]:
            continue
        info = fetch_corpus(out, query)
        rankings = rank(conn, qid, info)
        results[qid] = {"corpus": len(info["corpus"]), "cited": info["cited"], "rankings": rankings}
        print(f"{len(results)}/{args.n} {qid}: {len(info['corpus'])} papers, {len(info['cited'])} cited", flush=True)
        if len(results) >= args.n:
            break
    (out / "rankings.json").write_text(json.dumps(results))
    info = {"commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout}
    (out / "run.json").write_text(json.dumps(info | {"finished": datetime.now().isoformat(timespec="seconds")}))


def analyze(args):
    from scipy.stats import wilcoxon

    results = json.loads((Path(args.out) / "rankings.json").read_text())
    metrics = {}
    for variant in VARIANTS:
        rows = []
        for r in results.values():
            cited = set(r["cited"])
            ranking = r["rankings"][variant]
            rows.append(
                {
                    "top10": len(cited & set(ranking[:10])),
                    "recall100": len(cited & set(ranking)) / len(cited) if cited else 0.0,
                }
            )
        metrics[variant] = rows
    n = len(results)
    random10 = statistics.mean(10 * len(r["cited"]) / r["corpus"] for r in results.values())
    print(
        f"{n} queries; corpus {statistics.mean(r['corpus'] for r in results.values()):.0f} papers and "
        f"{statistics.mean(len(r['cited']) for r in results.values()):.1f} cited papers on average; "
        f"a random 10 would hold {random10:.2f} cited papers"
    )
    print(f"{'variant':10} {'cited in top 10':>16} {'recall@100':>11}   vs bert_tf (Wilcoxon p: top 10, recall@100)")
    for variant in VARIANTS:
        top10 = [row["top10"] for row in metrics[variant]]
        recall = [row["recall100"] for row in metrics[variant]]
        line = f"{variant:10} {statistics.mean(top10):16.2f} {statistics.mean(recall):11.3f}"
        if variant != "bert_tf":
            base10 = [row["top10"] for row in metrics["bert_tf"]]
            base_recall = [row["recall100"] for row in metrics["bert_tf"]]

            def p(a, b):
                return wilcoxon(a, b).pvalue if any(x != y for x, y in zip(a, b, strict=True)) else 1.0

            wins = sum(a > b for a, b in zip(top10, base10, strict=True))
            losses = sum(a < b for a, b in zip(top10, base10, strict=True))
            line += f"   p = {p(top10, base10):.3f}, {p(recall, base_recall):.3f}"
            line += f"  (top 10: {wins} better, {losses} worse)"
        print(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--eval-dir", required=True, help="the OUT directory of scripts/eval_reranking.py")
    run_parser.add_argument("--out", required=True)
    run_parser.add_argument("--n", type=int, default=100, help="query papers (at most the eligible ones)")
    run_parser.add_argument("--db", default="bm25_eval")
    analyze_parser = sub.add_parser("analyze")
    analyze_parser.add_argument("--out", required=True)
    args = parser.parse_args()
    {"run": run, "analyze": analyze}[args.command](args)


if __name__ == "__main__":
    main()
