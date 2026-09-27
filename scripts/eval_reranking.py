"""How much does the LLM reranker add over hybrid search? The thesis' citation-proxy eval, re-run with the order kept.

The thesis (ch. 6.3.2) reported +60 % RRR@10 for reranking over the hybrid ranking. Re-checked in 2026, its eval
notebook had re-fetched the hybrid top 100 from OpenAlex without restoring their order, so the "before" list was in the
API's order, not the hybrid order (docs: thesis_project.md §4 in the prep notes). This script repeats the protocol with
the current pipeline and keeps the order.

Per query paper (the thesis' 100 query papers, from its 11 Aug 2024 run log in git, in a fixed random order):
  top-10 OpenAlex topics of its abstract -> the 2,000 newest works since 2020 on them (RetrievalService.ingest)
  + its cited papers that have an abstract -> hybrid top 110, order and scores kept (no 0.95 filter)
  -> pool = the top 100, the query paper itself excluded -> setwise heapsort to the top 10.
The query paper's references are the relevance labels. `analyze` compares hybrid's top 10, the LLM's top 10 and a
random 10 of the pool: cited papers in the top 10, RRR@10, MRCR@10 (reference overlap), paired Wilcoxon tests, and the
effect of the thesis' "drop hybrid scores >= 0.95" filter.

An experiment, not part of the pipeline. It runs in a separate scratch database (never the one configured in .env,
which holds the demo corpus), made once with:
    docker exec project_name-db-1 psql -U <DB_USER> -d postgres -c "CREATE DATABASE rerank_eval"
    docker exec -i project_name-db-1 psql -U <DB_USER> -d rerank_eval < setup/ddl.sql
    docker exec project_name-db-1 sh -c "pg_dump -U <DB_USER> -d <DB_NAME> --data-only -t openalex_domain \\
        -t openalex_field -t openalex_subfield -t openalex_topic | psql -U <DB_USER> -d rerank_eval"

Resumable: each stage writes its result to OUT/q/, and a restart skips finished stages. The corpus of the query in
progress stays in the scratch DB (OUT/current.json names it), so a restart re-embeds only what's missing. Every paid
API call is appended to OUT/ledger.jsonl as it happens, and the budget check before each stage uses the ledger's
total over all runs: a stage overshoots by at most its own cost (~$0.07 retrieval, ~$0.02 rerank). Exit code 3 means
the budget is reached; the same command with a higher --budget continues.

    uv run --env-file .env scripts/eval_reranking.py run --out <dir> --budget 4 [--n 30] [--model gpt-6-luna]
    uv run --with numpy --with scipy scripts/eval_reranking.py analyze --out <dir> [--model gpt-6-luna ...]
"""

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from datetime import datetime
from itertools import chain
from pathlib import Path

import pyalex
from sqlalchemy import text

from core import create_services
from core.config import Settings
from core.instrumentation import Trace
from core.llm_interfaces.openai import OpenAIInterface
from core.services.setwise_reranker import SetwiseHeapsortReranker
from core.works import Work

REPO = Path(__file__).resolve().parent.parent
# the notebook of the thesis' 11 Aug 2024 eval run; its output logs the 100 query papers
THESIS_RUN = ("36e8938", "notebooks/retrieval+reranking_eval.ipynb")

CORPUS_START = datetime(2020, 1, 1)  # thesis: works since 2020, newest first
SEARCH_START = datetime(1900, 1, 1)  # search everything, so that the (older) cited papers are found
CORPUS_LIMIT, NUM_TOPICS, HYBRID_N, POOL_N, K = 2000, 10, 110, 100, 10
MIN_REFERENCES = 20  # thesis: at least 20 references
ESTIMATE = {"retrieval": 0.12, "rerank": 0.04}  # USD per stage, conservative; refined from finished stages


class Paths:
    def __init__(self, out: str):
        self.out = Path(out)
        self.q = self.out / "q"
        self.ledger = self.out / "ledger.jsonl"
        self.current = self.out / "current.json"
        self.queries = self.out / "queries.json"


# ---------------------------------------------------------------- run


def settings(db: str, rerank_model: str | None = None) -> Settings:
    env = {**os.environ, "DB_NAME": db}
    if rerank_model:
        env["OPENAI_RERANK_MODEL"] = rerank_model
        # GPT-6 takes temperature 0 only with effort "none"; older models get no effort parameter
        env["OPENAI_RERANK_REASONING_EFFORT"] = "none" if rerank_model.startswith("gpt-6") else ""
    return Settings.from_env(env)


def spent(paths: Paths) -> float:
    if not paths.ledger.exists():
        return 0.0
    return sum(json.loads(line)["cost_usd"] for line in paths.ledger.read_text().splitlines() if line.strip())


def ledger_listener(paths: Paths, query: str, stage: str):
    def on_event(event: dict):
        if event["type"] == "llm_call":
            record = {"time": datetime.now().isoformat(timespec="seconds"), "query": query, "stage": stage}
            record |= {key: event[key] for key in ("model", "input_tokens", "output_tokens", "cost_usd")}
            with paths.ledger.open("a") as f:
                f.write(json.dumps(record) + "\n")

    return on_event


def check_budget(paths: Paths, stage: str, args: argparse.Namespace):
    pattern = "*.pool.json" if stage == "retrieval" else "*.rerank.*.json"
    costs = [json.loads(path.read_text())["cost_usd"] for path in paths.q.glob(pattern)]
    estimate = max(ESTIMATE[stage], 1.2 * max(costs)) if len(costs) >= 3 else ESTIMATE[stage]
    total = spent(paths)
    if total + estimate > args.budget:
        print(
            f"\nBUDGET STOP before {stage}: spent ${total:.3f}, next stage ~${estimate:.2f}, budget ${args.budget:.2f}."
        )
        print(f"Resume: the same command with a higher --budget, e.g. --budget {args.budget + 1:.0f}")
        sys.exit(3)


def thesis_query_ids() -> list[str]:
    commit, path = THESIS_RUN
    notebook = subprocess.run(
        ["git", "-C", str(REPO), "show", f"{commit}:{path}"], capture_output=True, text=True, check=True
    ).stdout
    log = "".join("".join(output.get("text", "")) for output in json.loads(notebook)["cells"][7]["outputs"])
    ids = re.findall(r"Run: \d+ --- Title: .*?, id: https://openalex.org/(W\d+)", log)
    assert len(ids) == 100, f"expected 100 query papers in the run log, found {len(ids)}"
    return ids


def selection(paths: Paths) -> list[str]:
    if paths.queries.exists():
        return json.loads(paths.queries.read_text())
    ids = sorted(thesis_query_ids())
    random.Random(0).shuffle(ids)
    paths.queries.write_text(json.dumps(ids))
    return ids


def query_paper(paths: Paths, wid: str) -> dict:
    path = paths.q / f"{wid}.query.json"
    if path.exists():
        return json.loads(path.read_text())
    work = pyalex.Works()[wid]
    info = {
        "id": wid,
        "title": work["title"],
        "publication_date": work["publication_date"],
        "abstract": work["abstract"] if work["abstract_inverted_index"] else None,
        "referenced_works": [ref.rsplit("/", 1)[-1] for ref in work["referenced_works"]],
    }
    info["eligible"] = bool(info["abstract"]) and len(info["referenced_works"]) >= MIN_REFERENCES
    path.write_text(json.dumps(info))
    return info


def openalex_by_ids(ids: list[str], **filters) -> list[pyalex.Work]:
    """The works with these IDs, in OpenAlex's order (not the given one; look them up by ID)."""
    works = []
    for start in range(0, len(ids), 100):
        query = pyalex.Works().filter(openalex="|".join(ids[start : start + 100]), **filters)
        works.extend(chain(*query.paginate(per_page=100, n_max=None)))
    return works


def retrieval_stage(paths: Paths, q: dict, services, args: argparse.Namespace):
    path = paths.q / f"{q['id']}.pool.json"
    if path.exists():
        return
    check_budget(paths, "retrieval", args)
    retrieval = services.retrieval
    repo = retrieval.publication_repository
    session = repo.session
    assert session.execute(text("SELECT current_database()")).scalar() == args.db

    current = json.loads(paths.current.read_text())["query"] if paths.current.exists() else None
    if current != q["id"]:  # a new query: empty the scratch corpus; the same query (resume): keep what's embedded
        session.execute(text("DELETE FROM publication"))
        session.commit()
        paths.current.write_text(json.dumps({"query": q["id"]}))

    started = time.perf_counter()
    with Trace(on_event=ledger_listener(paths, q["id"], "retrieval")) as trace:
        topics, added = retrieval.ingest(q["abstract"], CORPUS_START, limit=CORPUS_LIMIT, num_topics=NUM_TOPICS)

        # the cited papers that have an abstract (as in the thesis), unless already in the corpus
        known = repo.get_all_openalex_ids()
        cited = [Work.from_openalex(work) for work in openalex_by_ids(q["referenced_works"], has_abstract=True)]
        cited = [work for work in cited if work.abstract]
        new_cited = list({work.id: work for work in cited if work.id not in known}.values())
        if new_cited:
            embeddings = retrieval.llm.create_embedding_batch([work.abstract for work in new_cited])
            for work, embedding in zip(new_cited, embeddings, strict=True):
                repo.add(work, embedding, datetime.now())
            repo.commit()
        repo.rebuild_bm25()

        corpus_ids = repo.get_all_openalex_ids()
        cited_ids = {int(ref[1:]) for ref in q["referenced_works"]}
        ranking = retrieval._hybrid_search(q["abstract"], HYBRID_N, SEARCH_START)
        hybrid_event = next(e for e in reversed(trace.events) if e["type"] == "ranking" and e["method"] == "hybrid")
        parts = {result["id"]: result for result in hybrid_event["results"]}

    query_id = int(q["id"][1:])
    hybrid_ids = [work_id for work_id in ranking if work_id != query_id]
    # reference lists for the overlap metric, looked up by ID: the 2024 eval lost the hybrid order right here
    refs = {
        work["id"].rsplit("/", 1)[-1]: [ref.rsplit("/", 1)[-1] for ref in work["referenced_works"]]
        for work in openalex_by_ids([f"W{work_id}" for work_id in hybrid_ids])
    }
    pool_ids = hybrid_ids[:POOL_N]
    stored = {p.openalex_id: p for p in repo.get_by_openalex_ids(pool_ids)}
    result = {
        "query": q["id"],
        "topics": [{"id": topic.id, "name": topic.name} for topic in topics],
        "corpus_size": len(corpus_ids),
        "topic_works_added_this_run": len(added),
        "cited_total": len(q["referenced_works"]),
        "cited_with_abstract": len(cited),
        "cited_in_corpus": sorted(cited_ids & corpus_ids),
        "query_paper_in_corpus": query_id in corpus_ids,
        "hybrid": [
            {
                "id": work_id,
                "score": ranking[work_id],
                "semantic": parts[work_id]["semantic"],
                "bm25": parts[work_id]["bm25"],
                "is_cited": work_id in cited_ids,
                "referenced_works": refs.get(f"W{work_id}"),
            }
            for work_id in hybrid_ids
        ],
        "pool": [{"id": i, "title": stored[i].title, "abstract": stored[i].abstract} for i in pool_ids],
        "cost_usd": trace.usage["cost_usd"],
        "duration_s": round(time.perf_counter() - started, 1),
    }
    path.write_text(json.dumps(result))


def rerank_stage(paths: Paths, q: dict, args: argparse.Namespace):
    path = paths.q / f"{q['id']}.rerank.{args.model}.json"
    if path.exists():
        return
    check_budget(paths, "rerank", args)
    pool = json.loads((paths.q / f"{q['id']}.pool.json").read_text())
    works = [
        Work(id=p["id"], title=p["title"], authors=[], abstract=p["abstract"], publication_date=None)
        for p in pool["pool"]
    ]
    llm = OpenAIInterface(settings(args.db, args.model))
    started = time.perf_counter()
    with Trace(on_event=ledger_listener(paths, q["id"], f"rerank:{args.model}")) as trace:
        # lookahead 0: exactly the sequential heapsort's comparisons, no speculative calls (fewer tokens)
        order = SetwiseHeapsortReranker(llm, lookahead=0).rerank(q["abstract"], works, k=K)
    usage = trace.usage
    path.write_text(
        json.dumps(
            {
                "query": q["id"],
                "model": args.model,
                "k": K,
                "lookahead": 0,
                "order": [work.id for work in order],
                **{key: usage[key] for key in ("llm_calls", "input_tokens", "output_tokens", "cost_usd")},
                "duration_s": round(time.perf_counter() - started, 1),
            }
        )
    )


def run(args: argparse.Namespace):
    if args.db == os.environ.get("DB_NAME"):
        sys.exit(f"--db {args.db} is the database configured in .env (the demo corpus); use a scratch database")
    paths = Paths(args.out)
    paths.q.mkdir(parents=True, exist_ok=True)
    services = create_services(settings(args.db))  # its llm embeds; rerank_stage builds one per reranker model
    pyalex.config.max_retries = 5  # transient OpenAlex errors; anything else crashes, and the next run resumes
    pyalex.config.retry_backoff_factor = 1.0
    done = 0
    for wid in selection(paths):
        q = query_paper(paths, wid)
        if not q["eligible"]:
            print(f"{wid}: skipped (no abstract or < {MIN_REFERENCES} references)")
            continue
        retrieval_stage(paths, q, services, args)
        rerank_stage(paths, q, args)
        done += 1
        pool = json.loads((paths.q / f"{wid}.pool.json").read_text())
        cited_in_pool = sum(h["is_cited"] for h in pool["hybrid"][:POOL_N])
        print(
            f"[{done}/{args.n}] {wid}: corpus {pool['corpus_size']}, cited in corpus {len(pool['cited_in_corpus'])}, "
            f"cited in pool {cited_in_pool}; spent ${spent(paths):.3f}",
            flush=True,
        )
        if done >= args.n:
            break
    print(f"Done: {done} queries, total spent ${spent(paths):.3f}")


# ---------------------------------------------------------------- analyze


def analyze(args: argparse.Namespace):
    import numpy as np
    from scipy import stats

    paths = Paths(args.out)

    def ncro(query_refs: set[str], refs: list[str] | None) -> float:
        """Normalized common reference overlap (thesis eq. 7); undefined without references."""
        if refs is None:
            return np.nan
        denominator = min(len(query_refs), len(set(refs)))
        return len(query_refs & set(refs)) / denominator if denominator else np.nan

    def cited_flags(ids: list[int], cited: set[int]):
        return np.array([i in cited for i in ids], dtype=float)

    def ncros(ids: list[int], query_refs: set[str], by_id: dict[int, dict]):
        return np.array([ncro(query_refs, by_id[i]["referenced_works"]) for i in ids])

    def per_query(model: str, min_corpus: int) -> list[dict]:
        rows = []
        for pool_path in sorted(paths.q.glob("*.pool.json")):
            wid = pool_path.name.split(".")[0]
            rerank_path = paths.q / f"{wid}.rerank.{model}.json"
            if not rerank_path.exists():
                continue
            pool, rerank = json.loads(pool_path.read_text()), json.loads(rerank_path.read_text())
            if pool["corpus_size"] < min_corpus:
                continue
            query_refs = set(json.loads((paths.q / f"{wid}.query.json").read_text())["referenced_works"])
            by_id = {h["id"]: h for h in pool["hybrid"]}
            cited = set(pool["cited_in_corpus"])
            hybrid_ids = [h["id"] for h in pool["hybrid"]]
            filtered_ids = [h["id"] for h in pool["hybrid"] if h["score"] < 0.95]  # the 2024 eval's filter
            pool_ids = hybrid_ids[:POOL_N]
            llm_ids = rerank["order"][:K]
            assert set(rerank["order"]) == set(pool_ids)
            pool_cited = cited_flags(pool_ids, cited).sum()
            rows.append(
                {
                    "cited_in_corpus": len(cited),
                    "pool_cited": pool_cited,
                    "hybrid_cited_rank": cited_flags(hybrid_ids[:K], cited),
                    "llm_cited_rank": cited_flags(llm_ids, cited),
                    "hybrid_ncro_rank": ncros(hybrid_ids[:K], query_refs, by_id),
                    "llm_ncro_rank": ncros(llm_ids, query_refs, by_id),
                    "random_cited": pool_cited * K / len(pool_ids),
                    "oracle_cited": min(K, pool_cited),
                    "pool_ncro": np.nanmean(ncros(pool_ids, query_refs, by_id)),
                    "filtered_hybrid_cited": cited_flags(filtered_ids[:K], cited).sum(),
                    "filtered_hybrid_mrcr": np.nanmean(ncros(filtered_ids[:K], query_refs, by_id)),
                    "filtered_pool_cited": cited_flags(filtered_ids[:POOL_N], cited).sum(),
                    "n_ge_095": sum(h["score"] >= 0.95 for h in pool["hybrid"]),
                    "n_ge_095_cited": sum(h["score"] >= 0.95 and h["is_cited"] for h in pool["hybrid"]),
                    "cost_usd": pool["cost_usd"] + rerank["cost_usd"],
                }
            )
        return rows

    def paired(label: str, a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        d = b - a
        d = d[~np.isnan(d)]
        p = stats.wilcoxon(d).pvalue if np.any(d != 0) else float("nan")
        print(
            f"  {label:44s} {np.nanmean(a):6.3f} -> {np.nanmean(b):6.3f}  (Δ {np.mean(d):+.3f}; better/same/worse "
            f"{int((d > 0).sum())}/{int((d == 0).sum())}/{int((d < 0).sum())}; Wilcoxon p = {p:.2g})"
        )

    def report(model: str, min_corpus: int = 0):
        rows = per_query(model, min_corpus)
        if not rows:
            print(f"{model}: no results")
            return

        def col(key):
            return np.array([row[key] for row in rows], dtype=float)

        hybrid_cited = np.array([row["hybrid_cited_rank"].sum() for row in rows])
        llm_cited = np.array([row["llm_cited_rank"].sum() for row in rows])
        mrcr_hybrid = np.array([np.nanmean(row["hybrid_ncro_rank"]) for row in rows])
        mrcr_llm = np.array([np.nanmean(row["llm_ncro_rank"]) for row in rows])
        subset = f", corpus >= {min_corpus}" if min_corpus else ""
        print(f"\n=== reranker {model}{subset}: {len(rows)} queries, cost ${col('cost_usd').sum():.2f}")
        print(
            f"  cited papers in corpus {col('cited_in_corpus').mean():.1f}; in hybrid top 100 "
            f"{col('pool_cited').mean():.1f} ({(col('pool_cited') / col('cited_in_corpus')).mean():.0%})"
        )
        print(
            f"  cited in top 10: random 10 of pool {col('random_cited').mean():.2f} | hybrid {hybrid_cited.mean():.2f} "
            f"| LLM {llm_cited.mean():.2f} | oracle {col('oracle_cited').mean():.2f}"
        )
        at_oracle = int((hybrid_cited == col("oracle_cited")).sum())
        print(f"  queries where hybrid's top 10 is already the oracle: {at_oracle}/{len(rows)}")
        paired("cited@10: hybrid -> LLM", hybrid_cited, llm_cited)
        rrr = col("cited_in_corpus")
        paired("RRR@10: hybrid -> LLM", hybrid_cited / rrr, llm_cited / rrr)
        paired("MRCR@10: hybrid -> LLM", mrcr_hybrid, mrcr_llm)
        paired("cited@10: random-10 -> hybrid", col("random_cited"), hybrid_cited)
        paired("cited@10: random-10 -> LLM", col("random_cited"), llm_cited)
        paired("MRCR@10: pool average -> LLM", col("pool_ncro"), mrcr_llm)
        for name, key in (("hybrid", "hybrid"), ("LLM", "llm")):
            cited_rank = np.nanmean(np.vstack([row[f"{key}_cited_rank"] for row in rows]), axis=0)
            ncro_rank = np.nanmean(np.vstack([row[f"{key}_ncro_rank"] for row in rows]), axis=0)
            print(f"  {name:6s} P(cited) by rank 1..10: {' '.join(f'{x:.2f}' for x in cited_rank)}")
            print(f"  {name:6s} RMRCR     by rank 1..10: {' '.join(f'{x:.3f}' for x in ncro_rank)}")
        print("  0.95 filter (hybrid, no LLM):")
        hits, cited_hits = col("n_ge_095"), col("n_ge_095_cited")
        print(
            f"    hits >= 0.95 per query: mean {hits.mean():.2f}, >= 1 in {(hits >= 1).mean():.0%} of queries; "
            f"cited among them: {int(cited_hits.sum())} of {int(hits.sum())}"
        )
        paired("cited@10 hybrid: without -> with filter", hybrid_cited, col("filtered_hybrid_cited"))
        paired("MRCR@10 hybrid: without -> with filter", mrcr_hybrid, col("filtered_hybrid_mrcr"))
        paired("cited in pool: without -> with filter", col("pool_cited"), col("filtered_pool_cited"))

    for model in args.model:
        report(model)
        report(model, min_corpus=1000)  # sensitivity: without queries whose newest works were mostly junk stubs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", help="run the eval (paid API calls, resumable)")
    run_parser.add_argument("--out", required=True, help="output directory (per-query results, ledger)")
    run_parser.add_argument("--budget", type=float, required=True, help="USD cap over all runs (OUT/ledger.jsonl)")
    run_parser.add_argument("--n", type=int, default=30, help="number of eligible query papers")
    run_parser.add_argument("--model", default="gpt-6-luna", help="reranker model")
    run_parser.add_argument("--db", default="rerank_eval", help="scratch database (see above)")
    analyze_parser = commands.add_parser("analyze", help="metrics from the results (no API calls)")
    analyze_parser.add_argument("--out", required=True)
    analyze_parser.add_argument("--model", nargs="+", default=["gpt-6-luna"], help="reranker model(s)")
    args = parser.parse_args()
    run(args) if args.command == "run" else analyze(args)


if __name__ == "__main__":
    main()
