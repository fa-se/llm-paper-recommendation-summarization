"""Runs the whole pipeline for a demo query under a Trace, prints time, tokens and cost per stage, and saves the events.

The saved trace can be replayed without DB or API keys (core.instrumentation.replay), e.g. as the demo's fallback.
Usage (with a running db):
    uv run --env-file .env setup/trace_run.py rag_hallucinations [--ingest] [--summaries 3] [--out traces/rag.json]
"""

import argparse
import logging
from pathlib import Path

from core import retrieval, summarization
from core.instrumentation import Trace
from setup.demo_queries import DEMO_QUERIES, START_DATE


def print_stages(trace: Trace):
    print(f"{'stage':<10} {'seconds':>8} {'calls':>6} {'tokens in':>10} {'tokens out':>10} {'cost $':>8}")
    for event in trace.events:
        if event["type"] == "stage_end":
            print(
                f"{event['stage']:<10} {event['duration_s']:>8.2f} {event['llm_calls']:>6} {event['input_tokens']:>10} "
                f"{event['output_tokens']:>10} {event['cost_usd']:>8.4f}"
            )
    usage = trace.usage
    print(
        f"{'total':<10} {trace.events[-1]['t']:>8.2f} {usage['llm_calls']:>6} {usage['input_tokens']:>10} "
        f"{usage['output_tokens']:>10} {usage['cost_usd']:>8.4f}"
    )


def main(name: str, ingest: bool, n: int, summaries: int, out: str | None):
    query = DEMO_QUERIES[name]
    with Trace() as trace:
        trace.emit("run_start", query_name=name, query=query, start_date=START_DATE.date().isoformat(), n=n)
        if ingest:
            retrieval.initialize_for_query(query, START_DATE, limit=2000, num_topics=10)
        works = retrieval.get_relevant_works_for_query(query, n=n, start_date=START_DATE)
        if summaries:
            summarization.summarize_works_for_query(query, works[:summaries])
        trace.emit("run_end", top=[work.id for work in works[:n]])

    print_stages(trace)
    for work in works[:n]:
        print(f"  {work.title}")
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        trace.save(out)
        print(f"Saved {len(trace.events)} events to {out}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("query", choices=DEMO_QUERIES)
    parser.add_argument("--ingest", action="store_true", help="also match topics and fetch/embed new works")
    parser.add_argument("-n", type=int, default=5, help="number of works to rerank into the top")
    parser.add_argument("--summaries", type=int, default=0, help="number of top works to summarize")
    parser.add_argument("--out", help="where to save the trace (JSON)")
    args = parser.parse_args()
    main(args.query, args.ingest, args.n, args.summaries, args.out)
