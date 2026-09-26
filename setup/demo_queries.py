"""Demo queries for the botario team talk (2026-09-28), and a script that pre-ingests their corpus.

Pre-ingesting before the demo avoids the minutes-long OpenAlex fetch + embedding step during the live run.
Usage (with a running db): uv run --env-file .env setup/demo_queries.py [--limit 2000] [--measure]
"""

import argparse
import logging
import time
from datetime import datetime

from core import retrieval

START_DATE = datetime(2025, 1, 1)

DEMO_QUERIES = {
    # FFTD from thesis Table 5 (p.54); lets the demo reproduce the BM25 vs dense vs hybrid comparison (Tables 6-9)
    "coral_reefs": "My research investigates the impact of climate change on coral reef degradation and marine "
    "species adaptation. I am working on computational modeling to predict future changes. I follow the latest "
    "research in marine biology, climate science, and ecological modeling, and I'm also interested in conservation "
    "strategies and policies to mitigate climate change effects on marine environments.",
    "rag_hallucinations": "I work on LLM-based customer-service chatbots that answer questions from a company's own "
    "knowledge base using retrieval-augmented generation. My main concern is hallucination: answers that sound "
    "plausible but are not supported by the retrieved documents. I'm interested in methods to detect and reduce "
    "this, such as grounding answers in cited sources, better retrieval and reranking, faithfulness evaluation "
    "(including LLM-as-judge), and getting the model to abstain or ask a clarifying question when the knowledge base "
    "does not contain the answer.",
}


def measure_latency(query: str, n: int = 5):
    """Times the live part of the pipeline: hybrid search + OpenAlex hydration, then setwise reranking."""
    t = time.perf_counter()
    candidates = retrieval.get_relevant_works_for_query(query, n=n * 10, start_date=START_DATE, rerank=False)
    t_hybrid = time.perf_counter() - t
    t = time.perf_counter()
    top = retrieval._rerank(query, candidates, k=n)
    t_rerank = time.perf_counter() - t
    print(f"  hybrid search + hydration: {t_hybrid:.1f}s, rerank top {n} of {len(candidates)}: {t_rerank:.1f}s")
    for work in top[:n]:
        print(f"    {work.title}")


def main(limit: int, num_topics: int, measure: bool):
    for name, query in DEMO_QUERIES.items():
        t = time.perf_counter()
        topics, added = retrieval.initialize_for_query(query, START_DATE, limit=limit, num_topics=num_topics)
        print(f"[{name}] added {len(added)} works from {len(topics)} topics in {time.perf_counter() - t:.1f}s")
        print("  topics: " + "; ".join(topic.name for topic in topics))
        if measure:
            measure_latency(query)
    print(f"Tracked OpenAI cost: ${retrieval.llm_interface.accumulated_costs:.4f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=2000, help="newest works to fetch per query")
    parser.add_argument("--num-topics", type=int, default=10)
    parser.add_argument("--measure", action="store_true", help="also time hybrid search and reranking")
    args = parser.parse_args()
    main(args.limit, args.num_topics, args.measure)
