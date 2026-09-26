"""End-to-end smoke test: ingests 100 works for a query, then searches, reranks and summarizes.

Tests the database with its extensions, the OpenAI and OpenAlex APIs. The ingested works stay in the database.
Usage (with a running db): uv run --env-file .env scripts/smoke_test.py "llm rerankers"
"""

import argparse
from datetime import datetime

from core import retrieval, summarization


def main(query: str):
    start_date = datetime(2024, 1, 1)
    retrieval.ingest(query, start_date, limit=100, num_topics=10)

    works = retrieval.search(query, n=5, start_date=start_date)
    print("Top 5 relevant works:")
    for work in works[:5]:
        print(work)

    print("\nSummaries for top 3 works:")
    for summarized_work in summarization.summarize(query, works[:3]):
        print(summarized_work)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("query", help="free-form description of a research interest")
    main(parser.parse_args().query)
