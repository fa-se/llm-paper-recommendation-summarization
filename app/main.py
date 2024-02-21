import typer

from open_alex_interface import Work, search_works
from relevance import rate_relevance
from summary import summarize_work

app = typer.Typer()


def search(query: str):
    return search_works(query)


def rate(works: [Work]):
    for work in works:
        work.relevance_score = rate_relevance(work)
    return works


def summarize(works: [Work], query: str):
    for work in works:
        work.summary = summarize_work(work, query)
    return works

@app.command()
def search_rate_summarize(query: str, limit: int = typer.Option(3, help="Summarize only the n most relevant works")):
    works = search(query)

    rated_works = rate(works)
    # sort by relevance score (descending)
    rated_works = sorted(rated_works, key=lambda w: w.relevance_score, reverse=True)
    print(f"Top {limit} relevant works:")
    for work in rated_works[:limit]:
        print(f"{work} - relevance score: {work.relevance_score}")

    print(f"\nSummarizing top {limit} works:")
    summarized_works = summarize(rated_works[:limit], query)
    for work in summarized_works:
        print(f"{work}\nSummary: {work.summary}\n")


if __name__ == "__main__":
    app()