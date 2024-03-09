import typer

from db.database import Session
from llm_interfaces import LLMInterface, OpenAIInterface
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


@app.command()
def get_matching_topics(
        ctx: typer.Context,
        research_area_description: str = typer.Argument(..., help="A string describing the user's area of research.")
):
    llm_interface: LLMInterface = ctx.obj
    query_embedding = llm_interface.create_embedding(research_area_description)

    from sqlalchemy import select, desc
    from db.models import Topic

    with Session() as session:
        statement = select(Topic, (1 - Topic.embedding.cosine_distance(query_embedding)).label(
            'cosine_similarity')).order_by(desc('cosine_similarity')).limit(5)
        results = session.execute(statement).fetchall()

    for result in results:
        typer.echo(f"{result.Topic.name}: {result.cosine_similarity}")


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    if ctx.invoked_subcommand is None:
        typer.echo("Please specify a subcommand.")
        raise typer.Exit(1)
    # instantiate LLMInterface, and add it to context so that it's available in command handlers
    llm_interface: LLMInterface = OpenAIInterface()
    ctx.obj = llm_interface  # TODO: define custom class for context or use ctx.params


if __name__ == "__main__":
    app()