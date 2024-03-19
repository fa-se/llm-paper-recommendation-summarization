import logging
import os

import typer

from config import user_commands
from llm_interfaces import LLMInterface, OpenAIInterface
from open_alex_interface import Work, search_works
from relevance import rate_relevance
from summary import summarize_work

logger = logging.getLogger(__name__)
app = typer.Typer()
# add "user" subcommands
app.add_typer(user_commands, name="user")

def setup_logging(is_cli: bool = False) -> None:
    filename = "app.log"
    filemode = "a"
    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    level = logging.DEBUG if os.getenv("DEBUG") else logging.INFO
    logging.basicConfig(level=level, format=fmt, filename=filename, filemode=filemode, force=True)


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




@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    setup_logging(is_cli=True)

    if ctx.invoked_subcommand is None:
        # show help if called without subcommand
        typer.echo("No subcommand specified.")
        typer.echo(ctx.get_help())
        raise typer.Exit(1)
    # instantiate LLMInterface, and add it to context so that it's available in command handlers
    llm_interface: LLMInterface = OpenAIInterface()
    ctx.obj = llm_interface  # TODO: define custom class for context or use ctx.params


if __name__ == "__main__":
    app()
