import datetime
from typing import Annotated, Optional

import typer

from core.dataclasses.data_classes import Work
from core.services.publication_service import PublicationService
from core.services.user_service import UserService
from core.sqlalchemy_models import UserConfigTopicAssociation
from utils.decorator import handle_db_exceptions

# Define subcommands for the user command to manage users
publication_commands = typer.Typer()


@publication_commands.command()
@handle_db_exceptions  # TODO: this shouldn't be necessary anymore and be handled in the services / repositories
def initialize(context: typer.Context, user_name: str, start_date: datetime.datetime) -> None:
    user_service: UserService = context.obj.user_service
    publication_service: PublicationService = context.obj.publication_service

    typer.echo(f"Initializing publications for user {user_name} starting from {start_date}...")
    added_works = publication_service.initialize_for_user(user_name, start_date)
    typer.echo(f"Added {len(added_works)} works. Works that have been seen before are ignored.")


@publication_commands.command()
def most_relevant(
    context: typer.Context,
    user_name: str,
    n: Annotated[int, typer.Argument()] = 10,
    start_date: Annotated[Optional[datetime.datetime], typer.Argument] = None,
) -> None:
    publication_service: PublicationService = context.obj.publication_service

    typer.echo(f"Getting the {n} most relevant works for user {user_name}...")
    scored_works = publication_service.get_relevant_works_for_user(user_name, n, start_date)
    for scored_work in scored_works:
        typer.echo(scored_work)


@publication_commands.command()
def search(
    context: typer.Context,
    query: str,
    n: Annotated[int, typer.Argument()] = 10,
    start_date: Annotated[Optional[datetime.datetime], typer.Argument] = None,
) -> None:
    publication_service: PublicationService = context.obj.publication_service

    typer.echo(f"Getting the {n} most relevant works for query '{query}'...")
    works = publication_service.get_relevant_works_for_query(query, n, start_date)
    for work in works:
        typer.echo(work)


@publication_commands.command()
def summarize(context: typer.Context, user_name: str, openalex_id: str) -> None:
    publication_service: PublicationService = context.obj.publication_service

    work = publication_service.get_work_by_openalex_id(openalex_id)
    summarized_work = publication_service.summarize_works_for_user(user_name, [work])[0]
    typer.echo(f"Summary for work {summarized_work.work}:")
    typer.echo(summarized_work.summary)


@publication_commands.command()
def rate_all_published_after(context: typer.Context, user_name: str, date: datetime.datetime) -> None:
    user_service: UserService = context.obj.user_service
    publication_service: PublicationService = context.obj.publication_service

    topics: list[UserConfigTopicAssociation] = user_service.followed_topics(user_name)
    topic_ids = [topic.topic_id for topic in topics]

    typer.echo(f"Getting works published after {date}...")
    works: list[Work] = publication_service.get_works_by_topics(topic_ids, published_after=date, require_abstract=True)
    typer.echo(f"Found {len(works)} works. Works without abstract are ignored.")

    typer.echo("Proceeding with scoring by embedding similarity...")
    scored_works = publication_service.compute_relevance_score_by_embedding_similarity(user_name, works)
    typer.echo(f"Scored {len(scored_works)} works.")

    # sort rated works by score
    scored_works.sort(reverse=True)

    typer.echo("\nThe 10 most relevant works are:")
    for scored_work in scored_works[:10]:
        typer.echo(scored_work)

    typer.echo("\nSummarizing the 3 most relevant works...")
    # extract the top 3 works
    most_relevant_works = [scored_work.work for scored_work in scored_works[:3]]
    summarized_works = publication_service.summarize_works_for_user(user_name, most_relevant_works)
    typer.echo("Summaries:")
    for summarized_work in summarized_works:
        typer.echo("[" + summarized_work.__str__() + ";]\n")
