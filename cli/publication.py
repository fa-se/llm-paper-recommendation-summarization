import datetime

import typer

from core.dataclasses.data_classes import Work
from core.services.publication_service import PublicationService
from core.services.user_service import UserService
from core.sqlalchemy_models import UserConfigTopicAssociation
from utils.decorator import handle_db_exceptions

# Define subcommands for the user command to manage users
publication_commands = typer.Typer()


@publication_commands.command()
@handle_db_exceptions
def rate_all_published_after(context: typer.Context, user_name: str, date: datetime.datetime) -> None:
    user_service: UserService = context.obj.user_service
    publication_service: PublicationService = context.obj.publication_service

    topics: list[UserConfigTopicAssociation] = user_service.followed_topics(user_name)
    topic_ids = [topic.topic_id for topic in topics]

    typer.echo(f"Getting works published after {date}...")
    works: list[Work] = publication_service.get_works_by_topics(topic_ids, published_after=date)
    typer.echo(f"Found {len(works)} works.")

    typer.echo("Proceeding with scoring by embedding similarity...")
    scored_works = publication_service.compute_relevance_score_by_embedding_similarity(user_name, works)
    typer.echo(f"Scored {len(scored_works)} works. Works without abstracts are ignored.")

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
