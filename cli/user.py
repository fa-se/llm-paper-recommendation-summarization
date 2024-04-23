from typing import Annotated, Optional

import typer

from core.services.user_service import UserService
from utils.decorator import handle_db_exceptions

# Define subcommands for the user command to manage users
user_commands = typer.Typer()


@user_commands.command()
@handle_db_exceptions
def create(
    context: typer.Context,
    user_name: str,
    display_name: str,
    email: Annotated[Optional[str], typer.Argument()] = None,
) -> int:
    user_service: UserService = context.obj.user_service
    user = user_service.create_user(user_name, display_name, email)
    typer.echo(f"User {user_name} created with ID {user.id}.")
    return user.id


@user_commands.command()
def delete(context: typer.Context, user_name: str) -> None:
    user_service: UserService = context.obj.user_service
    user_service.delete_user(user_name)
    typer.echo(f"User {user_name} deleted.")


@user_commands.command()
@handle_db_exceptions
def remove_topic(context: typer.Context, user_name: str, topic_id: int) -> None:
    user_service: UserService = context.obj.user_service
    user_service.remove_followed_topic(user_name, topic_id)
    typer.echo(f"Successfully removed topic {topic_id} from user {user_name}.")


@user_commands.command()
@handle_db_exceptions
def followed_topics(context: typer.Context, user_name: str) -> None:
    user_service: UserService = context.obj.user_service
    associations = user_service.followed_topics(user_name)
    for association in associations:
        typer.echo(
            f"Id: {association.topic.id}, Name: {association.topic.name}, Relevance Score: {association.relevance_score:.2f}"
        )


@user_commands.command()
@handle_db_exceptions
def set_area_of_interest(
    context: typer.Context,
    user_name: str,
    area_of_interest_description: Annotated[
        str,
        typer.Argument(..., help="A natural language description of the user's area of interest."),
    ],
    dont_align: bool = typer.Option(
        False,
        "--no-align",
        help="Don't rewrite the description to improve topic matching.",
    ),
) -> None:
    align = not dont_align  # Flip the value, because the feature should be enabled by default
    user_service: UserService = context.obj.user_service
    topic_associations, description = user_service.set_area_of_interest(user_name, area_of_interest_description, align)

    typer.echo(f"Successfully set the area of interest for user {user_name}.")
    if align:
        typer.echo(f"Aligned description: {description}")
    typer.echo(f"The following topics match the description and have been added to {user_name}'s followed topics:")
    for association in topic_associations:
        typer.echo(
            f"Id: {association.topic.id}, Name: {association.topic.name}, Relevance Score: {association.relevance_score:.2f}"
        )
    typer.echo("\nIf you want to remove a topic, use the 'user remove-topic' command.")
