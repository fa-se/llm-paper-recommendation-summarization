import logging
import os
from dataclasses import dataclass, field

import typer

from core.llm_interfaces import OpenAIInterface, LLMInterface
from core.repositories import UserRepository, TopicRepository
from core.repositories.publication_repository import PublicationRepository
from core.services.publication_service import PublicationService
from core.services.user_service import UserService
from db import Session
from publication import publication_commands
from user import user_commands

logger = logging.getLogger(__name__)
app = typer.Typer()
# add "user" subcommands
app.add_typer(user_commands, name="user")
app.add_typer(publication_commands, name="publication")


@dataclass
class Services:
    user_service: UserService = field(init=False)
    publication_service: PublicationService = field(init=False)

    def __post_init__(self):
        session = Session()
        llm_interface: LLMInterface = OpenAIInterface()
        self.user_service = UserService(
            user_repository=UserRepository(session),
            topic_repository=TopicRepository(session),
            llm_interface=llm_interface,
        )
        self.publication_service = PublicationService(PublicationRepository(session), self.user_service, llm_interface)


def setup_logging(is_cli: bool = False) -> None:
    filename = "app.log"
    filemode = "a"
    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    level = logging.DEBUG if os.getenv("DEBUG") == "1" else logging.INFO
    logging.basicConfig(level=level, format=fmt, filename=filename, filemode=filemode, force=True)


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    setup_logging(is_cli=True)

    if ctx.invoked_subcommand is None:
        # show help if called without subcommand
        typer.echo("No subcommand specified.")
        typer.echo(ctx.get_help())
        raise typer.Exit(1)
    # instantiate services, and add them to the context so that they're available in command handlers
    ctx.obj = Services()  # TODO: define custom class for context or use ctx.params


if __name__ == "__main__":
    app()
