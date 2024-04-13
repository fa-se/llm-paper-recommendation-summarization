import logging
import os

import typer

from config import user_commands
from llm_interfaces import LLMInterface, OpenAIInterface

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
