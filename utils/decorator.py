import logging
import os
from functools import wraps

import typer

logger = logging.getLogger(__name__)


def handle_db_exceptions(func):
    from sqlalchemy.exc import IntegrityError, DatabaseError, NoResultFound, SQLAlchemyError
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except IntegrityError as ie:
            logger.error(ie, exc_info=True)
            typer.echo(f"Database integrity error: An entity with the same key might already exist.")
            raise typer.Exit(code=-1)
        except DatabaseError as de:
            logger.error(de, exc_info=True)
            typer.echo(f"Database error. Please check the logs for more information.")
            raise typer.Exit(code=-1)
        except NoResultFound as nrf:
            logger.error(nrf, exc_info=True)
            typer.echo(f"No result found. This can happen if the entity you are trying to access does not exist.")
            raise typer.Exit(code=-1)
        # catch all SQLAlchemyErrors unless in debug mode
        except SQLAlchemyError as se:
            if os.environ.get("DEBUG"):
                raise se
            logger.error(se, exc_info=True)
            typer.echo(f"An error occurred. Please check the logs for more information.")
            raise typer.Exit(code=-1)

    return wrapper
