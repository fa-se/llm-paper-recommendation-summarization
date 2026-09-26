import logging

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, scoped_session, sessionmaker

from core.config import Settings


def create_session(settings: Settings) -> scoped_session[Session]:
    """A thread-local session: each thread that uses it gets a session of its own, since a Session must not be shared
    between threads (e.g. two requests of a web app that run the pipeline at the same time)."""
    engine = create_engine(settings.database_url)
    if settings.debug:
        logging.getLogger("sqlalchemy.engine").setLevel(logging.INFO)
    return scoped_session(sessionmaker(bind=engine))
