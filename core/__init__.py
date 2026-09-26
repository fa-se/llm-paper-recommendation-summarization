"""Paper recommendation and tailored summaries from a free-form description of a research interest.

    from core import retrieval, summarization

    topics, added = retrieval.ingest(query, start_date, limit=2000, num_topics=10)  # build the corpus
    works = retrieval.search(query, n=5, start_date=start_date)  # hybrid search + setwise LLM reranking
    summaries = summarization.summarize(query, works[:3])  # summaries tailored to the query

retrieval and summarization are created on first use, from the environment (core/config.py). So importing core needs
neither a database nor an API key; create_services() builds a separate set, e.g. with other settings.
"""

from functools import cache
from typing import NamedTuple

import pyalex

from core.config import Settings
from core.database import create_session
from core.llm_interfaces import OpenAIInterface
from core.repositories.publication_repository import PublicationRepository
from core.repositories.topic_repository import TopicRepository
from core.services.retrieval_service import RetrievalService
from core.services.summarization_service import SummarizationService


class Services(NamedTuple):
    retrieval: RetrievalService
    summarization: SummarizationService


def create_services(settings: Settings | None = None) -> Services:
    settings = settings or Settings.from_env()
    pyalex.config.email = settings.openalex_contact_email
    session = create_session(settings)
    llm = OpenAIInterface(settings)
    return Services(
        retrieval=RetrievalService(PublicationRepository(session), TopicRepository(session), llm),
        summarization=SummarizationService(llm),
    )


@cache
def _default_services() -> Services:
    return create_services()


# declared for type checkers and IDEs; the values come from __getattr__
retrieval: RetrievalService
summarization: SummarizationService


def __getattr__(name: str):
    # module attributes retrieval and summarization (PEP 562), created on first access instead of at import time
    if name in Services._fields:
        return getattr(_default_services(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Services", "create_services", "retrieval", "summarization"]
