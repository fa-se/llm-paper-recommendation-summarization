import datetime
import logging
from itertools import chain
from os import environ

import pyalex

logger = logging.getLogger(__name__)
pyalex.config.email = environ.get("OPENALEX_CONTACT_EMAIL")


class Work:
    def __init__(self, pyalex_work: pyalex.Work):
        self.id = int(
            pyalex_work["ids"]["openalex"].split("W")[-1]
        )  # OpenAlex returns ids as URL in the form 'https://openalex.org/W12345'
        self.title = pyalex_work["title"]
        # First three authors for now
        self.authors = [
            author["author"]["display_name"]
            for author in pyalex_work["authorships"][:3]
        ]
        # Pyalex overrides __getitem__ to reconstruct the abstract from the inverted index on the fly.
        # Calling pyalex_work.get would circumvent that custom logic and return None, even if an abstract is present.
        # For that reason, we need to check if abstract_inverted_index exists.
        self.abstract = (
            pyalex_work["abstract"] if pyalex_work["abstract_inverted_index"] else None
        )

        self.topics = {}
        for topic in pyalex_work["topics"]:
            topic_id = int(
                topic["id"].split("T")[-1]
            )  # OpenAlex returns ids as URL in the form 'https://openalex.org/T12345'
            self.topics[topic_id] = {
                "name": topic["display_name"],
                "score": topic["score"],
            }
        # Pyalex returns dates as ISO 8601 strings (e.g. "2017-08-08"), we want to store them as UTC datetime objects
        self.publication_date = datetime.datetime.fromisoformat(
            pyalex_work["publication_date"]
        ).replace(tzinfo=datetime.timezone.utc)
        self.created_date = datetime.datetime.fromisoformat(
            pyalex_work["created_date"]
        ).replace(tzinfo=datetime.timezone.utc)

    def __str__(self) -> str:
        return f"'{self.title}' by [{', '.join(self.authors)}] ({self.id})"


def search_works(query: str) -> [Work]:
    pyalex_works = pyalex.Works().search(query).sort(relevance_score="desc").get()

    return [Work(pyalex_work) for pyalex_work in pyalex_works]


def get_works_by_topics(
    topic_ids: [int], published_after: datetime.date, n_max: int = 2000
) -> list[Work]:
    topic_filter_str = "|".join(f"T{str(topic_id)}" for topic_id in topic_ids)
    # ISO 8601 date format
    from_publication_date_str = published_after.strftime("%Y-%m-%d")
    to_publication_date_str = datetime.datetime.now().strftime("%Y-%m-%d")

    query = (
        pyalex.Works()
        .filter(primary_topic={"id": topic_filter_str})
        .filter(from_publication_date=from_publication_date_str)
        .filter(to_publication_date=to_publication_date_str)
    )

    logger.info(
        f"Querying OpenAlex for works with topics {topic_ids} published after {published_after} (n_max={n_max}) Query URL: {query.url}"
    )

    works = []
    for pyalex_work in chain(*query.paginate(per_page=200, n_max=n_max)):
        works.append(Work(pyalex_work))

    return works
