import datetime
from functools import total_ordering
from typing import Self

import pyalex


class Work:
    def __init__(self, pyalex_work: pyalex.Work):
        self.id = int(
            pyalex_work["ids"]["openalex"].split("W")[-1]
        )  # OpenAlex returns ids as URL in the form 'https://openalex.org/W12345'
        self.title = pyalex_work["title"]
        # First three authors for now
        self.authors = [author["author"]["display_name"] for author in pyalex_work["authorships"][:3]]
        # Pyalex overrides __getitem__ to reconstruct the abstract from the inverted index on the fly.
        # Calling pyalex_work.get would circumvent that custom logic and return None, even if an abstract is present.
        # For that reason, we need to check if abstract_inverted_index exists.
        self.abstract = pyalex_work["abstract"] if pyalex_work["abstract_inverted_index"] else None

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
        self.publication_date = datetime.datetime.fromisoformat(pyalex_work["publication_date"]).replace(
            tzinfo=datetime.timezone.utc
        )
        self.created_date = datetime.datetime.fromisoformat(pyalex_work["created_date"]).replace(
            tzinfo=datetime.timezone.utc
        )

    def openalex_url(self) -> str:
        return f"https://openalex.org/W{self.id}"

    def __str__(self) -> str:
        return f"'{self.title}' by [{', '.join(self.authors)}] ({self.openalex_url()})"


@total_ordering
class ScoredWork:
    def __init__(self, work: Work, score: float):
        self.work = work
        self.score = score

    @property
    def score(self) -> float:
        return self._score

    @score.setter
    def score(self, value: float):
        if not isinstance(value, float) or value < 0.0 or value > 1.0:
            raise ValueError("Score must be a float.")
        self._score = value

    def __str__(self) -> str:
        return f"Score: {self.score} - {self.work}"

    def __eq__(self, other: Self):
        return self.score == other.score

    def __lt__(self, other: Self):
        return self.score < other.score
