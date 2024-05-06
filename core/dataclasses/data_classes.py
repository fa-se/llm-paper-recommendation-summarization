import datetime
from functools import total_ordering
from typing import Self

import pyalex


class Work:
    def __init__(self, pyalex_work: pyalex.Work):
        self.id: int = int(
            pyalex_work["ids"]["openalex"].split("W")[-1]
        )  # OpenAlex returns ids as URL in the form 'https://openalex.org/W12345'
        self.title: str = pyalex_work["title"]
        # First three authors for now
        self.authors: list[str] = [author["author"]["display_name"] for author in pyalex_work["authorships"][:3]]
        # Pyalex overrides __getitem__ to reconstruct the abstract from the inverted index on the fly.
        # Calling pyalex_work.get would circumvent that custom logic and return None, even if an abstract is present.
        # For that reason, we need to check if abstract_inverted_index exists.
        self.abstract: str = pyalex_work["abstract"] if pyalex_work["abstract_inverted_index"] else None

        self.topics: dict = {}
        for topic in pyalex_work["topics"]:
            topic_id = int(
                topic["id"].split("T")[-1]
            )  # OpenAlex returns ids as URL in the form 'https://openalex.org/T12345'
            self.topics[topic_id] = {
                "name": topic["display_name"],
                "score": topic["score"],
            }
        # Pyalex returns dates as ISO 8601 strings (e.g. "2017-08-08"), we want to store them as UTC datetime objects
        self.publication_date: datetime.datetime = datetime.datetime.fromisoformat(
            pyalex_work["publication_date"]
        ).replace(tzinfo=datetime.timezone.utc)
        self.created_date: datetime.datetime = datetime.datetime.fromisoformat(pyalex_work["created_date"]).replace(
            tzinfo=datetime.timezone.utc
        )
        self.cited_by_count = pyalex_work["cited_by_count"]

    def openalex_url(self) -> str:
        return f"https://openalex.org/W{self.id}"

    def __str__(self) -> str:
        return f"'{self.title}' by [{', '.join(self.authors)}] ({self.publication_date:%Y-%m}) | # cited by: {self.cited_by_count} | {self.openalex_url()}"


@total_ordering
class ScoredWork:
    def __init__(self, work: Work, score: float):
        self.work: Work = work
        self.score: float = score

    @property
    def score(self) -> float:
        return self._score

    @score.setter
    def score(self, value: float):
        if not isinstance(value, float) or value < 0.0 or value > 1.0:
            raise ValueError("Score must be a float.")
        self._score = value

    def __str__(self) -> str:
        return f"Score: {self.score:.2f} - {self.work}"

    def __eq__(self, other: Self):
        return self.score == other.score

    def __lt__(self, other: Self):
        return self.score < other.score


class SummarizedWork:
    def __init__(self, work: Work, summary: str):
        self.work: Work = work
        self.summary: str = summary

    def __str__(self) -> str:
        return f"{self.work.title}\nSummary: {self.summary}"
