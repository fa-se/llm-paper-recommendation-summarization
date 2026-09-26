"""The publications the pipeline passes around. OpenAlex calls publications "works"; a Publication is a stored work."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Self

import pyalex

from core.sqlalchemy_models import Publication


# equality and hash by id only
@dataclass(frozen=True)
class Work:
    id: int  # OpenAlex ID without the "W" prefix
    title: str | None = field(compare=False)
    authors: list[str] = field(compare=False)  # the first three
    abstract: str | None = field(compare=False)
    publication_date: datetime = field(compare=False)  # UTC

    @classmethod
    def from_openalex(cls, work: pyalex.Work) -> Self:
        return cls(
            # OpenAlex IDs are URLs, e.g. "https://openalex.org/W12345"
            id=int(work["ids"]["openalex"].split("W")[-1]),
            title=work["title"],
            authors=[authorship["author"]["display_name"] for authorship in work["authorships"][:3]],
            # pyalex.Work reconstructs the abstract from the inverted index in __getitem__; work.get("abstract") would
            # bypass that and return None, so check for the inverted index instead
            abstract=work["abstract"] if work["abstract_inverted_index"] else None,
            # ISO 8601 date, e.g. "2017-08-08"
            publication_date=datetime.fromisoformat(work["publication_date"]).replace(tzinfo=UTC),
        )

    @classmethod
    def from_publication(cls, publication: Publication) -> Self:
        return cls(
            id=publication.openalex_id,
            title=publication.title,
            authors=publication.authors or [],
            abstract=publication.abstract,
            publication_date=publication.publication_datetime_utc,
        )

    @property
    def openalex_url(self) -> str:
        return f"https://openalex.org/W{self.id}"

    def to_dict(self) -> dict:
        """JSON-serializable form, e.g. for trace events."""
        return {
            "id": self.id,
            "title": self.title,
            "authors": self.authors,
            "publication_date": self.publication_date.date().isoformat(),
            "abstract": self.abstract,
        }

    def __str__(self) -> str:
        return f"'{self.title}' by [{', '.join(self.authors)}] ({self.publication_date:%Y-%m}) | {self.openalex_url}"


@dataclass(frozen=True)
class SummarizedWork:
    work: Work
    summary: str
    # the filled out reasoning structure that led to the summary (key findings, methodologies, ...)
    reasoning: dict | None = None

    def __str__(self) -> str:
        return f"{self.work.title}\nSummary: {self.summary}"
