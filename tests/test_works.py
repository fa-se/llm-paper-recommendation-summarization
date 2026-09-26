"""Run from the repo root: uv run python -m unittest tests.test_works"""

import unittest
from datetime import UTC, datetime
from types import SimpleNamespace

import pyalex

from core.works import Work

OPENALEX_WORK = {
    "id": "https://openalex.org/W4392593764",
    "ids": {"openalex": "https://openalex.org/W4392593764", "doi": "https://doi.org/10.1/x"},
    "title": "A title",
    "authorships": [{"author": {"display_name": name}} for name in ["A", "B", "C", "D"]],
    "abstract_inverted_index": {"Coral": [0], "reefs": [1], "decline.": [2]},
    "publication_date": "2024-03-05",
}


class WorkTest(unittest.TestCase):
    def test_from_openalex(self):
        work = Work.from_openalex(pyalex.Work(OPENALEX_WORK))
        self.assertEqual(work.id, 4392593764)
        self.assertEqual(work.authors, ["A", "B", "C"])
        self.assertEqual(work.abstract, "Coral reefs decline.")
        self.assertEqual(work.publication_date, datetime(2024, 3, 5, tzinfo=UTC))
        self.assertEqual(str(work), "'A title' by [A, B, C] (2024-03) | https://openalex.org/W4392593764")

    def test_without_abstract(self):
        work = Work.from_openalex(pyalex.Work({**OPENALEX_WORK, "abstract_inverted_index": None}))
        self.assertIsNone(work.abstract)

    def test_from_publication_and_to_dict(self):
        published = datetime(2025, 1, 2, tzinfo=UTC)
        publication = SimpleNamespace(
            openalex_id=1, title="T", authors=None, abstract="x", publication_datetime_utc=published
        )
        work = Work.from_publication(publication)
        self.assertEqual(
            work.to_dict(), {"id": 1, "title": "T", "authors": [], "publication_date": "2025-01-02", "abstract": "x"}
        )

    def test_equal_by_id(self):
        work = Work.from_openalex(pyalex.Work(OPENALEX_WORK))
        other_version = Work(work.id, "Other title", [], None, work.publication_date)
        self.assertEqual(work, other_version)
        self.assertEqual(len({work, other_version}), 1)


if __name__ == "__main__":
    unittest.main()
