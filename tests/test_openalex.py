import unittest
from datetime import UTC, datetime
from unittest import mock

from core import openalex
from core.works import Work


def work(n: int) -> Work:
    return Work(id=n, title=f"t{n}", abstract="a", authors=[], publication_date=datetime(2026, 9, 1, tzinfo=UTC))


class FetchPerTopicTest(unittest.TestCase):
    def test_fetches_each_topic_with_its_own_limit(self):
        calls = []

        def fake(topic_ids, start_date, limit):
            calls.append((tuple(topic_ids), limit))
            return [work(topic_ids[0] * 100 + i) for i in range(3)]

        with mock.patch.object(openalex, "fetch_works_by_topics", side_effect=fake):
            result = openalex.fetch_works_per_topic([1, 2, 3], datetime(2025, 1, 1), per_topic=3)

        self.assertEqual(sorted(calls), [((1,), 3), ((2,), 3), ((3,), 3)])
        self.assertEqual(list(result), [1, 2, 3])  # in the order of the topics
        self.assertEqual([w.id for w in result[2]], [200, 201, 202])


if __name__ == "__main__":
    unittest.main()
