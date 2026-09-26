"""Run from the repo root: python -m unittest tests.test_embedding_batches"""

import unittest
from types import SimpleNamespace

import tiktoken

from core.llm_interfaces.openai import OpenAIInterface

ENCODING = tiktoken.get_encoding("cl100k_base")


class FakeEmbeddings:
    def __init__(self):
        self.requests: list[list[str]] = []

    def create(self, input, model, dimensions):
        self.requests.append(list(input))
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[float(len(text))]) for text in input],
            usage=SimpleNamespace(total_tokens=sum(len(ENCODING.encode(text)) for text in input)),
        )


class EmbeddingBatchTest(unittest.TestCase):
    def setUp(self):
        self.interface = OpenAIInterface()
        self.fake = FakeEmbeddings()
        self.interface.client = SimpleNamespace(embeddings=self.fake)
        self.interface.embedding_max_tokens_per_input = 20
        self.interface.embedding_max_inputs_per_request = 3
        self.interface.embedding_max_request_size = 30

    def test_requests_respect_the_limits_and_keep_the_order(self):
        texts = [" ".join(["hello"] * n) for n in (5, 12, 1, 9, 9, 9, 2, 2, 2, 2)]
        embeddings = self.interface.create_embedding_batch(texts)
        self.assertEqual(embeddings, [[float(len(text))] for text in texts])
        for request in self.fake.requests:
            self.assertLessEqual(len(request), 3)
            self.assertLessEqual(sum(-(-len(text.encode()) // 4) for text in request), 30)
        # greedy packing by size estimate (bytes / 4): [8, 18, 2] (input limit), [14, 14], [14, 3, 3] (input limit), [3, 3]
        self.assertEqual([len(request) for request in self.fake.requests], [3, 2, 3, 2])

    def test_long_texts_are_truncated_not_dropped(self):
        embeddings = self.interface.create_embedding_batch(["short", " ".join(["hello"] * 50)])
        self.assertEqual(len(embeddings), 2)
        truncated = [text for request in self.fake.requests for text in request][1]
        self.assertEqual(len(ENCODING.encode(truncated)), 20)
        self.assertTrue(" ".join(["hello"] * 50).startswith(truncated))


if __name__ == "__main__":
    unittest.main()
