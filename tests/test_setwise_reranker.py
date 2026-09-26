"""Checks that the concurrent setwise heapsort ranks exactly like the sequential one of llm-rankers.

Run from the repo root: uv run python -m unittest tests.test_setwise_reranker (no API calls, no DB)
"""

import hashlib
import random
import threading
import time
import unittest
from types import SimpleNamespace

from core.instrumentation import Trace
from core.llm_interfaces.base import LLMInterface
from core.services.setwise_reranker import SetwiseHeapsortReranker


def arbitrary_choice(passages: list[str]) -> int:
    """A deterministic but intransitive judge: any heapsort that asks the same questions gets the same answers."""
    return int(hashlib.sha1("|".join(passages).encode()).hexdigest(), 16) % len(passages)


def consistent_choice(passages: list[str]) -> int:
    """A judge that knows the true relevance: the number in the passage."""
    return max(range(len(passages)), key=lambda i: int(passages[i].split()[1]))


class FakeLLM(LLMInterface):
    def __init__(self, choose):
        self.choose = choose
        self.calls = 0
        self.lock = threading.Lock()

    def handle_task(self, task) -> str:
        with self.lock:
            self.calls += 1
        time.sleep(random.uniform(0, 0.002))  # shuffle the order in which concurrent calls finish
        return f"Passage {task.labels[self.choose(task.passages)]}"


def reference_heapsort(passages: list[str], k: int, choose, num_child: int = 2) -> list[str]:
    """SetwiseLlmRanker.heapify/heapSort of llm-rankers, with compare() replaced by choose()."""
    arr = list(passages)

    def heapify(n, i):
        if num_child * i + 1 < n:
            inds = [i] + list(range(num_child * i + 1, min(num_child * (i + 1) + 1, n)))
            largest = inds[choose([arr[j] for j in inds])]
            if largest != i:
                arr[i], arr[largest] = arr[largest], arr[i]
                heapify(n, largest)

    n = len(arr)
    ranked = 0
    for i in range(n // num_child, -1, -1):
        heapify(n, i)
    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        ranked += 1
        if ranked == k:
            break
        heapify(i, 0)
    return list(reversed(arr))[:k]


def make_works(n: int, seed: int) -> list[SimpleNamespace]:
    relevance = list(range(n))
    random.Random(seed).shuffle(relevance)
    return [SimpleNamespace(id=i, abstract=f"doc {relevance[i]}") for i in range(n)]


class SetwiseHeapsortRerankerTest(unittest.TestCase):
    def test_same_ranking_as_llm_rankers(self):
        for n, k, lookahead, num_child, seed in [
            (50, 5, 0, 2, 0),
            (50, 5, 1, 2, 1),
            (50, 5, 3, 2, 2),
            (50, 5, 10, 2, 3),
            (100, 10, 1, 2, 4),
            (37, 37, 1, 2, 5),
            (20, 5, 1, 3, 6),
            (2, 5, 1, 2, 7),
            (1, 5, 1, 2, 8),
        ]:
            with self.subTest(n=n, k=k, lookahead=lookahead, num_child=num_child):
                works = make_works(n, seed)
                reranker = SetwiseHeapsortReranker(FakeLLM(arbitrary_choice), num_child=num_child, lookahead=lookahead)
                result = reranker.rerank("query", works, k)
                expected = reference_heapsort([work.abstract for work in works], k, arbitrary_choice, num_child)
                self.assertEqual([work.abstract for work in result[: len(expected)]], expected)
                # the rest keeps its original order
                self.assertEqual(sorted(work.id for work in result), list(range(n)))

    def test_finds_true_top_k(self):
        works = make_works(50, seed=9)
        result = SetwiseHeapsortReranker(FakeLLM(consistent_choice)).rerank("query", works, 5)
        self.assertEqual([work.abstract for work in result[:5]], [f"doc {r}" for r in range(49, 44, -1)])

    def test_lookahead_trades_calls_for_latency(self):
        works = make_works(50, seed=10)
        sequential_calls = FakeLLM(arbitrary_choice)
        SetwiseHeapsortReranker(sequential_calls, lookahead=0).rerank("query", works, 5)
        speculative_calls = FakeLLM(arbitrary_choice)
        SetwiseHeapsortReranker(speculative_calls, lookahead=2).rerank("query", works, 5)
        self.assertGreater(speculative_calls.calls, sequential_calls.calls)

    def test_events_reconstruct_the_heap(self):
        works = make_works(50, seed=11)
        with Trace() as trace:
            result = SetwiseHeapsortReranker(FakeLLM(arbitrary_choice)).rerank("query", works, 5)
        heap = next(event for event in trace.events if event["type"] == "heap_init")["docids"]
        ranked = []
        for event in trace.events:
            if event["type"] == "swap":
                a, b = event["positions"]
                heap[a], heap[b] = heap[b], heap[a]
            elif event["type"] == "ranked":
                self.assertEqual(heap[event["position"]], event["docid"])
                ranked.append(event["docid"])
        self.assertEqual(ranked, [work.id for work in result[:5]])
        starts = [event for event in trace.events if event["type"] == "compare_start"]
        ends = [event for event in trace.events if event["type"] == "compare_end"]
        self.assertEqual(sorted(e["call"] for e in starts), sorted(e["call"] for e in ends))


if __name__ == "__main__":
    unittest.main()
